"""
@FileName: my_encoder.py
@Time    : 7/12/2022
@Author  : Ning Zhang
@GitHub: https://github.com/noahzn
@ver: resnet18+transformer, model5
"""

import numpy as np
from collections import OrderedDict
import torch
#import torch.nn as nn
from torch import nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from timm.models.layers import DropPath
from torchvision import transforms
from timm.models.layers import trunc_normal_
import math
import torch.cuda
from layers import *


class DistillN(nn.Module):
    def __init__(self, t_num_ch_enc, s_num_ch_enc):
        super().__init__()
        self.t_num_ch_enc = t_num_ch_enc
        self.s_num_ch_enc = s_num_ch_enc

        self.mlp = nn.ModuleList()
        """
        for i in range(len(self.s_num_ch_enc)):
            self.mlp.append(nn.Sequential(
                LayerNorm(self.t_num_ch_enc[i], eps=1e-6),
                nn.Linear(self.t_num_ch_enc[i], self.s_num_ch_enc[i]),
                # nn.GELU(),
                # nn.BatchNorm2d(self.num_ch_enc[i]),
            ))
        """

        for i in range(len(self.s_num_ch_enc)):
            self.mlp.append(nn.Sequential(
                nn.Conv2d(self.t_num_ch_enc[i], self.s_num_ch_enc[i], 1, 1, 0, bias=False),
                #nn.BatchNorm2d(self.s_num_ch_enc[i]),
                # nn.ReLU(inplace=True)
                #
            ))

        # input(self.mlp)

    def forward(self, input_features):
        # for i in input_features:
        #     print(i.shape)
        # input()
        features = []
        for i in range(len(self.t_num_ch_enc)):
            f = self.mlp[i](input_features[i])
            features.append(f)

        return features


class DistillD(nn.Module):
    def __init__(self, num_ch, size=[96, 320]):
        super().__init__()
        self.num_ch = num_ch
        self.height, self.width = size
        self.mlp = nn.ModuleList()

        for i in range(1, len(self.num_ch)):
            self.mlp.append(nn.Sequential(
                nn.Upsample([self.height, self.width], mode="bilinear", align_corners=False),
                nn.Conv2d(self.num_ch[i], self.num_ch[0], 1, 1, 0, bias=False),
            ))

        # input(self.mlp)

    def forward(self, input_features):
        # for i in input_features:
        #     print(i.shape)
        # input()
        features = []
        for i in range(1, len(self.num_ch)):
            f = self.mlp[i-1](input_features[i])
            features.append(f)

        return features


class LGFI(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.,
                 distill=False):
        super().__init__()

        self.dim = dim
        self.distill = distill

        # if self.distill:
        #     self.dim += 1
        #     self.distill_token = nn.Parameter(torch.randn(1, self.h[self.stage], self.w[self.stage]))

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=self.dim)

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.norm = LayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()  # TODO: MobileViT is using 'swish'
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_ = x
        # print(x.shape, self.dim, self.stage)

        # XCA
        B, C, H, W = x.shape
        # if self.distill:
        #     # distill_token = repeat(self.distill_token, '() 1 h w -> b 1 h w', b=B)
        #     distill_token = self.distill_token.reshape(B, 1, H, W)
        #     # print(x.shape)
        #     # input(distill_token.shape)
        #     x = torch.cat((x, distill_token), dim=1)
        #     B, C, H, W = x.shape
        #     input_ = x

        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            # print(x.shape, pos_encoding.shape)
            x = x + pos_encoding

        x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_ + self.drop_path(x)

        return x


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.ReLU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''

        output = self.conv(input)
        return output


class DilatedConv(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """
    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.ddwconv(x)
        x = self.bn1(x)
        # x = self.act(x)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


class SimpleDilatedConv(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """
    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)

        # self.norm = LayerNorm(dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        # self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # input = x

        x = self.ddwconv(x)
        x = self.bn1(x)
        # x = self.act(x)

        """
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        """

        return x


class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)

        return x


class LiteMono(nn.Module):
    """
    Lite-Mono
    """
    def __init__(self, in_chans=1, model='lite-mono', distill=True,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):
        super().__init__()

        if model == 'lite-mono':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 10]
            self.dims = [48, 80, 128]
            self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

        elif model == 'lite-mono-small':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 7]
            self.dims = [48, 80, 128]
            self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]

        elif model == 'lite-mono-tiny':
            self.num_ch_enc = np.array([32, 64, 128])
            self.depth = [4, 4, 7]
            self.dims = [32, 64, 128]
            self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]

        self.distill = distill

        for g in global_block_type:
            assert g in ['None', 'LGFI']

        self.downsample_layers = nn.ModuleList()
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        self.stem2 = nn.Sequential(
            Conv(self.dims[0] + 1, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        self.downsample_layers.append(stem1)

        self.input_downsample = nn.ModuleList()
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i] * 2 + 1, self.dims[i + 1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        stage_blocks.append(LGFI(self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 distill=self.distill))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        features_distill = []
        x = (x - 0.45) / 0.225

        x_down = []
        for i in range(4):
            x_down.append(self.input_downsample[i](x))

        tmp_x = []
        x = self.downsample_layers[0](x)
        features_distill.append(x)  # 1
        x = self.stem2(torch.cat((x, x_down[0]), dim=1))

        tmp_x.append(x)

        # for s in range(len(self.stages[0])-1):
        #     x = self.stages[0][s](x)
        # x = self.stages[0][-1](x)

        x = self.stages[0](x)
        features_distill.append(x)  # 2
        tmp_x.append(x)
        features.append(x)

        for i in range(1, 3):
            tmp_x.append(x_down[i])
            x = torch.cat(tmp_x, dim=1)
            x = self.downsample_layers[i](x)

            tmp_x = [x]

            # for s in range(len(self.stages[i]) - 1):
            #     x = self.stages[i][s](x)
            # x = self.stages[i][-1](x)
            x = self.stages[i](x)
            features_distill.append(x)  # 3 and 4
            tmp_x.append(x)
            features.append(x)
        if self.distill:
            return features, features_distill
        return features

    def forward(self, x, distill=True):
        self.distill = distill
        x = self.forward_features(x)

        return x


class DroneMono(nn.Module):
    """
    Drone-Mono
    """
    def __init__(self, in_chans=1, model='drone-mono', distill=True,
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6, **kwargs):
        super().__init__()

        if model == 'drone-mono':
            self.num_ch_enc = np.array([32, 48, 80])
            self.depth = [3, 3, 6]
            self.dims = [32, 48, 80]
            self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]

        self.distill = distill

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        self.downsample_layers.append(stem1)

        self.input_downsample = nn.ModuleList()
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))

        self.stem2 = nn.Sequential(
            Conv(self.dims[0] + 1, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i] * 2 + 1, self.dims[i + 1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                stage_blocks.append(SimpleDilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j],
                                                      drop_path=dp_rates[cur + j],
                                                      layer_scale_init_value=layer_scale_init_value,
                                                      expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.apply(self._init_weights)

        # input(self.stages)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        features_distill = []
        x = (x - 0.45) / 0.225

        x_down = []
        for i in range(4):
            x_down.append(self.input_downsample[i](x))
        tmp_x = []

        x = self.downsample_layers[0](x)
        # features_distill.append(x)

        # x = self.stem2(x)
        x = self.stem2(torch.cat((x, x_down[0]), dim=1))
        tmp_x.append(x)
        x = self.stages[0](x)

        features_distill.append(x)
        tmp_x.append(x)

        features.append(x)

        for i in range(1, 3):
            tmp_x.append(x_down[i])
            x = torch.cat(tmp_x, dim=1)
            x = self.downsample_layers[i](x)
            tmp_x = [x]
            x = self.stages[i](x)
            features_distill.append(x)
            features.append(x)
            tmp_x.append(x)

        if self.distill:
            return features, features_distill
        return features

    def forward(self, x, distill=True):
        self.distill = distill
        x = self.forward_features(x)

        return x


def disp_to_depth(disp):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 0.1
    max_disp = 10
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return depth


class DroneDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(3), num_output_channels=1, use_skips=True, onnx=False):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.onnx = onnx

        self.num_ch_enc = num_ch_enc
        # self.num_ch_dec = (self.num_ch_enc / 2).astype('int')
        self.num_ch_dec = [16, 32, 40]
        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.up = [nn.Upsample(size=(64, 80), mode="bilinear", align_corners=True),
                   nn.Upsample(size=(32, 40), mode="bilinear", align_corners=True),
                   nn.Upsample(size=(16, 20), mode="bilinear", align_corners=True)]

        # self.up = [nn.Upsample(size=(112, 160), mode="bilinear", align_corners=True),
        #            nn.Upsample(size=(56, 80), mode="bilinear", align_corners=True),
        #            nn.Upsample(size=(28, 40), mode="bilinear", align_corners=True)]

        self.up2 = [nn.Upsample(size=(128, 160), mode="bilinear", align_corners=True),
                    nn.Upsample(size=(64, 80), mode="bilinear", align_corners=True),
                    nn.Upsample(size=(32, 40), mode="bilinear", align_corners=True)]
        # self.up2 = [nn.Upsample(size=(224, 320), mode="bilinear", align_corners=True),
        #             nn.Upsample(size=(112, 160), mode="bilinear", align_corners=True),
        #             nn.Upsample(size=(56, 80), mode="bilinear", align_corners=True)]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features, distill=True):
        # self.outputs = {}
        self.distill=distill
        self.outputs = []
        out_logits = []
        features = []
        x = input_features[-1]

        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [self.up[i](x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]

            # print(x[0].shape, x[1].shape)
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            features.append(x)

            if i in self.scales:
                f = self.up2[i](self.convs[("dispconv", i)](x))

                # self.outputs[("logits", i)] = f
                # self.outputs[("disp", i)] = self.sigmoid(f)
                self.outputs.append(self.sigmoid(f))

                # print(self.sigmoid(f).shape)
                # self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        # return self.outputs
        if not self.onnx:
            return self.outputs, features
        else:
            return self.outputs[-1]


class DroneDepthDecoder_onnx(nn.Module):
    def __init__(self, num_ch_enc, scales=range(3), num_output_channels=1, use_skips=True, onnx=False):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.onnx = onnx

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 40]

        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels, use_refl=False)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        # self.up = [nn.Upsample(size=(64, 80), mode="bilinear", align_corners=True),
        #            nn.Upsample(size=(32, 40), mode="bilinear", align_corners=True),
        #            nn.Upsample(size=(16, 20), mode="bilinear", align_corners=True)]

        self.up = [nn.Upsample(size=(112, 160), mode="bilinear", align_corners=True),
                   nn.Upsample(size=(56, 80), mode="bilinear", align_corners=True),
                   nn.Upsample(size=(28, 40), mode="bilinear", align_corners=True)]

        self.up2 = nn.Upsample(size=(224, 320), mode="bilinear", align_corners=True)


    def forward(self, input_features):
        x = input_features[-1]

        for i in range(2, -1, -1):
            # print(i)
            x = self.convs[("upconv", i, 0)](x)

            x = [self.up[i](x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            # if i == 0:
        # f = self.sigmoid(upsample(self.convs[("dispconv", 0)](x), mode='bilinear'))
        f = self.convs[("dispconv", 0)](x)
        f = self.sigmoid(self.up2(f))


        return f


class DroneMono2(nn.Module):
    """
    Drone-Mono
    """
    def __init__(self, in_chans=1, model='drone-mono', distill=True, onnx=False,
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6, **kwargs):
        super().__init__()

        if model == 'drone-mono':
            self.num_ch_enc = np.array([32, 64, 80])
            self.depth = [3, 3, 6]
            self.dims = [32, 64, 80]
            self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]

        self.distill = distill
        self.onnx = onnx

        self.decoder = DroneDepthDecoder(self.num_ch_enc, range(3))

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        self.downsample_layers.append(stem1)

        self.stem2 = nn.Sequential(
            Conv(self.dims[0], self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i], self.dims[i + 1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                stage_blocks.append(SimpleDilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j],
                                                      drop_path=dp_rates[cur + j],
                                                      layer_scale_init_value=layer_scale_init_value,
                                                      expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        # self.apply(self._init_weights)

        # input(self.stages)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        features_distill = []
        # x = (x - 0.45) / 0.225

        # x_down = []
        # for i in range(4):
        #     x_down.append(self.input_downsample[i](x))
        # tmp_x = []

        x = self.downsample_layers[0](x)
        features_distill.append(x)

        # x = self.stem2(x)
        x = self.stem2(x)
        # tmp_x.append(x)
        x = self.stages[0](x)

        features_distill.append(x)
        # tmp_x.append(x)

        features.append(x)

        for i in range(1, 3):
            # tmp_x.append(x_down[i])
            # x = torch.cat(tmp_x, dim=1)
            x = self.downsample_layers[i](x)
            tmp_x = [x]
            x = self.stages[i](x)
            features_distill.append(x)
            features.append(x)
            # tmp_x.append(x)

        if not self.onnx:
            return features, features_distill
        else:
            return features

    def forward(self, x, distill=False, is_train=False):
        self.distill = distill
        outputs2 = {}
        if self.onnx:
            features = self.forward_features(x)
            outputs = self.decoder(features)
            depth = disp_to_depth(outputs)
            return depth
        else:
            features, features_distill = self.forward_features(x)
            outputs, s_dis_dec = self.decoder(features)
            for i in range(len(outputs)):
                outputs2[("disp", i)] = outputs[len(outputs)-i-1]
            return features, features_distill, outputs2, s_dis_dec


    def profile_encoder(self, x):
        self.distill = False
        return self.forward_features(x)

    def profile_decoder(self, x):
        self.distill = False
        return self.decoder(x)


def get_obst_map(depth):
    avgpool = nn.AvgPool2d(kernel_size=(20, 20), stride=20)
    # depth = depth[:, :, 54:74, :]
    depth = depth[:, :, 102:122, :]
    map_t = avgpool(depth).squeeze()
    # input(map_t.squeeze().shape)
    return map_t


class DroneMono2_onnx(nn.Module):
    """
    Drone-Mono
    """
    def __init__(self, in_chans=1, model='drone-mono', distill=True, onnx=False,
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6, **kwargs):
        super().__init__()

        if model == 'drone-mono':
            self.num_ch_enc = np.array([32, 64, 80])
            self.depth = [3, 3, 6]
            self.dims = [32, 64, 80]
            self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]

        self.distill = distill
        self.onnx = onnx

        self.decoder = DroneDepthDecoder_onnx(self.num_ch_enc, range(3), onnx=self.onnx)

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        self.downsample_layers.append(stem1)

        self.stem2 = nn.Sequential(
            Conv(self.dims[0], self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i], self.dims[i + 1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                stage_blocks.append(SimpleDilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j],
                                                      drop_path=dp_rates[cur + j],
                                                      layer_scale_init_value=layer_scale_init_value,
                                                      expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

    def forward_features(self, x):
        features = []
        # x = (x-x.min())/(x.max()-x.min())
        # x = (x - 0.45) / 0.225

        x = self.downsample_layers[0](x)

        x = self.stem2(x)
        x = self.stages[0](x)

        features.append(x)

        for i in range(1, 3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)

        if not self.onnx:
            return features
        else:
            return features

    def forward(self, x, distill=False, is_train=False):
        self.distill = distill
        outputs2 = {}
        if self.onnx:
            features = self.forward_features(x)
            outputs = self.decoder(features)
            # depth = disp_to_depth(outputs)
            obmap = get_obst_map(outputs)
            # obmap = disp_to_depth(obmap)

            return obmap
        else:
            features, features_distill = self.forward_features(x)
            outputs, s_dis_dec = self.decoder(features)
            for i in range(len(outputs)):
                outputs2[("disp", i)] = outputs[len(outputs)-i-1]
            return features, features_distill, outputs2, s_dis_dec


    def profile_encoder(self, x):
        self.distill = False
        return self.forward_features(x)

    def profile_decoder(self, x):
        self.distill = False
        return self.decoder(x)



