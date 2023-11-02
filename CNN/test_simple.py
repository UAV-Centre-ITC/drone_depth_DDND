# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR
import cv2
import heapq
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--model_path', type=str,
                        help='name of a pretrained model to use',
                        )

    parser.add_argument('--test',
                        action='store_true',
                        help='if set, read images from a .txt file',
                        )
    """
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    """
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_path is not None, \
        "You must specify the --model_path parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_path:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    # download_model_if_doesnt_exist(args.model_name)
    # model_path = os.path.join("./tmp", args.model_name)
    print("-> Loading model from ", args.model_path)
    encoder_path = os.path.join(args.model_path, "encoder.pth")
    # depth_decoder_path = os.path.join(args.model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.DroneMono2()
    # encoder = networks.LiteMono()
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    # depth_decoder = networks.MyDecoder7(
    #     num_ch_enc=encoder.num_ch_enc, scales=range(3))

    # depth_decoder = networks.DepthDecoder(
    #     num_ch_enc=encoder.num_ch_enc, scales=range(3))

    # loaded_dict = torch.load(depth_decoder_path, map_location=device)
    # filtered_dict_dec = {k: v for k, v in loaded_dict.items() if k in depth_decoder.state_dict()}
    # depth_decoder.load_state_dict(filtered_dict_dec)

    # depth_decoder.to(device)
    # depth_decoder.eval()

    args.test = True

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path) and not args.test:
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isfile(args.image_path) and args.test:
        gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        # reading images from .txt file
        paths = []
        with open(args.image_path) as f:
            filenames = f.readlines()
            for i in range(len(filenames)):
                filename = filenames[i]
                line = filename.split()

                folder = line[0]
                if len(line) == 3:
                    frame_index = int(line[1])
                    side = line[2]

                f_str = "{:010d}{}".format(frame_index, '.jpg')
                f_str = "{}{}".format(frame_index, '.jpg')

                image_path = os.path.join(
                    '/data2/zhangn/project/drone_mono/drone_data',
                    folder,
                    # "image_0{}/data".format(side_map[side]),
                    # f_str)
                    f_str)
                paths.append(image_path)

        output_directory = 'output/GCI/dronemono w kd3'

    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    pred_disps = []
    errors = []
    names = []
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('L')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)

            # features = encoder(input_image, distill=False)
            # features = encoder(input_image)
            # outputs, _ = depth_decoder(features)
            # outputs = depth_decoder(features)

            features, features_distill, outputs, s_dis_dec=encoder(input_image, distill=True)
            disp = outputs[("disp", 0)]
            # print(original_height, original_width)
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # input(os.path.splitext(image_path))
            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            output_name = os.path.splitext(image_path)[0].split('/')[-1]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            if args.pred_metric_depth:
                name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                np.save(name_dest_npy, metric_depth)
            else:
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(idx))
                # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            pred_disps.append(scaled_disp.cpu().numpy())

            # print(gt_depths[idx].shape)
            # input(scaled_disp.shape)
            # error = ev(scaled_disp.cpu().numpy().squeeze(), gt_depths[idx])
            # input(error)
            # errors.append(error[2])
            names.append(image_path)

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            disp_resized_np = scaled_disp.squeeze().cpu().numpy()
            # input(disp_resized_np.shape)
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(idx))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))

    # ind = getListMaxNumIndex(errors)

    # a = os.path.join(output_directory, "best.txt")
    # with open(a, mode='w') as f:
    #     for i in list(ind):
    #         l = names[i]+ '  '+ str(errors[i])
    #         f.write(l)
    #         f.write('\n')


    print('-> Done!')


def getListMaxNumIndex(num_list, topk=50):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    # max_num_index=map(num_list.index, heapq.nlargest(topk,num_list))
    min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))
    # print ('max_num_index:',max_num_index)

    return min_num_index

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def ev(pred_disp, gt_depth):
    gt_height, gt_width = gt_depth.shape[:2]
    # print(gt_height, gt_width, pred_disp.shape)
    pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
    pred_depth = 1 / pred_disp

    mask = np.logical_and(gt_depth > 1e-3, gt_depth < 80)

    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)


    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]

    pred_depth *= 1

    ratio = np.median(gt_depth) / np.median(pred_depth)
    # ratios.append(ratio)
    pred_depth *= ratio

    pred_depth[pred_depth < 1e-3] = 1e-3
    pred_depth[pred_depth > 80] = 80

    return compute_errors(gt_depth, pred_depth)




if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
