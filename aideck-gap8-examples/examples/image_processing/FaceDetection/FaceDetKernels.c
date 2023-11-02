#include "FaceDetKernels.h"
L1_CL_MEM AT_L1_POINTER FaceDet_L1_Memory;
L2_MEM AT_L2_POINTER FaceDet_L2_Memory;
void ResizeImage_1(
		unsigned char * In,
		unsigned char * Out)

{
	/* Shared L1: 21504 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerResizeBilinear_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _N_In, _Off_In;
	unsigned int _SN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 8]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 3072 [Tile0, 8:[64x6, 6:64x6, 64x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 8:[64x6, 6:64x6, 64x6], 1]
		Tile0: [0, 384, 384], Tile1: [384, 384, 384], Tile2; [768, 384, 384]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 79056 [Tile0, 8:[324x32, 6:324x32, 324x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 8:[324x32, 6:324x32, 324x32], 1]
		Tile0: [0, 10368, 10368], Tile1: [9720, 10368, 10368], Tile2; [19440, 10368, 10368]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Win = (unsigned int) (324);
	KerArg0->Hin = (unsigned int) (244);
	KerArg0->Wout = (unsigned int) (64);
	KerArg0->Hout = (unsigned int) (48);
	KerArg0->HTileOut = (unsigned int) (6);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=384;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0+0), 10368, 0, &DmaR_Evt1);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	for (T0Ind=0; T0Ind<8; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
		int T0Ind_Last = (T0Ind==7), T0Ind_NextLast = ((T0Ind+1)==7);
		/*================================= Prepare Tiles ===================================*/
		_SN_In = 0;
		if (!(T0Ind_Last)) {
			_N_In = _N_In + 0; _Off_In = (((331776*((T0Ind)+1)*6)>>16)*324); _SN_In = (10368); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
		if (_SN_In) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In+_Off_In), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0+10368*((T0Ind_Total+1)%2)),
					_SN_In, 0, &DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->In = (unsigned char * __restrict__) (FaceDet_L1_Memory+0+10368*((T0Ind_Total)%2));
		KerArg0->Out = (unsigned char * __restrict__) (FaceDet_L1_Memory+20736+384*((T0Ind_Total)%2));
		KerArg0->FirstLineIndex = (unsigned int) ((331776*(T0Ind)*6)>>16);
		AT_FORK(gap_ncore(), (void *) KerResizeBilinear, (void *) KerArg0);
		__CALL(KerResizeBilinear, KerArg0);
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+20736+384*((T0Ind_Total)%2)),
				_SC_Out, 1, &DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T0Ind_Last)) {
			_C_Out = _C_Out + (384); _SC_Out = (384); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void ProcessIntegralImage_1(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage)

{
	/* Shared L1: 15616 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerPrimeImage_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;
	KerProcessImage_ArgT S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: KerIn, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3072 [Tile0, 1:[64x48], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[64x48], 1]
		Tile0: [0, 3072, 3072], Tile1: [0, 3072, 3072], Tile2; [0, 3072, 3072]
	Ker Arg: KerBuffer, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 1:[64x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[64x1], 4]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: KerOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 12288 [Tile0, 1:[64x48], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[64x48], 4]
		Tile0: [0, 12288, 12288], Tile1: [0, 12288, 12288], Tile2; [0, 12288, 12288]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned int) (64);
	KerArg1->In = (unsigned char * __restrict__) (FaceDet_L1_Memory+0);
	KerArg1->W = (unsigned int) (64);
	KerArg1->H = (unsigned int) (48);
	KerArg1->IntegralImage = (unsigned int * __restrict__) (FaceDet_L1_Memory+3328);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ImageIn+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0), 3072, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read ImageIn */
	/*============================= End Read Tiles Prolog ===============================*/
	/*====================== Call Kernel LOC_LOOP_PROLOG =========================*/
	KerArg0->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+3072);
	AT_FORK(gap_ncore(), (void *) KerIntegralImagePrime, (void *) KerArg0);
	__CALL(KerIntegralImagePrime, KerArg0);
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg1->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+3072);
		AT_FORK(gap_ncore(), (void *) KerIntegralImageProcess, (void *) KerArg1);
		__CALL(KerIntegralImageProcess, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) IntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+3328), 12288, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write IntegralImage */
	/*============================ End Write Tiles Epilog ===============================*/
}
void ProcessSquaredIntegralImage_1(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage)

{
	/* Shared L1: 15616 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerPrimeImage_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;
	KerProcessImage_ArgT S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: KerIn, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3072 [Tile0, 1:[64x48], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[64x48], 1]
		Tile0: [0, 3072, 3072], Tile1: [0, 3072, 3072], Tile2; [0, 3072, 3072]
	Ker Arg: KerBuffer, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [Tile0, 1:[64x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[64x1], 4]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: KerOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 12288 [Tile0, 1:[64x48], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[64x48], 4]
		Tile0: [0, 12288, 12288], Tile1: [0, 12288, 12288], Tile2; [0, 12288, 12288]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned int) (64);
	KerArg1->In = (unsigned char * __restrict__) (FaceDet_L1_Memory+0);
	KerArg1->W = (unsigned int) (64);
	KerArg1->H = (unsigned int) (48);
	KerArg1->IntegralImage = (unsigned int * __restrict__) (FaceDet_L1_Memory+3328);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ImageIn+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0), 3072, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read ImageIn */
	/*============================= End Read Tiles Prolog ===============================*/
	/*====================== Call Kernel LOC_LOOP_PROLOG =========================*/
	KerArg0->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+3072);
	AT_FORK(gap_ncore(), (void *) KerIntegralImagePrime, (void *) KerArg0);
	__CALL(KerIntegralImagePrime, KerArg0);
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg1->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+3072);
		AT_FORK(gap_ncore(), (void *) KerSquaredIntegralImageProcess, (void *) KerArg1);
		__CALL(KerSquaredIntegralImageProcess, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) IntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+3328), 12288, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write IntegralImage */
	/*============================ End Write Tiles Epilog ===============================*/
}
void ProcessCascade_1(
		unsigned int *  __restrict__ IntegralImage,
		unsigned int *  __restrict__ SquaredIntegralImage,
		void *  cascade_model,
		int  *  __restrict__ CascadeReponse)

{
	/* Shared L1: 24904 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_IntegralImage;
	unsigned int _SN_IntegralImage;
	unsigned int _N_SquaredIntegralImage;
	unsigned int _SN_SquaredIntegralImage;
	unsigned int _C_CascadeReponse;
	unsigned int _SP_CascadeReponse, _SC_CascadeReponse;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 25]
	Ker Arg: KerII, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 25 logical tiles, 25 physical tiles
			Total Size: 12288 [Tile0, 25:[64x24, 23:64x24, 64x24], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 25:[64x24, 23:64x24, 64x24], 4]
		Tile0: [0, 6144, 6144], Tile1: [256, 6144, 6144], Tile2; [512, 6144, 6144]
	Ker Arg: KerIISQ, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 25 logical tiles, 25 physical tiles
			Total Size: 12288 [Tile0, 25:[64x24, 23:64x24, 64x24], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 25:[64x24, 23:64x24, 64x24], 4]
		Tile0: [0, 6144, 6144], Tile1: [256, 6144, 6144], Tile2; [512, 6144, 6144]
	Ker Arg: KerOut, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 25 logical tiles, 25 physical tiles
			Total Size: 4100 [Tile0, 25:[41x1, 23:41x1, 41x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 25:[41x1, 23:41x1, 41x1], 4]
		Tile0: [0, 164, 164], Tile1: [164, 164, 164], Tile2; [328, 164, 164]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) IntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0+0), 6144, 0, &DmaR_Evt1);
	_N_IntegralImage=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SquaredIntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+12288+0), 6144, 0, &DmaR_Evt2);
	_N_SquaredIntegralImage=0;
	_C_CascadeReponse=0; _SC_CascadeReponse=164;
	_SP_CascadeReponse=0;
	/*============================= End Read Tiles Prolog ===============================*/
	for (T0Ind=0; T0Ind<25; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
		int T0Ind_Last = (T0Ind==24), T0Ind_NextLast = ((T0Ind+1)==24);
		/*================================= Prepare Tiles ===================================*/
		_SN_IntegralImage = 0;
		if (!(T0Ind_Last)) {
			_N_IntegralImage = _N_IntegralImage + (256); _SN_IntegralImage = (6144); 
		}
		_SN_SquaredIntegralImage = 0;
		if (!(T0Ind_Last)) {
			_N_SquaredIntegralImage = _N_SquaredIntegralImage + (256); _SN_SquaredIntegralImage = (6144); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read IntegralImage */
		if (_SN_IntegralImage) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) IntegralImage+_N_IntegralImage), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0+6144*((T0Ind_Total+1)%2)),
					_SN_IntegralImage, 0, &DmaR_Evt1);
		}
		AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read SquaredIntegralImage */
		if (_SN_SquaredIntegralImage) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SquaredIntegralImage+_N_SquaredIntegralImage), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+12288+6144*((T0Ind_Total+1)%2)),
					_SN_SquaredIntegralImage, 0, &DmaR_Evt2);
		}
		/*============================= End Read Tiles ======================================*/
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerEvaluateCascade(
			(unsigned int * __restrict__) (FaceDet_L1_Memory+0+6144*((T0Ind_Total)%2)),
			(unsigned int * __restrict__) (FaceDet_L1_Memory+12288+6144*((T0Ind_Total)%2)),
			(unsigned int) (64),
			(unsigned int) (24),
			(void *) (cascade_model),
			(unsigned char) (24),
			(unsigned char) (24),
			(int * __restrict__) (FaceDet_L1_Memory+24576+164*((T0Ind_Total)%2))
		);
		/*================================= Write Tiles =====================================*/
		if (_SP_CascadeReponse) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write CascadeReponse */
		AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) CascadeReponse+_C_CascadeReponse), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+24576+164*((T0Ind_Total)%2)),
				_SC_CascadeReponse, 1, &DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_CascadeReponse = _SC_CascadeReponse;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_CascadeReponse = 0;
		if (!(T0Ind_Last)) {
			_C_CascadeReponse = _C_CascadeReponse + (164); _SC_CascadeReponse = (164); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write CascadeReponse */
	/*============================ End Write Tiles Epilog ===============================*/
}
void ResizeImage_2(
		unsigned char * In,
		unsigned char * Out)

{
	/* Shared L1: 21896 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerResizeBilinear_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _N_In, _Off_In;
	unsigned int _SN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 8]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 1938 [Tile0, 8:[51x5, 6:51x5, 51x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 8:[51x5, 6:51x5, 51x3], 1]
		Tile0: [0, 255, 255], Tile1: [255, 255, 255], Tile2; [510, 255, 255]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 79056 [Tile0, 8:[324x33, 6:324x33, 324x21], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 8:[324x33, 6:324x33, 324x21], 1]
		Tile0: [0, 10692, 10692], Tile1: [10044, 10692, 10692], Tile2; [20412, 10692, 10692]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Win = (unsigned int) (324);
	KerArg0->Hin = (unsigned int) (244);
	KerArg0->Wout = (unsigned int) (51);
	KerArg0->Hout = (unsigned int) (38);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=255;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0+0), 10692, 0, &DmaR_Evt1);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	for (T0Ind=0; T0Ind<8; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
		int T0Ind_Last = (T0Ind==7), T0Ind_NextLast = ((T0Ind+1)==7);
		/*================================= Prepare Tiles ===================================*/
		_SN_In = 0;
		if (!(T0Ind_Last)) {
			_N_In = _N_In + 0; _Off_In = (((419085*((T0Ind)+1)*5)>>16)*324); _SN_In = ((T0Ind_NextLast)?6804:10692); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
		if (_SN_In) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In+_Off_In), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0+10692*((T0Ind_Total+1)%2)),
					_SN_In, 0, &DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->In = (unsigned char * __restrict__) (FaceDet_L1_Memory+0+10692*((T0Ind_Total)%2));
		KerArg0->Out = (unsigned char * __restrict__) (FaceDet_L1_Memory+21384+256*((T0Ind_Total)%2));
		KerArg0->HTileOut = (unsigned int) (T0Ind_Last?3:5);
		KerArg0->FirstLineIndex = (unsigned int) ((419085*(T0Ind)*5)>>16);
		AT_FORK(gap_ncore(), (void *) KerResizeBilinear, (void *) KerArg0);
		__CALL(KerResizeBilinear, KerArg0);
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+21384+256*((T0Ind_Total)%2)),
				_SC_Out, 1, &DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T0Ind_Last)) {
			_C_Out = _C_Out + (255); _SC_Out = ((T0Ind_NextLast)?153:255); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void ProcessIntegralImage_2(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage)

{
	/* Shared L1: 9896 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerPrimeImage_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;
	KerProcessImage_ArgT S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: KerIn, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1938 [Tile0, 1:[51x38], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[51x38], 1]
		Tile0: [0, 1938, 1938], Tile1: [0, 1938, 1938], Tile2; [0, 1938, 1938]
	Ker Arg: KerBuffer, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 204 [Tile0, 1:[51x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[51x1], 4]
		Tile0: [0, 204, 204], Tile1: [0, 204, 204], Tile2; [0, 204, 204]
	Ker Arg: KerOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 7752 [Tile0, 1:[51x38], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[51x38], 4]
		Tile0: [0, 7752, 7752], Tile1: [0, 7752, 7752], Tile2; [0, 7752, 7752]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned int) (51);
	KerArg1->In = (unsigned char * __restrict__) (FaceDet_L1_Memory+0);
	KerArg1->W = (unsigned int) (51);
	KerArg1->H = (unsigned int) (38);
	KerArg1->IntegralImage = (unsigned int * __restrict__) (FaceDet_L1_Memory+2144);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ImageIn+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0), 1938, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read ImageIn */
	/*============================= End Read Tiles Prolog ===============================*/
	/*====================== Call Kernel LOC_LOOP_PROLOG =========================*/
	KerArg0->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+1940);
	AT_FORK(gap_ncore(), (void *) KerIntegralImagePrime, (void *) KerArg0);
	__CALL(KerIntegralImagePrime, KerArg0);
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg1->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+1940);
		AT_FORK(gap_ncore(), (void *) KerIntegralImageProcess, (void *) KerArg1);
		__CALL(KerIntegralImageProcess, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) IntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+2144), 7752, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write IntegralImage */
	/*============================ End Write Tiles Epilog ===============================*/
}
void ProcessSquaredIntegralImage_2(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage)

{
	/* Shared L1: 9896 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerPrimeImage_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;
	KerProcessImage_ArgT S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: KerIn, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1938 [Tile0, 1:[51x38], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[51x38], 1]
		Tile0: [0, 1938, 1938], Tile1: [0, 1938, 1938], Tile2; [0, 1938, 1938]
	Ker Arg: KerBuffer, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 204 [Tile0, 1:[51x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[51x1], 4]
		Tile0: [0, 204, 204], Tile1: [0, 204, 204], Tile2; [0, 204, 204]
	Ker Arg: KerOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 7752 [Tile0, 1:[51x38], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[51x38], 4]
		Tile0: [0, 7752, 7752], Tile1: [0, 7752, 7752], Tile2; [0, 7752, 7752]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned int) (51);
	KerArg1->In = (unsigned char * __restrict__) (FaceDet_L1_Memory+0);
	KerArg1->W = (unsigned int) (51);
	KerArg1->H = (unsigned int) (38);
	KerArg1->IntegralImage = (unsigned int * __restrict__) (FaceDet_L1_Memory+2144);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ImageIn+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0), 1938, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read ImageIn */
	/*============================= End Read Tiles Prolog ===============================*/
	/*====================== Call Kernel LOC_LOOP_PROLOG =========================*/
	KerArg0->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+1940);
	AT_FORK(gap_ncore(), (void *) KerIntegralImagePrime, (void *) KerArg0);
	__CALL(KerIntegralImagePrime, KerArg0);
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg1->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+1940);
		AT_FORK(gap_ncore(), (void *) KerSquaredIntegralImageProcess, (void *) KerArg1);
		__CALL(KerSquaredIntegralImageProcess, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) IntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+2144), 7752, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write IntegralImage */
	/*============================ End Write Tiles Epilog ===============================*/
}
void ProcessCascade_2(
		unsigned int *  __restrict__ IntegralImage,
		unsigned int *  __restrict__ SquaredIntegralImage,
		void *  cascade_model,
		int  *  __restrict__ CascadeReponse)

{
	/* Shared L1: 17184 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: KerII, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 7752 [Tile0, 1:[51x38], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[51x38], 4]
		Tile0: [0, 7752, 7752], Tile1: [0, 7752, 7752], Tile2; [0, 7752, 7752]
	Ker Arg: KerIISQ, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 7752 [Tile0, 1:[51x38], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[51x38], 4]
		Tile0: [0, 7752, 7752], Tile1: [0, 7752, 7752], Tile2; [0, 7752, 7752]
	Ker Arg: KerOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1680 [Tile0, 1:[28x15], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[28x15], 4]
		Tile0: [0, 1680, 1680], Tile1: [0, 1680, 1680], Tile2; [0, 1680, 1680]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) IntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0), 7752, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read IntegralImage */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SquaredIntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+7752), 7752, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read SquaredIntegralImage */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerEvaluateCascade(
			(unsigned int * __restrict__) (FaceDet_L1_Memory+0),
			(unsigned int * __restrict__) (FaceDet_L1_Memory+7752),
			(unsigned int) (51),
			(unsigned int) (38),
			(void *) (cascade_model),
			(unsigned char) (24),
			(unsigned char) (24),
			(int * __restrict__) (FaceDet_L1_Memory+15504)
		);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) CascadeReponse+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+15504), 1680, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write CascadeReponse */
	/*============================ End Write Tiles Epilog ===============================*/
}
void ResizeImage_3(
		unsigned char * In,
		unsigned char * Out)

{
	/* Shared L1: 22352 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerResizeBilinear_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _N_In, _Off_In;
	unsigned int _SN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 8]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 1200 [Tile0, 8:[40x4, 6:40x4, 40x2], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 8:[40x4, 6:40x4, 40x2], 1]
		Tile0: [0, 160, 160], Tile1: [160, 160, 160], Tile2; [320, 160, 160]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 79056 [Tile0, 8:[324x34, 6:324x34, 324x18], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 8:[324x34, 6:324x34, 324x18], 1]
		Tile0: [0, 11016, 11016], Tile1: [10368, 11016, 11016], Tile2; [20736, 11016, 11016]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Win = (unsigned int) (324);
	KerArg0->Hin = (unsigned int) (244);
	KerArg0->Wout = (unsigned int) (40);
	KerArg0->Hout = (unsigned int) (30);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=160;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0+0), 11016, 0, &DmaR_Evt1);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	for (T0Ind=0; T0Ind<8; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
		int T0Ind_Last = (T0Ind==7), T0Ind_NextLast = ((T0Ind+1)==7);
		/*================================= Prepare Tiles ===================================*/
		_SN_In = 0;
		if (!(T0Ind_Last)) {
			_N_In = _N_In + 0; _Off_In = (((530841*((T0Ind)+1)*4)>>16)*324); _SN_In = ((T0Ind_NextLast)?5832:11016); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
		if (_SN_In) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In+_Off_In), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0+11016*((T0Ind_Total+1)%2)),
					_SN_In, 0, &DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->In = (unsigned char * __restrict__) (FaceDet_L1_Memory+0+11016*((T0Ind_Total)%2));
		KerArg0->Out = (unsigned char * __restrict__) (FaceDet_L1_Memory+22032+160*((T0Ind_Total)%2));
		KerArg0->HTileOut = (unsigned int) (T0Ind_Last?2:4);
		KerArg0->FirstLineIndex = (unsigned int) ((530841*(T0Ind)*4)>>16);
		AT_FORK(gap_ncore(), (void *) KerResizeBilinear, (void *) KerArg0);
		__CALL(KerResizeBilinear, KerArg0);
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+22032+160*((T0Ind_Total)%2)),
				_SC_Out, 1, &DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T0Ind_Last)) {
			_C_Out = _C_Out + (160); _SC_Out = ((T0Ind_NextLast)?80:160); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void ProcessIntegralImage_3(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage)

{
	/* Shared L1: 6160 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerPrimeImage_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;
	KerProcessImage_ArgT S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: KerIn, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1200 [Tile0, 1:[40x30], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[40x30], 1]
		Tile0: [0, 1200, 1200], Tile1: [0, 1200, 1200], Tile2; [0, 1200, 1200]
	Ker Arg: KerBuffer, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 160 [Tile0, 1:[40x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[40x1], 4]
		Tile0: [0, 160, 160], Tile1: [0, 160, 160], Tile2; [0, 160, 160]
	Ker Arg: KerOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4800 [Tile0, 1:[40x30], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[40x30], 4]
		Tile0: [0, 4800, 4800], Tile1: [0, 4800, 4800], Tile2; [0, 4800, 4800]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned int) (40);
	KerArg1->In = (unsigned char * __restrict__) (FaceDet_L1_Memory+0);
	KerArg1->W = (unsigned int) (40);
	KerArg1->H = (unsigned int) (30);
	KerArg1->IntegralImage = (unsigned int * __restrict__) (FaceDet_L1_Memory+1360);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ImageIn+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0), 1200, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read ImageIn */
	/*============================= End Read Tiles Prolog ===============================*/
	/*====================== Call Kernel LOC_LOOP_PROLOG =========================*/
	KerArg0->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+1200);
	AT_FORK(gap_ncore(), (void *) KerIntegralImagePrime, (void *) KerArg0);
	__CALL(KerIntegralImagePrime, KerArg0);
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg1->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+1200);
		AT_FORK(gap_ncore(), (void *) KerIntegralImageProcess, (void *) KerArg1);
		__CALL(KerIntegralImageProcess, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) IntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+1360), 4800, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write IntegralImage */
	/*============================ End Write Tiles Epilog ===============================*/
}
void ProcessSquaredIntegralImage_3(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage)

{
	/* Shared L1: 6160 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerPrimeImage_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;
	KerProcessImage_ArgT S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: KerIn, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1200 [Tile0, 1:[40x30], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[40x30], 1]
		Tile0: [0, 1200, 1200], Tile1: [0, 1200, 1200], Tile2; [0, 1200, 1200]
	Ker Arg: KerBuffer, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 160 [Tile0, 1:[40x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[40x1], 4]
		Tile0: [0, 160, 160], Tile1: [0, 160, 160], Tile2; [0, 160, 160]
	Ker Arg: KerOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4800 [Tile0, 1:[40x30], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[40x30], 4]
		Tile0: [0, 4800, 4800], Tile1: [0, 4800, 4800], Tile2; [0, 4800, 4800]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned int) (40);
	KerArg1->In = (unsigned char * __restrict__) (FaceDet_L1_Memory+0);
	KerArg1->W = (unsigned int) (40);
	KerArg1->H = (unsigned int) (30);
	KerArg1->IntegralImage = (unsigned int * __restrict__) (FaceDet_L1_Memory+1360);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ImageIn+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0), 1200, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read ImageIn */
	/*============================= End Read Tiles Prolog ===============================*/
	/*====================== Call Kernel LOC_LOOP_PROLOG =========================*/
	KerArg0->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+1200);
	AT_FORK(gap_ncore(), (void *) KerIntegralImagePrime, (void *) KerArg0);
	__CALL(KerIntegralImagePrime, KerArg0);
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg1->KerBuffer = (unsigned int * __restrict__) (FaceDet_L1_Memory+1200);
		AT_FORK(gap_ncore(), (void *) KerSquaredIntegralImageProcess, (void *) KerArg1);
		__CALL(KerSquaredIntegralImageProcess, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) IntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+1360), 4800, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write IntegralImage */
	/*============================ End Write Tiles Epilog ===============================*/
}
void ProcessCascade_3(
		unsigned int *  __restrict__ IntegralImage,
		unsigned int *  __restrict__ SquaredIntegralImage,
		void *  cascade_model,
		int  *  __restrict__ CascadeReponse)

{
	/* Shared L1: 10076 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: KerII, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4800 [Tile0, 1:[40x30], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[40x30], 4]
		Tile0: [0, 4800, 4800], Tile1: [0, 4800, 4800], Tile2; [0, 4800, 4800]
	Ker Arg: KerIISQ, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4800 [Tile0, 1:[40x30], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[40x30], 4]
		Tile0: [0, 4800, 4800], Tile1: [0, 4800, 4800], Tile2; [0, 4800, 4800]
	Ker Arg: KerOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 476 [Tile0, 1:[17x7], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[17x7], 4]
		Tile0: [0, 476, 476], Tile1: [0, 476, 476], Tile2; [0, 476, 476]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) IntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0), 4800, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read IntegralImage */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) SquaredIntegralImage+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+4800), 4800, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read SquaredIntegralImage */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerEvaluateCascade(
			(unsigned int * __restrict__) (FaceDet_L1_Memory+0),
			(unsigned int * __restrict__) (FaceDet_L1_Memory+4800),
			(unsigned int) (40),
			(unsigned int) (30),
			(void *) (cascade_model),
			(unsigned char) (24),
			(unsigned char) (24),
			(int * __restrict__) (FaceDet_L1_Memory+9600)
		);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) CascadeReponse+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+9600), 476, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write CascadeReponse */
	/*============================ End Write Tiles Epilog ===============================*/
}
void final_resize(
		unsigned char * In,
		unsigned char * Out)

{
	/* Shared L1: 17456 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerResizeBilinear_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _N_In, _Off_In;
	unsigned int _SN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 12]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 12 logical tiles, 12 physical tiles
			Total Size: 19200 [Tile0, 12:[160x10, 10:160x10, 160x10], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 12:[160x10, 10:160x10, 160x10], 1]
		Tile0: [0, 1600, 1600], Tile1: [1600, 1600, 1600], Tile2; [3200, 1600, 1600]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 12 logical tiles, 12 physical tiles
			Total Size: 79056 [Tile0, 12:[324x22, 10:324x22, 324x22], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 12:[324x22, 10:324x22, 324x22], 1]
		Tile0: [0, 7128, 7128], Tile1: [6480, 7128, 7128], Tile2; [12960, 7128, 7128]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Win = (unsigned int) (324);
	KerArg0->Hin = (unsigned int) (244);
	KerArg0->Wout = (unsigned int) (160);
	KerArg0->Hout = (unsigned int) (120);
	KerArg0->HTileOut = (unsigned int) (10);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=1600;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0+0), 7128, 0, &DmaR_Evt1);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	for (T0Ind=0; T0Ind<12; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
		int T0Ind_Last = (T0Ind==11), T0Ind_NextLast = ((T0Ind+1)==11);
		/*================================= Prepare Tiles ===================================*/
		_SN_In = 0;
		if (!(T0Ind_Last)) {
			_N_In = _N_In + 0; _Off_In = (((132710*((T0Ind)+1)*10)>>16)*324); _SN_In = (7128); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
		if (_SN_In) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In+_Off_In), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+0+7128*((T0Ind_Total+1)%2)),
					_SN_In, 0, &DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->In = (unsigned char * __restrict__) (FaceDet_L1_Memory+0+7128*((T0Ind_Total)%2));
		KerArg0->Out = (unsigned char * __restrict__) (FaceDet_L1_Memory+14256+1600*((T0Ind_Total)%2));
		KerArg0->FirstLineIndex = (unsigned int) ((132710*(T0Ind)*10)>>16);
		AT_FORK(gap_ncore(), (void *) KerResizeBilinear, (void *) KerArg0);
		__CALL(KerResizeBilinear, KerArg0);
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) FaceDet_L1_Memory+14256+1600*((T0Ind_Total)%2)),
				_SC_Out, 1, &DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T0Ind_Last)) {
			_C_Out = _C_Out + (1600); _SC_Out = (1600); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
