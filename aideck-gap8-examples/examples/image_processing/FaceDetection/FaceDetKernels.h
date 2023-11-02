#ifndef __FACEDETKERNEL_H__
#define __FACEDETKERNEL_H__

#include "AutoTilerLibTypes.h"
#include "FaceDetBasicKernels.h"
#define _FaceDet_L1_Memory_SIZE 24904
#define _FaceDet_L2_Memory_SIZE 0
extern char *FaceDet_L1_Memory; /* Size given for generation: 25000 bytes, used: 24904 bytes */
extern char *FaceDet_L2_Memory; /* Size used for generation: 0 bytes */
extern void ResizeImage_1(
		unsigned char * In,
		unsigned char * Out);
extern void ProcessIntegralImage_1(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage);
extern void ProcessSquaredIntegralImage_1(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage);
extern void ProcessCascade_1(
		unsigned int *  __restrict__ IntegralImage,
		unsigned int *  __restrict__ SquaredIntegralImage,
		void *  cascade_model,
		int  *  __restrict__ CascadeReponse);
extern void ResizeImage_2(
		unsigned char * In,
		unsigned char * Out);
extern void ProcessIntegralImage_2(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage);
extern void ProcessSquaredIntegralImage_2(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage);
extern void ProcessCascade_2(
		unsigned int *  __restrict__ IntegralImage,
		unsigned int *  __restrict__ SquaredIntegralImage,
		void *  cascade_model,
		int  *  __restrict__ CascadeReponse);
extern void ResizeImage_3(
		unsigned char * In,
		unsigned char * Out);
extern void ProcessIntegralImage_3(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage);
extern void ProcessSquaredIntegralImage_3(
		unsigned char *  __restrict__ ImageIn,
		unsigned int *  __restrict__ IntegralImage);
extern void ProcessCascade_3(
		unsigned int *  __restrict__ IntegralImage,
		unsigned int *  __restrict__ SquaredIntegralImage,
		void *  cascade_model,
		int  *  __restrict__ CascadeReponse);
extern void final_resize(
		unsigned char * In,
		unsigned char * Out);
#endif
