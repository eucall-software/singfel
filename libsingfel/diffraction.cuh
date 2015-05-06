#ifndef DIFFRACTION_CUH
#define DIFFRACTION_CUH
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <iostream>
//expect CUDA to reserve extra mem for context
#define CUDA_RESERVE_MEM (1<<27)
//CUDA workgroup (block) size (opt 512 - 1024)
#define CUDA_BLOCK_SIZE 512
//CUDA group X dim limit (as of CA <3)
#define CUDA_GROUP_LIMIT 65535
#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

int cuda_getDeviceCount();
	
double cuda_func(double);
int cuda_func1(int*, int);
void cuda_func3(int *, int *, int *, int);

void cuda_funcVec(int *, int *, int *, int);
void cuda_funcMat(float *, float *, float *, int, int);
void cuda_funcCube(float *, float *, float *, int, int, int);
//void cuda_structureFactor(float *, float *, float *, float *, int, int);
//void cuda_structureFactorChunk(float *, float *, float *, float *, int *, float *, int, int, int);
//void cuda_structureFactorChunkParallel(float *, float *, float *, float *, int *, float *, int, int, int);
void cuda_structureFactor(float *, float *, float *, float *, int *, int, int, int, int = 0);

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif

