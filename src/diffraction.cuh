#ifndef DIFFRACTION_CUH
#define DIFFRACTION_CUH
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <iostream>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

double cuda_func(double);
int cuda_func1(int*, int);
void cuda_func3(int *, int *, int *, int);

void cuda_funcVec(int *, int *, int *, int);
void cuda_funcMat(float *, float *, float *, int, int);
void cuda_funcCube(float *, float *, float *, int, int, int);
void cuda_structureFactor(float *, float *, float *, float *, int, int);
void cuda_structureFactorChunk(float *, float *, float *, float *, float *, float *, int, int, int);
void cuda_structureFactorChunkParallel(float *, float *, float *, float *, float *, float *, int, int, int);

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif

