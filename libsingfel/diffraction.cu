//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <iostream>
#include "diffraction.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math_constants.h>

//#define THREADS_PER_BLOCK 200

/*
double cuda_func(double ang) {
	// H has storage for 4 integers
    thrust::host_vector<int> H(4);
	
    // H and D are automatically deleted when the function returns
    return ang;
}

int cuda_func1(int *raw_ptr, int N) {
	// wrap raw pointer with a device_ptr 
	thrust::device_ptr<int> dev_ptr(raw_ptr);
	// use device_ptr in thrust algorithms
	thrust::fill(dev_ptr, dev_ptr + N, (int) 9);
	// access device memory through device_ptr
	dev_ptr[0] = 1;
    return dev_ptr[2];
}
*/

__global__ void addVec(int *a, int *b, int *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void addMat(float *a, float *b, float *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void addCube(float *a, float *b, float *c) { // colume-wise access
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void matAmp(float *F,float *Fim, int numPix){
	int index = ((blockIdx.y*blockDim.y + threadIdx.y)*gridDim.x + blockIdx.x )*blockDim.x + threadIdx.x;
	if (index<numPix){
		F[index]=F[index]*F[index]+Fim[index]*Fim[index];
	}
}

/*
__global__ void structureFactor(float *F, float *f, float *q, float *p, int numPix, int numAtoms){
	//int index = blockIdx.x;
	int index = ((blockIdx.y*blockDim.y + threadIdx.y)*gridDim.x + blockIdx.x )*blockDim.x + threadIdx.x;
	if (index < numPix) {
		float sf_real = 0;
		float sf_imag = 0;
		// p (Nx3)
		// q (py x px x 3)
		float map = 0;
		int f_ind = 0;
		for (int n = 0; n < numAtoms; n++) {
			map = 6.283185307F * (p[n]*q[index] + p[n+numAtoms]*q[index+numPix] + p[n+(2*numAtoms)]*q[index+(2*numPix)]);
			f_ind = index+(n*numPix);
			sf_real += f[f_ind] * cos(map);
			sf_imag += f[f_ind] * sin(map);
		}
		F[index] = sf_real * sf_real + sf_imag * sf_imag;
	}
}
*/
	
//structureFactor<<<dim3(0x6,1),dim3(0xB,1)>>>(d_F,d_f,d_q,d_p,d_i,numPix,chunkSize);
__global__ void structureFactor(float *Fre, float *Fim, float *f, float *q, \
                                float *p, int *i, int numPix, int chunkSize) {
	int index = ( (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x \
	            + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (index<numPix){
		// F (py x px) [re & im]
		// f (py x px x numAtomTypes)
		// q (py x px x 3)
		// i (1 x chunkSize)
		// p (chunkSize x 3)
		float map = 0;
		int f_ind = 0;
		for (int n = 0; n < chunkSize; n++) {
			map = 6.283185307F * (p[n] * q[index] + p[n+chunkSize] \
			      * q[index+(numPix)] + p[n+(2*chunkSize)]*q[index+(2*numPix)]);
			f_ind = index + i[n]*numPix;
			Fre[index] += f[f_ind] * cos(map);
			Fim[index] += f[f_ind] * sin(map);
		}
	}
}

/*
__global__ void structureFactorChunk(float *sf_real, float *sf_imag, float *f, float *q, int *i, float *p, int numAtomTypes, int numPix, int chunkSize){
	int index = ((blockIdx.y*blockDim.y + threadIdx.y)*gridDim.x + blockIdx.x )*blockDim.x + threadIdx.x;
	if (index<numPix){
		// F (py x px)
		// f (py x px x numAtomTypes)
		// q (py x px x 3)
		// i (1 x chunkSize)
		// p (chunkSize x 3)
		float map = 0;
		int f_ind = 0;
		for (int n = 0; n < chunkSize; n++) {
			map = 6.283185307F * (p[n]*q[index] + p[n+chunkSize]*q[index+(numPix)] + p[n+(2*chunkSize)]*q[index+(2*numPix)]);
			f_ind = index + i[n]*numPix;
			sf_real[index] += f[f_ind] * cos(map);
			sf_imag[index] += f[f_ind] * sin(map);
		}
	}
}

__global__ void structureFactorChunkParallel(float *pad_real, float *pad_imag, float *f, float *q, int *i, float *p, int numAtomTypes, int numPix, int chunkSize){
	int pixelId = blockIdx.x + blockIdx.y * gridDim.x;
	int chunkId = threadIdx.x;
	int index = pixelId + chunkId * numPix;
	if (pixelId < numPix && chunkId < chunkSize) {
		// F (py x px)
		// f (py x px x numAtomTypes)
		// q (py x px x 3)
		// i (1 x chunkSize)
		// p (chunkSize x 3)
		float map = 6.283185307F * (p[chunkId]*q[pixelId] + p[chunkId+chunkSize]*q[pixelId+(numPix)] + p[chunkId+(2*chunkSize)]*q[pixelId+(2*numPix)]);
		int f_ind = pixelId + i[chunkId]*numPix;
		pad_real[index] = f[f_ind] * cos(map);
		pad_imag[index] = f[f_ind] * sin(map);
	}
}
*/

void random_ints(int* a, int N)
{
   int i;
   for (i = 0; i < N; ++i)
    a[i] = rand() % 100; // between 0 and 100
}

void cuda_funcVec(int *a, int *b, int *c, int N) {
  int *d_a, *d_b, *d_c;
  int size = N*sizeof(int);
  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);
  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  // Launch add() kernel on GPU
  addVec<<<N,1>>>(d_a, d_b, d_c);
  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  // Cleanup
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void cuda_funcMat(float *a, float *b, float *c, int H, int W) {
  float *d_a, *d_b, *d_c;
  int size = H*W*sizeof(int);
  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);
  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  // Launch add() kernel on GPU
  addMat<<<H*W,1>>>(d_a, d_b, d_c);
  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  // Cleanup
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void cuda_funcCube(float *a, float *b, float *c, int H, int W, int Z) {
  float *d_a, *d_b, *d_c;
  int size = H*W*Z*sizeof(int);
  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);
  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  // Launch add() kernel on GPU
  addCube<<<H*W*Z,1>>>(d_a, d_b, d_c);
  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  // Cleanup
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

/*
void cuda_structureFactor(float *F, float *f, float *q, float *p, int py, int px, int numAtoms) {
	float *d_f, *d_q, *d_p;
	float *d_F;//, *d_F_real, *d_F_imag;
	int size_f = py*px*numAtoms*sizeof(float);
	int size_q = py*px*3*sizeof(float);
	int size_p = numAtoms*3*sizeof(float);
	int size_F = py*px*sizeof(float);
	// Malloc
	//float *F_real = (float *)malloc(size_F);
	//float *F_imag = (float *)malloc(size_F);
	// Allocate space for device copies of a, b, c
  	cudaMalloc((void **)&d_f, size_f);
  	cudaMalloc((void **)&d_q, size_q);
  	cudaMalloc((void **)&d_p, size_p);
	cudaMalloc((void **)&d_F, size_F);
	//cudaMalloc((void **)&d_F_imag, size_F);
	// Copy inputs to device
  	cudaMemcpy(d_f, f, size_f, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_q, q, size_q, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_p, p, size_p, cudaMemcpyHostToDevice);
  	// Launch add() kernel on GPU
  	structureFactor<<<py*px,1>>>(d_F, d_f, d_q, d_p, py, px, numAtoms);
  	// Copy result back to host
  	cudaMemcpy(F, d_F, size_F, cudaMemcpyDeviceToHost);
  	//cudaMemcpy(F_imag, d_F_imag, size_F, cudaMemcpyDeviceToHost);
	// Cleanup
  	cudaFree(d_f); cudaFree(d_q); cudaFree(d_p), cudaFree(d_F); //cudaFree(d_F_imag);
	//free(F_real); free(F_imag);
}
*/

/*
void cuda_structureFactor(float *F, float *f, float *q, float *p, int numPix, int numAtoms) {
	float *d_F, *d_f, *d_q, *d_p;
	int size_F = numPix*sizeof(float);
	int size_f = numPix*numAtoms*sizeof(float);
	int size_q = numPix*3*sizeof(float);
	int size_p = numAtoms*3*sizeof(float);
	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_F, size_F);
  	cudaMalloc((void **)&d_f, size_f);
  	cudaMalloc((void **)&d_q, size_q);
  	cudaMalloc((void **)&d_p, size_p);
	// Copy inputs to device
  	cudaMemcpy(d_f, f, size_f, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_q, q, size_q, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_p, p, size_p, cudaMemcpyHostToDevice);
  	// Launch add() kernel on GPU
	dim3 threads_per_block(20,10); // Maximum number of threads per block
	dim3 number_of_blocks(20,10,1);
  	structureFactor<<<number_of_blocks,threads_per_block>>>(d_F, d_f, d_q, d_p, numPix, numAtoms);
	//cudaThreadSynchronize();
  	//structureFactor<<<py*px,1>>>(d_F, d_f, d_q, d_p, py, px, numAtoms);
  	// Copy result back to host
  	cudaMemcpy(F, d_F, size_F, cudaMemcpyDeviceToHost);
	// Cleanup
  	cudaFree(d_F); cudaFree(d_f); cudaFree(d_q); cudaFree(d_p);
}
*/

void cuda_structureFactor(float *F, float *f, float *q, float *p, int *i, \
                          int numPix, int numAtoms, int numAtomTypes, \
                          int deviceID){
	int device,numBlocks;
	size_t size_F,size_f,size_q,size_i,size_p;
	size_t globalMem,fixedMem;
	int chunk,chunkSize,chunkSizeMax;
	float *d_F,*d_f,*d_p,*d_q,*d_Fim;
	int *d_i;
	struct cudaDeviceProp prop; 
	dim3 dimG,dimB;
	size_F = sizeof(float) * numPix;
	size_f = sizeof(float) * numPix * numAtomTypes;
	size_q = sizeof(float) * numPix * 3;
	fixedMem = size_F*2 + size_f + size_q;
	dimB.x = CUDA_BLOCK_SIZE;
	dimG.x = numBlocks = (numPix+dimB.x-1)/dimB.x;
	if (dimG.x > CUDA_GROUP_LIMIT){
		dimG.y = (numBlocks + CUDA_GROUP_LIMIT -1) / CUDA_GROUP_LIMIT;
		dimG.x = (numBlocks + dimG.y -1) / dimG.y;
	}
	//cudaGetDevice(&device);
	device = deviceID;
	cudaSetDevice(device);
	cudaGetDeviceProperties(&prop,device);
	
	globalMem = prop.totalGlobalMem  - CUDA_RESERVE_MEM;
	if (globalMem <= fixedMem){
		printf("Device memory[%lu] not enough to hold all data[>%lu]!\n", \
		       globalMem,fixedMem);
		exit(EXIT_FAILURE);
	}	
	chunkSizeMax = (globalMem - fixedMem) / (3 * sizeof(float) + sizeof(int));
	if (chunkSizeMax>numAtoms)
		chunkSizeMax=numAtoms;

	size_p = sizeof(float) * chunkSizeMax * 3;
	size_i = sizeof(int)   * chunkSizeMax;

	cudaMalloc((void **)&d_F, size_F);
	cudaMalloc((void **)&d_Fim, size_F);
  	cudaMalloc((void **)&d_f, size_f);
  	cudaMalloc((void **)&d_q, size_q);
  	cudaMalloc((void **)&d_p, size_p);
  	cudaMalloc((void **)&d_i, size_i);
  	cudaMemcpy(d_f, f, size_f, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_q, q, size_q, cudaMemcpyHostToDevice);
  	cudaMemset(d_F,0,size_F);
  	cudaMemset(d_Fim,0,size_F);
 	
  	chunkSize=chunkSizeMax;
  	for(chunk=0;chunk<numAtoms;chunk+=chunkSizeMax){
		
		if (chunkSize+chunk>=numAtoms)
			chunkSize=numAtoms-chunk;
		cudaMemcpy(d_i, i + chunk, sizeof(int)*	chunkSize, \
		           cudaMemcpyHostToDevice);
		cudaMemcpy(d_p, p + chunk, sizeof(float)*chunkSize, \
		           cudaMemcpyHostToDevice);
		cudaMemcpy(d_p + chunkSize, p + chunk + numAtoms, \
		           sizeof(float)*chunkSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_p + 2*chunkSize, p + chunk + 2*numAtoms, \
		           sizeof(float)*chunkSize, cudaMemcpyHostToDevice);
		structureFactor<<<dimG,dimB>>>(d_F, d_Fim, d_f, d_q, d_p, d_i, numPix, \
		                               chunkSize);
	}
  	matAmp<<<dimG,dimB>>>(d_F,d_Fim,numPix);
  	cudaMemcpy(F, d_F, size_F, cudaMemcpyDeviceToHost);
  	cudaFree(d_F);
  	cudaFree(d_Fim);
  	cudaFree(d_f);
  	cudaFree(d_q);
  	cudaFree(d_p);
  	cudaFree(d_i);
}

int cuda_getDeviceCount(){
	int tmp;
	if (cudaSuccess != cudaGetDeviceCount(&tmp))
		return 0;
	return tmp;
}

/*	
void cuda_structureFactorChunk(float *sf_real, float *sf_imag, float *f, float *q, int *i, float *p, int numAtomTypes, int numPix, int chunkSize) {
	float *d_sf_real, *d_sf_imag, *d_f, *d_q, *d_p; // Pointer to device memory
	int *d_i;
	int size_sf = numPix*sizeof(float);
	int size_f = numPix*numAtomTypes*sizeof(float);
	int size_q = numPix*3*sizeof(float);
	int size_i = chunkSize*sizeof(int);
	int size_p = chunkSize*3*sizeof(float);
	// Allocate space for device copies
	cudaMalloc((void **)&d_sf_real, size_sf);
	cudaMalloc((void **)&d_sf_imag, size_sf);
  	cudaMalloc((void **)&d_f, size_f);
  	cudaMalloc((void **)&d_q, size_q);
	cudaMalloc((void **)&d_i, size_i);
  	cudaMalloc((void **)&d_p, size_p);
	// Copy inputs to device
  	cudaMemcpy(d_sf_real, sf_real, size_sf, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_sf_imag, sf_imag, size_sf, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_f, f, size_f, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_q, q, size_q, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_i, i, size_i, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_p, p, size_p, cudaMemcpyHostToDevice);
  	// Launch kernel on GPU
	dim3 threads_per_block(512); // Maximum number of threads per block
	dim3 number_of_blocks(2048,2048,1);
  	//structureFactorChunk<<<number_of_blocks,threads_per_block>>>(d_sf_real, d_sf_imag, d_f, d_q, d_i, d_p, numAtomTypes, numPix, chunkSize);
	structureFactorChunkParallel<<<number_of_blocks,threads_per_block>>>(d_sf_real, d_sf_imag, d_f, d_q, d_i, d_p, numAtomTypes, numPix, chunkSize);
	//cudaThreadSynchronize();
  	// Copy result back to host
  	cudaMemcpy(sf_real, d_sf_real, size_sf, cudaMemcpyDeviceToHost);
  	cudaMemcpy(sf_imag, d_sf_imag, size_sf, cudaMemcpyDeviceToHost);
	// Cleanup
  	cudaFree(d_sf_real); cudaFree(d_sf_imag); cudaFree(d_f); cudaFree(d_q); cudaFree(d_i); cudaFree(d_p);
}

void cuda_structureFactorChunkParallel(float *pad_real, float *pad_imag, float *f, float *q, int *i, float *p, int numAtomTypes, int numPix, int chunkSize) {
	float *d_pad_real, *d_pad_imag, *d_f, *d_q, *d_p; // Pointer to device memory
	int *d_i;
	int size_pad = numPix*chunkSize*sizeof(float);
	int size_f = numPix*numAtomTypes*sizeof(float);
	int size_q = numPix*3*sizeof(float);
	int size_i = chunkSize*sizeof(int);
	int size_p = chunkSize*3*sizeof(float);
	// Allocate space for device copies
  	cudaMalloc((void **)&d_pad_real, size_pad);
  	cudaMalloc((void **)&d_pad_imag, size_pad);
  	cudaMalloc((void **)&d_f, size_f);
  	cudaMalloc((void **)&d_q, size_q);
	cudaMalloc((void **)&d_i, size_i);
  	cudaMalloc((void **)&d_p, size_p);
	// Copy inputs to device
  	cudaMemcpy(d_f, f, size_f, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_q, q, size_q, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_i, i, size_i, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_p, p, size_p, cudaMemcpyHostToDevice);
  	// Launch kernel on GPU
	dim3 threads_per_block(chunkSize); // Maximum number of threads per block
	dim3 number_of_blocks(numPix);
	structureFactorChunkParallel<<<number_of_blocks,threads_per_block>>>(d_pad_real, d_pad_imag, d_f, d_q, d_i, d_p, numAtomTypes, numPix, chunkSize);
  	// Copy result back to host
  	cudaMemcpy(pad_real, d_pad_real, size_pad, cudaMemcpyDeviceToHost);
  	cudaMemcpy(pad_imag, d_pad_imag, size_pad, cudaMemcpyDeviceToHost);
	// Cleanup
  	cudaFree(d_pad_real); cudaFree(d_pad_imag); cudaFree(d_f); cudaFree(d_q); cudaFree(d_i); cudaFree(d_p);
}
*/

