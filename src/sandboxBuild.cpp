#include <iostream>
#include "diffraction.cuh"
#include <cuda.h>
#ifdef COMPILE_WITH_CXX11
	#define ARMA_DONT_USE_CXX11
#endif
#include <armadillo>

using namespace std;
//using namespace arma;

int main(){

  arma::wall_clock timer;

  double ax = 2;
  double damn = cuda_func(ax);
  cout << damn << endl;
/*
  int *a;
  int *b;
  int *c;
  int N = 5;
  int size = N*sizeof(int);
  c = (int *)malloc(size); 
  // Alloc space for host copies of a, b, c and setup input values 
  cuda_func3(a, b, c, N);
  for (int i = 0; i < N; i++) {
  	cout << "c[" <<i<< "]: " << c[i] << endl;
  }
  free(c);
*/  

int xt = 2147483647; // -2147483648 to 2147483647 (2^32/2-1) is the maximum int
int yt = 0;
cout << "max int: " << xt+yt << endl;

float xf = 1.23456789e-28; // -2147483648 to 2147483647 (2^32/2-1) is the maximum int
float yf = 0;
cout << "max float: " << xf+yf << endl;
  
/*
  int* aux_mem = new int[N];
  arma::ivec yum(aux_mem, N, true, true); 
  yum.fill(7);
  int* aux_mem1 = new int[N];
  arma::ivec yum1(aux_mem1, N, true, true);
  //arma::Col<arma::sword> yum1(N);
  yum1.fill(5);
  yum.print("yum: ");
  yum1.print("yum1: ");
  int *d = (int *)malloc(size); 
  cuda_func4(aux_mem, aux_mem1, d, N);
  for (int i = 0; i < N; i++) {
  	cout << "d[" <<i<< "]: " << d[i] << endl;
  }
  free(d);
*/

// Vector GPU
  int N = 5;
  arma::Col<arma::sword> A(N);
  int* A_mem = A.memptr();
  A.fill(7);
  A.print("A: ");
  arma::Col<arma::sword> B(N);
  int* B_mem = B.memptr();
  B = arma::linspace<arma::icolvec>(10,15,N);
  //B.fill(5);
  B.print("B: ");
  //int *d = (int *)malloc(size);
  arma::Col<arma::sword> T(N);
  int* T_mem = T.memptr(); 
  cuda_funcVec(A_mem, B_mem, T_mem, N);
  T.print("T: ");

// Matrix GPU
  int W = 2;
  int H = 3;
  arma::Mat<float> C(H,W);
  float* C_mem = C.memptr();
  C.randu();
  C.print("C: ");
  arma::Mat<float> D(H,W);
  float* D_mem = D.memptr();
  D.randu();
  D.print("D: ");
  arma::Mat<float> G(H,W);
  float* G_mem = G.memptr();
  cuda_funcMat(C_mem, D_mem, G_mem, H, W);
  G.print("G: ");

// Cube GPU
  int Z = 2;
  arma::Cube<float> P(H,W,Z);
  float* P_mem = P.memptr();
  P.randu();
  P.print("P: ");
  arma::Cube<float> Q(H,W,Z);
  float* Q_mem = Q.memptr();
  Q.randu();
  Q.print("Q: ");
  arma::Cube<float> S(H,W,Z);
  float* S_mem = S.memptr();
  cuda_funcCube(P_mem, Q_mem, S_mem, H, W, Z);
  S.print("S: ");

/*
thrust::host_vector<int> hv = populate();        // make data on host

thrust::device_vector<int> dv(hv.begin(), hv.end()); // copy to device

thrust::sort(dv.begin(), dv.end());              // sort on device

thrust::copy(dv.begin(), dv.end(), hv.begin());  // copy back
*/
/*
arma::u32 Z_rows = 10;
//arma::u32 Z_cols = 20;

int* aux_mem = new int[Z_rows];
//mat Z(aux_mem,Z_rows,Z_cols,false,true);
//Z = randn(Z_rows, Z_cols);

arma::ivec Z(aux_mem, Z_rows, true, true);
Z.fill(5);
Z.print("Z:");
thrust::device_vector<int> vec(aux_mem, aux_mem + Z_rows);
thrust::fill(aux_mem, aux_mem + Z_rows, (int) 6);
// access device memory through device_ptr
aux_mem[0] = 1;
Z.print("Z cuda:");
//int y = cuda_func1(aux_mem,10);
  // free memory
//  cudaFree(aux_mem);
//  cout << y << endl;

mat A = randu<mat>(5,5);
double* A_mem = A.memptr();
*/

  int M = 5000000;
  timer.tic();
  arma::Col<arma::sword> myVec(M);
  myVec.fill(9);
  cout << "arma: " << timer.toc() << endl;

  timer.tic();
  
  // raw pointer to device memory
  size_t available, total;
  cudaMemGetInfo(&available, &total);
  cout << "av/total: " << available <<"/"<<total << endl;
  int * raw_ptr;
  cudaError_t err = cudaMalloc((void **) &raw_ptr, M * sizeof(int));
  if( err != cudaSuccess)
  {
     printf("CUDA error: %s\n", cudaGetErrorString(err));
     return 1;
  }
  int x = cuda_func1(raw_ptr,M); // thrust takes care of kernel launch
  cudaMemGetInfo(&available, &total);
  cout << "av/total: " << available <<"/"<<total << endl;
  // free memory
  cudaFree(raw_ptr);
  cout << "cuda_func1: " << timer.toc() << endl;
  cout << x << endl;

  return 0;
}
