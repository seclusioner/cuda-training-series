/*
Matrix multiplication for square matrix
*/
#include <stdio.h>

// these are just for timing measurments
#include <cstdlib>  // rand()
#include <ctime>    // time()

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int DSIZE = 4096;
const int block_size = 16;  // CUDA maximum is 1024 *total* threads in block
const float EPS = 1e-5;

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

  int idx = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  int idy = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds)){
    float temp = 0;
    for (int i = 0; i < ds; i++)
      temp += A[idy*ds+i] * B[i*ds+idx];   // dot product of row and column
    C[idy*ds+idx] = temp;
  }
}

void mmul_host(const float *A, const float *B, float *C, int ds){
  int m, n, p; // m x p = m x n @ n x p
  m = n = p = ds; // square matrix
  for(auto i=0;i<m;i++){
    for(auto j=0;j<p;j++){
      for(auto k=0;k<n;k++){
        C[i*p+j] += A[i*n+k] * B[k*p+j];
      }
    }
  }
}

void printMatrix(float* matrix, int rows, int cols){
    for (int i = 0; i < rows; i++){
        printf("[");
        for (int j = 0; j < cols; j++){
          printf("%.3f, ", matrix[i*cols+j]);
        }
        printf("],\n");
    }
}


int main(){
  srand(time(0));
  
  float *h_A, *h_B, *h_C, *h_C2, *d_A, *d_B, *d_C;

  // these are just for timing
  clock_t t0, t1, t2;
  double t1sum=0.0;
  double t2sum=0.0;

  h_A = new float[DSIZE*DSIZE];
  h_B = new float[DSIZE*DSIZE];
  h_C = new float[DSIZE*DSIZE];
  h_C2 = new float[DSIZE*DSIZE];
  for (int i = 0; i < DSIZE*DSIZE; i++){
    h_A[i] = rand() % 10;
    h_B[i] = rand() % 10;
    h_C[i] = 0;
    h_C2[i] = 0;
  }

  // printf("Matrix A:\n");
  // printMatrix(h_A, DSIZE, DSIZE);
  // printf("Matrix B:\n");
  // printMatrix(h_B, DSIZE, DSIZE);
  // printf("Matrix C (from CPU):\n");
  t0 = clock();

  mmul_host(h_A, h_B, h_C, DSIZE);
  // printMatrix(h_C, DSIZE, DSIZE);
  
  // CPU timing
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf ("CPU computation time: %.2f seconds\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C2, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

  // printf("Matrix C (from GPU):\n");
  // printMatrix(h_C2, DSIZE, DSIZE);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf ("GPU computation time: %.2f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete
  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < DSIZE*DSIZE; i++) {
    if (fabs(h_C[i] - h_C2[i]) > EPS) {
        printf("Mismatch at index %d, was: %f, should be: %f\n", i, h_C2[i], h_C[i]);
        return -1;
    }
}
  printf("Success!\n");

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C2);

  return 0;
}
  
