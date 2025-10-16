#include <stdio.h>

// these are just for timing measurments
#include <time.h>
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
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block
const float EPS = 1e-5;

// matrix multiply (naive) kernel: C = A * B
// (M, N) = (M, K) @ (K, N)
__global__ void mmul(const float *A, const float *B, float *C, int M, int K, int N) {
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float temp = 0.0f;

  for (int i = 0; i < (K + block_size - 1) / block_size; i++) {
    int tiledCol = i * block_size + threadIdx.x;
    int tiledRow = i * block_size + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0f;

    __syncthreads();

    for (int k = 0; k < block_size; ++k) {
      temp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = temp;
  }
}

void mmul_host(const float *A, const float *B, float *C, int m, int n, int p){
  for(int i=0;i<m;i++){
    for(int j=0;j<p;j++){
      for(int k=0;k<n;k++){
        C[i*p+j] += A[i*n+k] * B[k*p+j];
      }
    }
  }
}

int main(int argc, char* argv[]){
  if (argc != 4) {
      printf("Usage: %s <m> <n> <p>\n", argv[0]);
      return 1;
  }
  srand(time(0));

  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int p = atoi(argv[3]);

  if (m <= 0 || n <= 0 || p <= 0 || m > DSIZE || n > DSIZE || p > DSIZE) {
      printf("Matrix sizes must be in range 1 to %d\n", DSIZE);
      return 1;
  }
  printf("%d by %d multiply %d by %d\n", m, n, n, p);

  float *h_A, *h_B, *h_C, *h_C2, *d_A, *d_B, *d_C;

  // these are just for timing
  clock_t t0, t1, t2;
  double t1sum=0.0;
  double t2sum=0.0;

  // initialize
  h_A = new float[m*n]; // m x n
  h_B = new float[n*p]; // n x p
  h_C = new float[m*p]; // m x p
  h_C2 = new float[m*p];
  for (int i = 0; i < (m*n); i++){
    h_A[i] = (float)(rand() % 10);
  }
  for (int i = 0; i < (n*p); i++){
    h_B[i] = (float)(rand() % 10);
  }
  memset(h_C, 0, sizeof(float)*m*p);
  memset(h_C2, 0, sizeof(float)*m*p);

  // start timing
  t0 = clock();
  mmul_host(h_A, h_B, h_C, m, n, p);
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("CPU took %f seconds.\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, m*n*sizeof(float));
  cudaMalloc(&d_B, n*p*sizeof(float));
  cudaMalloc(&d_C, m*p*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, m*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, n*p*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid((p+block.x-1)/block.x, (m+block.y-1)/block.y);
  cudaMemset(d_C, 0, m * p * sizeof(float));
  mmul<<<grid, block>>>(d_A, d_B, d_C, m, n, p);
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C2, d_C, m*p*sizeof(float), cudaMemcpyDeviceToHost);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf ("GPU took %f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < m*p; i++) {
    if (fabs(h_C[i] - h_C2[i]) > EPS) {
        printf("Mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], h_C2[i]);
        return -1;
    }
  }
  printf("Success!\n");
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_C2;

  return 0;
}
  
