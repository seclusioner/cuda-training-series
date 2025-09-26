#include <stdio.h>
#include <math.h>

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

const float EPS = 1e-6;
const int DSIZE = 4096;
const int block_size = 256;  // CUDA maximum is 1024
// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds){

  int idx = threadIdx.x + blockIdx.x * blockDim.x; // create typical 1D thread index from built-in variables
  if (idx < ds){
    C[idx] = A[idx] + B[idx];         // do the vector (element) add here
  }
}

bool check(const float *a, const float *b, const float *c, int len){
  for(int i=0;i<len;i++){
    if(fabs((a[i] + b[i]) - c[i]) > EPS){
      printf("Mismatch at index[%d]: answer=%f, operation=%f\n", i, a[i]+b[i], c[i]);
      return false;
    }
  }

  return true;
}

int main(){

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  h_A = new float[DSIZE];  // allocate space for vectors in host memory
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];
  for (int i = 0; i < DSIZE; i++){  // initialize vectors in host memory
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
    h_C[i] = 0;
  }
  cudaMalloc(&d_A, DSIZE*sizeof(float));  // allocate device space for vector A
  cudaMalloc(&d_B, DSIZE*sizeof(float)); // allocate device space for vector B
  cudaMalloc(&d_C, DSIZE*sizeof(float)); // allocate device space for vector C
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy vector A to device:
  cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  // copy vector B to device:
  cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);

  cudaCheckErrors("cudaMemcpy H2D failure");
  //cuda processing sequence step 1 is complete
  vadd<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector C from device to host:
  cudaMemcpy(h_C, d_C, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if(check(h_A, h_B, h_C, DSIZE))
    printf("OK\n");
  else
    printf("WTF\n");

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  return 0;
}
