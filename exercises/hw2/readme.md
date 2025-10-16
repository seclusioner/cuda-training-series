# Homework 2

These exercises will help reinforce the concept of Shared Memory on the GPU.

## **1. 1D Stencil Using Shared Memory**

Your first task is to create a 1D stencil application that uses shared memory. The code skeleton is provided in *stencil_1d.cu*. Edit that file, paying attention to the FIXME locations. The code will verify output and report any errors.

Execute:
``` bash
nvcc -o stencil_1d stencil_1d.cu
.\stencil_1d.exe
```

Sample output:
``` bash
Success!
```

If you have trouble, you can look at *stencil_1d_solution* for a complete example.

## **2. 2D Matrix Multiply Using Shared Memory**

Next, let`s apply shared memory to the 2D matrix multiply we wrote in Homework 1. FIXME locations are provided in the code skeleton in *matrix_mul_shared.cu*. See if you can successfully load the required data into shared memory and then appropriately update the dot product calculation. 

Note that timing information is included. Go back and run your solution from Homework 1 and observe the runtime. What runtime impact do you notice after applying shared memory to this 2D matrix multiply? How does it differ from the runtime you observed in your previous implementation?

If you have trouble, you can look at *matrix_mul_shared_solution* for a complete example.

Execute:
``` bash
nvcc -o matrix_mul_shared matrix_mul_shared.cu
.\matrix_mul_shared.exe <m> <n> <p>
```
Example:
``` bash
.\matrix_mul_shared.exe 512 1024 2048
```

Sample output:
``` bash
512 by 1024 multiply 1024 by 2048
CPU took 2.828000 seconds.
GPU took 0.101000 seconds
Success!
```

### Important notes on `__syncthreads()` and shared memory usage
* `__syncthreads()` is a barrier synchronization that **requires all threads in a block to reach the call before any can proceed**. If some threads skip `__syncthreads()` due to conditional statements, it **causes undefined behavior or deadlocks.**
  
* When implementing tiled matrix multiplication with shared memory:
1. Avoid divergent control flow (like if/else/break/continue) around __syncthreads().
2. Prefer unified control flow per warp/block so all threads execute __syncthreads() the same number of times.

* To handle boundary conditions (when thread indices exceed matrix dimensions), perform boundary checks only when reading from global memory into shared memory and store zeros in shared memory for threads outside bounds.

Use the following main() function to randomly stress test your kernel over multiple iterations.
``` cpp
int main(){
  srand(time(0));

  int m, n, p;
  int count = 0;
  while(count < 100){
    m = (int)(rand() % DSIZE +1);
    n = (int)(rand() % DSIZE +1);
    p = (int)(rand() % DSIZE +1);
    printf("%d time: ", count+1);

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
    
    mmul_host(h_A, h_B, h_C, m, n, p);

    // Allocate device memory and copy input data over to GPU
    cudaMalloc(&d_A, m*n*sizeof(float));
    cudaMalloc(&d_B, n*p*sizeof(float));
    cudaMalloc(&d_C, m*p*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n*p*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    dim3 block(block_size, block_size);
    dim3 grid((p+block.x-1)/block.x, (m+block.y-1)/block.y);
    cudaMemset(d_C, 0, m * p * sizeof(float));
    mmul2<<<grid, block>>>(d_A, d_B, d_C, m, n, p);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel launch failure");

    cudaMemcpy(h_C2, d_C, m*p*sizeof(float), cudaMemcpyDeviceToHost);
    

    // Verify results
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    for (int i = 0; i < m*p; i++) {
      if (fabs(h_C[i] - h_C2[i]) > EPS) {
          printf("Mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], h_C2[i]);
          return -1;
      }
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C2;
    count++;
  }
  printf("Success!\n");
  return 0;
}
```