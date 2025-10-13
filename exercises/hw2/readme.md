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

Next, let's apply shared memory to the 2D matrix multiply we wrote in Homework 1. FIXME locations are provided in the code skeleton in *matrix_mul_shared.cu*. See if you can successfully load the required data into shared memory and then appropriately update the dot product calculation. Compile and run your code using the following:

```
module load cuda
nvcc -o matrix_mul matrix_mul_shared.cu
lsfrun ./matrix_mul
```

Note that timing information is included. Go back and run your solution from Homework 1 and observe the runtime. What runtime impact do you notice after applying shared memory to this 2D matrix multiply? How does it differ from the runtime you observed in your previous implementation?

If you have trouble, you can look at *matrix_mul_shared_solution* for a complete example.
