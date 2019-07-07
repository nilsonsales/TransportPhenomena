

# <p style="text-align: center;"> The Heat Equation: High-Performance Scientific Computing Case Study </p>

<p style="text-align: center;"> Nilson Sales de Carvalho </p>  


---

<br>
  
  
## Datailed System Description:
- **Software**:  
Operating System: Antergos Linux  
Kernel Version: 5.1.16-arch1-1-ARCH  
OS Type: 64-bit
GCC version: 9.1.0
CUDA version: 10.1.168

- **Hardware**:  
Processors: 4 × Intel® Core™ i7-7500U CPU @ 2.70GHz  
Memory: 7,7 GiB of RAM  
**GPU**:  GM108M [GeForce 930MX]; 64 bits; 2004 MiB  



<br>

# Introduction
This works aims to use the API **OpenMP** and the ***NVIDIA*** **CUDA Toolkit**, alongside with the language C, to solve high-performance computating (HPC) problems, such as the *heat equation*. HPC problems require huge computational power, which makes parallel computation as an important tool to reduce the computational time to solve such problems. With the use of parallel computation, we can make use of threads to solve a high number of opperations at the same time.

<br>

# Activity 1: Dot Product

1. Code the parallel dot product using the code listings above.

The mathematical operation **dot products** is made by multiplying each element of an array by the element in the same position of another array. The code bellow implements the dot product in the C programming language:

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
    int sum = 0;
    int N = 100;
    int a[N], b[N];
    
    //initialization
    for( int i = 0; i < N; i++ ){
        a[i] = b[i] = 1;
    }

    #pragma omp parallel
    {
        //dot product
        #pragma omp for reduction(+:sum)
        for( int i = 0; i < N; i++ ){
            sum += a[i] * b[i];
        }
    }

    printf("Dot Product: %d\n", sum);
    return 0;
}
```
<br>

2. Instead of statically allocating the arrays a and b, use the malloc function (or new keyword) to dynamically allocate memory. Be sure to initialize the arrays properly and deallocate the array memory when finished.

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
    int sum = 0;
    int N = 100;

    int* a = malloc(N * sizeof(int));
    int* b = malloc(N * sizeof(int));
    
    //initialization
    for( int i = 0; i < N; i++ ){
        a[i] = b[i] = 1;
    }

    double start = omp_get_wtime();
    #pragma omp parallel
    {
        //dot product
        #pragma omp for reduction(+:sum)
        for( int i = 0; i < N; i++ ){
            sum += a[i] * b[i];
        }
    }
    double end = omp_get_wtime();
    double elapsed = end - start;

    printf("Dot Product: %d\n", sum);
    printf("Time: %lf\n", elapsed);

    free(a);
    free(b);

    return 0;
}
```

- CUDA:  
Let's now redo the same operation using the CUDA toolkit. With CUDA we can get advantage of optimazed libraries to parallelise our code on our nvidea card (GPU). For this, we use the **NVCC** compiler, saving our code with the extension **.cu**. In the CUDA architecture the GPU and its memory is called *device* and the CPU and the RAM memory *host*. We need to copy our variables from the RAM memory to our GPU memory, which has some computational cost. Differently from CPUs, that are for general purpose, GPUs are made to process graphics and image processing efficiently, which makes them very useful for our purposes of paralellising instructions. You can see the implementation of our code below:

```c
#include <stdio.h>
#include <stdlib.h>


__global__ void dot_prod(int n, int *a, int *b, int *c)
{
    int index = threadIdx.x;  
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride){  // T threads per iteration
        c[i] = a[i] * b[i];
    }
}

int main(int argc, char **argv){
    int sum = 0;
    int N = 10000;
    int size = N * sizeof(int);


    int *a, *b, *c;  // Host copies of a, b, c
    int *dev_a, *dev_b, *dev_c;  // Device copies of a, b, c


    // Allocate space for device copies a, b, c
    cudaMalloc((void **) &dev_a, size);
    cudaMalloc((void **) &dev_b, size);
    cudaMalloc((void **) &dev_c, size);

    a = (int *) malloc(size);
    b = (int *) malloc(size);
    c = (int *) malloc(size);
    
    // initialization of the arrays
    for(int i = 0; i < N; i++){
        a[i] = b[i] = 1;
    }


    // Copy inputs to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    dot_prod<<<1,256>>>(N, dev_a, dev_b, dev_c); // 1 block, T=256 threads (block size)

    // Copy result back to host
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    for( int i = 0; i < N; i++ ){
        sum += c[i];
    }


    printf("Dot Product: %d\n", sum);
    printf("Elapsed time: %f ms\n", elapsedTime);

    // Clean up
    free(a); free(b);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    return 0;
}
```

<br>


3. Use omp_get_wtime() to compute the elapsed wall clock time for the main dot product loop.

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
    int sum = 0;
    int N = 100;

    int* a = malloc(N * sizeof(int));
    int* b = malloc(N * sizeof(int));
    
    //initialization
    for( int i = 0; i < N; i++ ){
        a[i] = b[i] = 1;
    }

    double start = omp_get_wtime();
    #pragma omp parallel
    {
        //dot product
        #pragma omp for reduction(+:sum)
        for( int i = 0; i < N; i++ ){
            sum += a[i] * b[i];
        }
    }
    double end = omp_get_wtime();
    double elapsed = end - start;

    printf("Dot Product: %d\n", sum);
    printf("Time: %lf\n", elapsed);

    free(a);
    free(b);

    return 0;
}
```

Output:
```bash
Dot Product: 100
Time: 0.005285
```

For CUDA, we use *CUDA events* the measure the time:

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


__global__ void dot_prod(int n, int *a, int *b, int *c)
{
    int index = threadIdx.x;  
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride){  // T threads per iteration
        c[i] = a[i] * b[i];
    }
}

int main(int argc, char **argv){
    int sum = 0;
    int N = 10000;
    int size = N * sizeof(int);
    
    // Create counter
    cudaEvent_t start, stop;
    float elapsedTime;


    int *a, *b, *c;  // Host copies of a, b, c
    int *dev_a, *dev_b, *dev_c;  // Device copies of a, b, c


    // Allocate space for device copies a, b, c
    cudaMalloc((void **) &dev_a, size);
    cudaMalloc((void **) &dev_b, size);
    cudaMalloc((void **) &dev_c, size);

    a = (int *) malloc(size);
    b = (int *) malloc(size);
    c = (int *) malloc(size);
    
    // initialization of the arrays
    for(int i = 0; i < N; i++){
        a[i] = b[i] = 1;
    }


    // Start counter
    cudaEventCreate(&start);
    cudaEventRecord(start,0);


    // Copy inputs to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    dot_prod<<<1,256>>>(N, dev_a, dev_b, dev_c); // 1 block, T=256 threads (block size)

    // Copy result back to host
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    for( int i = 0; i < N; i++ ){
        sum += c[i];
    }

    // Stop counter
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);


    printf("Dot Product: %d\n", sum);
    printf("Elapsed time: %f ms\n", elapsedTime);

    // Clean up
    free(a); free(b);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    return 0;
}
```
Output:
```bash
Dot Product: 100
Elapsed time: 0.039424 ms
```

<br>

4. Examine how this execution time changes as you increase the number of threads. Use values up to and beyond the number of processor cores on your system.

- Using OpenMP (N = 100):

| N of threads  | Time elapsed (s) |
|:-------------:|:-------------:|
|       1       |   0.000005    |
|       2       |   0.000041    |
|       4       |   0.002633    |
|       8       |   0.016769    |
|      16       |   0.000569    |
|      32       |   0.003972    |
|      64       |   0.004103    |
|     128       |   0.001972    |
|     256       |   0.007706    |
|     512       |   0.016946    |
|    1024       |   0.024588    |

With the use of *OpenMP*, we don't see a reduction in time with the increase of number of thread for this specific problem.


<br>

- Using CUDA (N = 10000):  

| N of threads  | Time elapsed (ms) |  
|:-------------:|:-------------:|  
| 1           |    2.558336   |
| 2           |    1.345920   |
| 4           |    0.753952   |
| 8           |    0.450976   |
| 16          |    0.287904   |
| 32          |    0.230144   |
| 64          |    0.186208   |
| 128         |    0.171072   |
| 256         |    0.163168   |
| 512         |    0.152608   |
| 1024        |    0.149344   |

Running our CUDA code in the GPU, we can see a reduction in time as the number of threads increase.

<br>


# Activity 2: Matrix-Vector Product

2. This can be interpreted as a series of dot products. Modify the dot product code from Activity 1 to compute a matrix-vector product.

Multiplying a matrix MxN for a one-dimensional array (1xN), we have an a vertical array (Mx1). We can do it with the following code:

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int main(int argc, char **argv){
    int N = 100;
    int M = 20;
    int a[M][N], b[N], c[M];


    //initialization
    for( int i = 0; i < M; i++ ){ // rows
        for(int j = 0; j < N; j++){ // columns
            a[i][j] = 1;
        }
    }

    for(int j = 0; j < N; j++){ // rows
        b[j] = 1;
    }

    for(int i = 0; i < M; i++){ // rows
        c[i] = 0;
    }

    /////////////////

    double start = omp_get_wtime();

    // Matrix multiplication
    for(int i = 0; i < M; i++){
        for( int j = 0; j < N; j++ ){
            c[i] += a[i][j] * b[j];
        }
    }

    double end = omp_get_wtime();
    double elapsed = end - start;

    // Resut
    for(int i = 0; i < M; i++){
        printf("[ %d ]\n", c[i]);
    }

    printf("Time: %lf\n", elapsed);
    return 0;
}
```

<br>

3. Parallelize this matrix-vector product and perform an analysis on the execution time versus number of threads for several different matrix dimensions.

```c
int main(int argc, char **argv){
    int N = 1500;
    int M = 1000;
    int a[M][N], b[N], c[M];


    //initialization
    for( int i = 0; i < M; i++ ){ // rows
        for(int j = 0; j < N; j++){ // columns
            a[i][j] = 1; //i+j;
        }
    }

    for(int j = 0; j < N; j++){ // rows
        b[j] = 1; //j;
    }

    for(int i = 0; i < M; i++){ // rows
        c[i] = 0;
    }

    /////////////////

    double start = omp_get_wtime();

    for(int k = 1; k <= 1024; k*=2){
        // Matrix multiplication
        #pragma omp parallel num_threads(4)
        {
            //#pragma omp for
            for(int i = 0; i < M; i++){
                #pragma omp for reduction(+:c[i])
                for( int j = 0; j < N; j++ ){
                    c[i] += a[i][j] * b[j];
                }
            }
        }

        double end = omp_get_wtime();
        double elapsed = end - start;

        printf("    %d   ", k);
        printf("     %lf\n", elapsed);
    }

    return 0;
}
```

| N of threads  | Time elapsed (s) |
|:-------------:|:-------------:|
| 1             | 0.013946      |
| 2             | 0.021076      |
| 3             | 0.025707      |
| 4             | 0.055863      |
| 5             | 0.080895      |
| 6             | 0.093933      |
| 7             | 0.120436      |
| 8             | 0.141777      |
| 9             | 0.174914      |
| 10            | 0.202872      |
| 11            | 0.238566      |
| 12            | 0.275555      |

<br>

- CUDA  
We can paralellising the matrix multiplication using CUDA with this code:

```c
#include <stdio.h>
#include <stdlib.h>


__global__ void multiplication(int n, int m, int *a, int *b)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index, j = index; i < m*n; i += stride){  // T threads per iteration
        a[i] = a[i] * b[j%n];
    }
}

int main(int argc, char **argv){
    int N = 1500;
    int M = 1000;
    //int a[M][N], b[N], c[M];

    int *a, *b, *c;  // Host copies of a, b, c
    int *dev_a, *dev_b;  // Device copies of a, b


    // Create counter
    cudaEvent_t start, stop;
    float elapsedTime;

    // Allocate space for device copies a, b, c
    cudaMalloc((void **) &dev_a, sizeof(int)*N*M);
    cudaMalloc((void **) &dev_b, sizeof(int)*N);

    a = (int *) malloc(sizeof(int)*N*M);
    b = (int *) malloc(sizeof(int)*N);
    c = (int *) malloc(sizeof(int)*M);


    //initialization
    for( int i = 0; i < (M*N); i++ ){ // all lines are in sequence
        a[i] = 1; //i+1;
    }

    for(int j = 0; j < N; j++){ // rows
        b[j] = 1; //j+1;
    }

    for(int i = 0; i < M; i++){ // rows
        c[i] = 0;
    }

    /////////////////


    // Start counter
    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    // Copy inputs to device
    cudaMemcpy(dev_a, a, sizeof(int)*N*M, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int)*N, cudaMemcpyHostToDevice);


    multiplication<<<1,512>>>(N, M, dev_a, dev_b); // 1 block, N=512 threads (block size)


    // Copy result back to host
    cudaMemcpy(a, dev_a, sizeof(int)*N*M, cudaMemcpyDeviceToHost);

    // We need to sum all the columns to make the matrix C
    for(int i = 0; i < M*N; i++){ // columns
        c[i/N] += a[i];
    }

    // Stop counter
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);

    // Resut
    for(int i = 0; i < M; i++){
       printf("[ %d ]\n", c[i]);
    }
    printf("Elapsed time: %f ms\n", elapsedTime);

    // Clean up
    free(a); free(b); free(c);
    cudaFree(dev_a); cudaFree(dev_b);

    return 0;
}
```

| N of threads  | Time elapsed (ms) |
|:-------------:|:-------------:|
|   1           |   383.244690  |
|   2           |   211.815552  |
|   4           |   130.989182  |
|   8           |   91.630783   |
|   16          |   52.529568   |
|   32          |   34.464417   |
|   64          |   24.129503   |
|   128         |   19.112801   |
|   256         |   16.846720   |
|   512         |   16.146624   |
|   1024        |   15.998624   |

# Activity 3: Derivation of the Backward Difference Formula

1. Use the Taylor series expansion to derive the backward difference formula (Equation 3). Since both difference formulas give us the same truncation error, can we derive a formula for ∂φ / ∂x that gives us a smaller error? In the previous derivations, we use the Taylor series expansion about i – ∆x or i + ∆x. However, there is one other way that we can approximate the derivative, a central difference, where we use both and compute their difference.

<br>

## Forward-Time, Centered Space


# Activity 4: FTCS Test Problem

1.

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#define N_THREADS 4

int main(int argc, char **argv){

    int N = 10000;

    float alpha = 1;
    float dt = 0.0005;
    float dx = 0.01; 
    int tmax = 5000;
    float r = alpha*dt / (dx*dx);
    float phiNew[N], phiOld[N];
    int t, i;

    double time = -omp_get_wtime();    

    #pragma omp parallel for private(i,t) shared(phiNew,phiOld) num_threads(N_THREADS)
    for( t = 0; t < tmax; t++ ){
        for( i = 0; i <= N; i++ ){
            phiNew[i] = phiOld[i] + r*(phiOld[i+1] - 2*phiOld[i] + phiOld[i-1] );
        }
    }

    time += omp_get_wtime();
    printf("%f\n", time);

}
```

<br>
  
## Source Codes
The sources codes can be found in 