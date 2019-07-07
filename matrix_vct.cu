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
    int *dev_a, *dev_b, *dev_c;  // Device copies of a, b, c


    // Create counter
    cudaEvent_t start, stop;
    float elapsedTime;

    // Allocate space for device copies a, b, c
    cudaMalloc((void **) &dev_a, sizeof(int)*N*M);
    cudaMalloc((void **) &dev_b, sizeof(int)*N);
    cudaMalloc((void **) &dev_c, sizeof(int)*M);

    a = (int *) malloc(sizeof(int)*N*M);
    b = (int *) malloc(sizeof(int)*N);
    c = (int *) malloc(sizeof(int)*M);


    //initialization
    for( int i = 0; i < (M*N); i++ ){ // all lines in sequence
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

    // 1 block, N threads (block size)
    multiplication<<<1,64>>>(N, M, dev_a, dev_b);

    // Copy result back to host
    cudaMemcpy(a, dev_a, sizeof(int)*N*M, cudaMemcpyDeviceToHost);

    // We need to sum all the columns to make the matrix C
    for(int i = 0; i < M*N; i++){ // columns
        c[i/N] += a[i];
        //printf("%d, ", a[i]);
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