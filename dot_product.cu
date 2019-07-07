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
    int N = 100;
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

    //dot_prod<<<1,N>>>(dev_a, dev_b, dev_c);  // 1 block, N threads (block size)
    dot_prod<<<1,64>>>(N, dev_a, dev_b, dev_c);

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

    for( int i = 0; i < N; i++ ){
        printf("%d ", c[i]);
    }


    printf("Dot Product: %d\n", sum);
    printf("Elapsed time: %f ms\n", elapsedTime);

    // Clean up
    free(a); free(b);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    return 0;
}