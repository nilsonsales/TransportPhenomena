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

    double start = omp_get_wtime();

    //dot product
    for( int i = 0; i < N; i++ ){
        sum += a[i] * b[i];
    }

    double end = omp_get_wtime();
    double elapsed = end - start;

    printf("Dot Product: %d\n", sum);
    printf("Time: %lf\n", elapsed);
    return 0;
}