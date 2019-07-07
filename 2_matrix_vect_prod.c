#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


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