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


    for(int k = 1; k <= 12; k++){
        // Matrix multiplication
        #pragma omp parallel num_threads(k)
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

    // Resut
    //for(int i = 0; i < M; i++){
    //    printf("[ %d ]\n", c[i]);
    //}
    //printf("Time: %lf\n", elapsed);
    return 0;
}