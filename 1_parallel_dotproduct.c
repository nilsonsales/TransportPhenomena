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

    for(int j = 1; j <= 12; j++){

        double start = omp_get_wtime();
        #pragma omp parallel num_threads(j)
        {
            //dot product
            #pragma omp for reduction(+:sum)
            for( int i = 0; i < N; i++ ){
                sum += a[i] * b[i];
            }
        }
        double end = omp_get_wtime();
        double elapsed = end - start;

        printf("    %d   ", j);
        printf("     %lf\n", elapsed);
    }

    free(a);
    free(b);

    return 0;
}