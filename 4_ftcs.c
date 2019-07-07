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