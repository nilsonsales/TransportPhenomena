#include <omp.h>
#include <stdio.h>

int main(){
    #pragma omp parallel num_threads(2)
    {
        int N = 10;

        #pragma omp for
        for(int i = 0; i < N; i++){
            //work to do
            printf("%d\n", i);
        }
    }
}