#include <stdio.h>
#include <omp.h>

int main(){
    int	sum = 0;
    int sum_2 = 0;
    int x = 1;
    #pragma omp parallel for collapse(2) schedule(static)
    for( int i = 0; i < 10; ++ i )
        for( int j = 0; j < 10; ++j )
            #pragma omp atomic
            sum += x;
    printf("%d\n",sum);
    printf("%d\n", sum_2);
}