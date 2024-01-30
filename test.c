#include<stdio.h>
#include<stdlib.h>
int* returnptr(){
    int arr[3] = {1, 2, 3};
    int* ptr = arr;
    return ptr;
}
int main() {
    int*ptr = returnptr();

    for(int i = 0 ; i < 3 ; i++){
        printf("%d ", ptr[i]);
    }
    return 0;
}