#include<stdio.h>
#include<stdlib.h>
int main() {
    float arr[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float* areas = arr;
    int size = sizeof(arr) / sizeof(float);

    for (int i = 0; i < size; ++i) {
        printf("%.2f\n", areas[i]);
    }

    arr[5] = 10.4;
    printf("=====After=====");
    for (int i = 0; i < size ; ++i) {
        printf("%.2f\n", areas[i]);
    }
    return 0;
}