#include <stdio.h>

void copyArray(int (*A)[3], int (*B)[3], int startRow, int numRowsB, int numCols) {
    for (int i = 0; i < numRowsB; i++) {
        for (int j = 0; j < numCols; j++) {
            *((*(A + startRow)) + j) = *((*(B + i)) + j);
        }
        A++;
    }
}

void Add10(int (*A)[3]){
    for (int i = 0; i < 5; i++) {
        int* ptr = &A[i][0];
        ptr[0] += 100;
        ptr[1] += 1000;
        ptr[2] += 10000;
    }
}

int main(int argc, char** argv) {
    int A[5][3] = {0}; // Array A of size 10x10
    int B[2][3];  // Array B of size 5x10
    // Populate array B with some values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            B[i][j] = i * 10 + j + 1; // Just an example
        }
    }

    // Copy array B into array A starting at row index 5
    copyArray(A + 3, B, 0, 3, 3);

    Add10(A);
    // Printing array A
    printf("Array A after copying array B into it:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%3d ", A[i][j]);
        }
        printf("\n");
    }
    int arr[3] = {5,6,7};
    int* ptr;
    ptr = arr;
    printf("%p\n", &arr);
    printf("Memory address pointed to by ptr: %p\n", (void *)ptr);
    return 0;
}
/*
int arr[3] = {5,6,7};
int* ptr;
ptr = arr;
printf("%p\n", &arr);
printf("Memory address pointed to by ptr: %p\n", (void *)ptr);
*/

/*
printf("Address of A: %p\n", (void *)A);

int(*ptr)[3] = A;
printf("Address of next index is %p\n", (*ptr) + 1);
printf("Address of A %p\n", *ptr);
ptr++;

printf("Address of A next row is %p", *ptr);
*/