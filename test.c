#include <stdio.h>

void add10(int *ptr, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            *(ptr + i * cols + j) += 10;
        }
    }
}

void add10_opt2(int *ptr, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        *ptr += 10;
        ptr++;
    }
}

void add10_opt3(int *ptr, int rows, int cols) {
    for (int i = 0; i < rows ; i++) {
        int* data = ptr;
        data[0] += 10;
        data[1] += 10;
        data[2] += 10;
        data[3] += 10;
        ptr += cols;
    }
}

void add10_opt4(int (*ptr)[4], int rows, int cols) {
    for (int i = 0; i < rows ; i++) {
        for(int j = 0 ; j < cols ; j++){
            ptr[i][j] += 10;
        }
    }
}

void add10_opt5(int (*ptr)[4], int rows, int cols) {
    for (int i = 0; i < rows ; i++) {
        int *first_column = *(ptr + i);
        for(int j = 0 ; j < cols ; j++){
           first_column[j] += 10;
        }
    }
}


int main() {
    int arr[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    // Pointer to the first row of the array
    int *ptr = &arr[0][0];

    // Get the number of rows and columns
    int rows = sizeof(arr) / sizeof(arr[0]);
    int cols = sizeof(arr[0]) / sizeof(arr[0][0]);

    add10_opt5(arr, rows, cols);

    // Print the modified array
    printf("Modified array:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }

    return 0;
}
