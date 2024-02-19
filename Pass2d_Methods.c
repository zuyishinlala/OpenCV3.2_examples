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

void add10_opt6(int rows, int cols, int arr[rows][cols]) {
    for (int i = 0; i < rows ; i++) {
        for(int j = 0 ; j < cols ; j++){
            arr[i][j] += 10;
        }
    }
}

void add10_opt7(int arr[3][4]) {
    for (int i = 0; i < 3 ; i++) {
        for(int j = 0 ; j < 4 ; j++){
            arr[i][j] += 10;
        }
    }
}

void FindMax(int (*ptr)[4], int* maxele){ // like void max_classpred in ReadFile.c
    for (int i = 0; i < 3 ; i++) {
        int *data = &ptr[i][0];
        int max_ele = 0;
        for(int j = 0 ; j < 4 ; j++){
           if(max_ele < data[j]) max_ele = data[j];
        }
        maxele[i] = max_ele;
    }
}

void access2row(int arr[3][4]){
    int* ptr = &arr[1][0];
    for(int i = 0 ; i < 4 ; i++){
        //*ptr += 10;
        ptr[i] += 10;
    }
}

int main() {
    int arr[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    int max_ele[3] = {0};
    // Pointer to the first row of the array
    int *ptr = &arr[0][0];
    
    // Get the number of rows and columns
    int rows = sizeof(arr) / sizeof(arr[0]);
    int cols = sizeof(arr[0]) / sizeof(arr[0][0]);

    //add10_opt6(rows, cols, arr);
    add10_opt2(ptr, 3, 4);
    //access2row(arr);
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
