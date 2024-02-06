#include <stdio.h>

void add10(int (*ptr)[4], int rowsize, int colsize) {
    for(int i = 0; i < rowsize; i++) {
        for(int j = 0; j < colsize; j++) {
            ptr[i][j] += 10;
        }
    }
}

int main() {
    int arr[2][4] = {{1, 2, 3, 4}, {3, 4, 5, 6}};
    int arr1[100][4] = {{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 10}};
    int arr2[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    int (*ptrs[3])[4] = {arr, arr1, arr2}; // Array of pointers to arrays of size 4

    for(int i = 0 ; i < 3 ; i++) {
        add10(ptrs[i], i == 0 ? 2 : i == 1 ? 3 : 4, 4); // Adjust rowsize based on the array
    }

    // Printing modified arrays
    printf("Modified arr:\n");
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }

    printf("\nModified arr1:\n");
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%d ", arr1[i][j]);
        }
        printf("\n");
    }

    printf("\nModified arr2:\n");
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%d ", arr2[i][j]);
        }
        printf("\n");
    }

    return 0;
}