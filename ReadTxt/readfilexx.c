#include <stdio.h>
#include <stdlib.h>
#include "Para.h"

#define ROWS Y_SIZE*Z_SIZE 
#define COLS X_SIZE

// Similiar to what we use !!!!!!!!!!
int main() {
    FILE *file;
    char filename[] = "array3D.txt"; // Replace with your file name
    int array[ROWS][COLS];
    int row = 0, col = 0;
    int value;

    // Open the file
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return 1;
    }
    int*ptr = &array[0][0];
    // Read the file and store values in the array
    /*
    while (fscanf(file, "%d", &value) == 1) {
        array[row][col] = value;
        row++;
        if (row == ROWS) {
            row = 0;
            col++;
            if (col == COLS) {
                break; // Array is filled, exit loop
            }
        }
    }
    */
    for(int c = 0 ; c < COLS ; ++c){
        for(int r = 0 ; r < ROWS ; ++r){
            fscanf(file, "%d", ptr+r*COLS+c);
        }
    }
    // Close the file
    fclose(file);

    // Now you have the contents of the file stored in array

    // Example usage: printing the array
    for (int i = 0; i < 3 ; i++) {
        for (int j = 0; j < COLS ; j++) {
            printf("%d ", array[i][j]);
        }
        printf("\n");
    }

    return 0;
}
