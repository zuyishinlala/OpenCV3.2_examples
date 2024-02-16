#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Include the string.h header for strtok

#define ROWS 99
#define COLS 99*99

int main() {
    FILE *file;
    char filename[] = "array3D.txt"; // Replace with your file name
    int matrix[ROWS][COLS];
    int row = 0, col = 0;
    int value;

    // Open the file
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return 1;
    }
    

    while (fscanf(file, "%d", &value) == 1) {
        matrix[row][col] = value;
        col++;
        if (col == COLS) {
            col = 0;
            row++;
            if (row == ROWS) {
                break; // Matrix is filled, exit loop
            }
        }
    }
    // Close the file
    fclose(file);
    
    // Now you have the contents of the file stored in matrix

    // Example usage: printing the matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 100; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}
