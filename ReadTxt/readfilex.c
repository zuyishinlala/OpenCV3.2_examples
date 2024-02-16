#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Include the string.h header for strtok

#define ROWS 80
#define COLS 3200

int main() {
    FILE *file;
    char filename[] = "array3D.txt"; // Replace with your file name
    char line[COLS * 100]; // Assuming each element has at most 2 characters
    int matrix[ROWS][COLS];
    int row = 0, col = 0;
    int value;

    // Open the file
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return 1;
    }
    
    // Read the file line by line
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
    printf("===End===");
    // Close the file
    fclose(file);
    printf("===Ended===");
    
    // Now you have the contents of the file stored in matrix

    // Example usage: printing the matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}
