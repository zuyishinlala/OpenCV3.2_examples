#include <stdio.h>
#include <stdlib.h>

#define ROWS 99*99 // 80*40
#define COLS 99

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

    // Read the file and store values in the array
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
