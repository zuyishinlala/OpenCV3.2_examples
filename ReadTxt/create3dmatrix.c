#include <stdio.h>

#define X_SIZE 80
#define Y_SIZE 40
#define Z_SIZE 80

int main() {
    // Create a 3D array
    int array3D[X_SIZE][Y_SIZE][Z_SIZE];

    // Initialize the array
    for (int x = 0; x < X_SIZE; x++) {
        for (int y = 0; y < Y_SIZE; y++) {
            for (int z = 0; z < Z_SIZE; z++) {
                // Assign some values to the array elements
                array3D[x][y][z] = x + y + z;
            }
        }
    }

    // Open a file for writing
    FILE *file = fopen("array3D.txt", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    // Write the array elements to the file
    for (int x = 0; x < X_SIZE; x++) {
        for (int y = 0; y < Y_SIZE; y++) {
            for (int z = 0; z < Z_SIZE; z++) {
                fprintf(file, "%d ", array3D[x][y][z]);
            }
            fprintf(file, "\n"); // Newline after each row in y-direction
        }
        fprintf(file, "\n"); // Extra newline after each x-z plane
    }

    // Close the file
    fclose(file);

    printf("Array data has been written to array3D.txt\n");

    return 0;
}
