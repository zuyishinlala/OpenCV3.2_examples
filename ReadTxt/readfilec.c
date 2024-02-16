#include <stdio.h>

#define X_SIZE 80
#define Y_SIZE 40
#define Z_SIZE 80

int main() {
    // Create a 3D array
    int array3D[X_SIZE][Y_SIZE][Z_SIZE];

    // Open the file for reading
    FILE *file = fopen("array3D.txt", "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    // Read the array elements from the file in column-major order
    for (int z = 0; z < Z_SIZE; z++) {
        for (int y = 0; y < Y_SIZE; y++) {
            for (int x = 0; x < X_SIZE; x++) {
                // Read integer from the file and store in the array
                if (fscanf(file, "%d", &array3D[x][y][z]) != 1) {
                    printf("Error reading from file!\n");
                    fclose(file);
                    return 1;
                }
            }
        }
    }

    // Close the file
    fclose(file);

    // Display some elements of the array as a test
    printf("=== First few elements of the array:\n");
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            for (int z = 0; z < 5; z++) {
                printf("%d ", array3D[x][y][z]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}
