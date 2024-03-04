#include <stdio.h>
#include <stdlib.h>

#define ROWS 32
#define COLS 25600

int main() {
    // Open the file
    FILE *file = fopen("./Outputs/output3.txt", "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    // Allocate memory for the data
    float data[ROWS][COLS];
    float* ptr = &data[0][0];
    // Read data from the file
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (fscanf(file, "%f", ptr + i*COLS + j) != 1) {
                printf("Error reading file!\n");
                fclose(file);
                return 1;
            }
        }
    }
    
    // Close the file
    fclose(file);

    // Print the data (just for testing)
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%f ", data[i][j]);
        }
        printf("\n");
    }

    return 0;
}
