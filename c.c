#include <stdio.h>

int main() {
    FILE *file = fopen("ccc.txt", "r");

    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }
    long offset = 9 * sizeof(float);

    // Move file pointer to the position of the 5th integer
    fseek(file, offset, SEEK_SET);

    // Read and print integers from the 5th to the 10th
    for (int i = 0; i < 5 ; ++i) {
        float num;
        if (fread(&num, sizeof(float), 1, file) != 1) {
            printf("Error reading from file\n");
            fclose(file);
            return 1;
        }
        printf("%f\n", num);
    }

    fclose(file);
    return 0;
}
