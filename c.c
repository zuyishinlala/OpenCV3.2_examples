#include <stdio.h>

#define MAX_FILENAME_LENGTH 256

void readFilename(const char *filename) {
    FILE *file;
    char buffer[MAX_FILENAME_LENGTH];

    // Open the file for reading
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    // Read the string from the file
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        // Print the read string
        printf("Read string from file: %s", buffer);
    }

    
    // Close the file
    fclose(file);
}

int main(int argc, char** argv) {
    readFilename(argv[1]);
    return 0;
}
