#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initial capacity of the dynamic array
    int capacity = 5;
    // Pointer to store the dynamically allocated array
    float *arr = malloc(capacity * sizeof(float));
    if (arr == NULL) {
        // Handle memory allocation failure
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Current size of the array
    int size = 0;
    printf("Original Array elements: ");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
    printf("====After====");
    // Add elements to the array within a for loop
    for (int i = 0; i < 10; i++) {
        // Check if resizing is needed
        if (size == capacity) {
            // Double the capacity
            capacity *= 2;
            // Resize the array
            float *temp = realloc(arr, capacity * sizeof(float));
            if (temp == NULL) {
                // Handle memory allocation failure
                fprintf(stderr, "Memory reallocation failed\n");
                free(arr); // Free the previously allocated memory
                return 1;
            }
            arr = temp;
        }
        // Add the new element to the array
        arr[size++] = i + 1; // Example: Adding integers 1 to 10
    }

    // Print the elements of the array
    printf("Array elements: ");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");

    // Free the dynamically allocated memory
    free(arr);

    return 0;
}
