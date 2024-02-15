#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 50000

// Struct with 10 attributes
struct MultiAttribute {
    int attr1;
    int attr2;
    int attr3;
    int attr4;
    int attr5;
    int attr6;
    int attr7;
    int attr8;
    int attr9;
    int attr10;
};

// Struct with 1 attribute
struct SingleAttribute {
    int attr;
};

// Comparison function for sorting MultiAttribute structs based on attr1
int compareMulti(const void *a, const void *b) {
    const struct MultiAttribute *x = (const struct MultiAttribute *)a;
    const struct MultiAttribute *y = (const struct MultiAttribute *)b;
    return x->attr1 - y->attr1;
}

// Comparison function for sorting SingleAttribute structs
int compareSingle(const void *a, const void *b) {
    const struct SingleAttribute *x = (const struct SingleAttribute *)a;
    const struct SingleAttribute *y = (const struct SingleAttribute *)b;
    return x->attr - y->attr;
}

// Function to initialize MultiAttribute struct with random values
void initializeMulti(struct MultiAttribute *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i].attr1 = rand();
        array[i].attr2 = rand();
        array[i].attr3 = rand();
        array[i].attr4 = rand();
        array[i].attr5 = rand();
        array[i].attr6 = rand();
        array[i].attr7 = rand();
        array[i].attr8 = rand();
        array[i].attr9 = rand();
        array[i].attr10 = rand();
    }
}

// Function to initialize SingleAttribute struct with random values
void initializeSingle(struct SingleAttribute *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i].attr = rand();
    }
}

// Function to measure the time taken to sort an array
double measureSortTime(void (*sortFunction)(void *, size_t, size_t, int (*)(const void *, const void *)), void *array, size_t count, size_t size, int (*compareFunction)(const void *, const void *)) {
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    sortFunction(array, count, size, compareFunction);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    return cpu_time_used;
}

int main() {
    // Initialize random seed
    srand(time(NULL));

    // Allocate memory for arrays
    struct MultiAttribute multiArray[ARRAY_SIZE];
    struct SingleAttribute singleArray[ARRAY_SIZE];

    // Initialize arrays with random values
    initializeMulti(multiArray, ARRAY_SIZE);
    initializeSingle(singleArray, ARRAY_SIZE);

    // Measure and print sorting time for MultiAttribute array
    printf("Sorting MultiAttribute array...\n");
    double multiSortTime = measureSortTime(qsort, multiArray, ARRAY_SIZE, sizeof(struct MultiAttribute), compareMulti);
    printf("MultiAttribute array sorted in %.6f seconds\n", multiSortTime);

    // Measure and print sorting time for SingleAttribute array
    printf("Sorting SingleAttribute array...\n");
    double singleSortTime = measureSortTime(qsort, singleArray, ARRAY_SIZE, sizeof(struct SingleAttribute), compareSingle);
    printf("SingleAttribute array sorted in %.6f seconds\n", singleSortTime);

    return 0;
}
