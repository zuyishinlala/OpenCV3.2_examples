#include <stdio.h>

int powerOf2Minus1(int exponent) {
    // Calculate 2 raised to the power of 'exponent'
    int power = 1LL << exponent;
    // Subtract 1 from the calculated power
    return power - 1;
}

int main() {
    int exponent = 5; // Example exponent
    int result = powerOf2Minus1(exponent);
    printf("2 raised to the power of %d minus 1 is: %d\n", exponent, result);
    printf("%d\n", (128 << 1));
    return 0;
}
