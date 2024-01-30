/*
#include<opencv2/imgcodecs/imgcodecs_c.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
*/
#include "Object.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
int cvRound(double value) {return(ceil(value));}

static void swap(struct Object *a, struct Object *b) {
    struct Object temp = *a;
    *a = *b;
    *b = temp;
}

static void qsort_descent_inplace(struct Object *Objects, int left, int right) {
    int i = left;
    int j = right;
    float p = Objects[(left + right) / 2].prob;

    while (i <= j) {
        while (Objects[i].prob > p)
            i++;

        while (Objects[j].prob < p)
            j--;

        if (i <= j) {
            swap(&Objects[i], &Objects[j]);
            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(Objects, left, j);
    if (i < right)
        qsort_descent_inplace(Objects, i, right);
}

float randomFloat(float min, float max) {
    return ((float)rand() / RAND_MAX) * (max - min) + min;
}

int main(int argc, char** argv){
    srand((unsigned int)time(NULL));
    int NUMOFOBJ = 10;
    struct Object objects[NUMOFOBJ];

    for (int i = 0; i < NUMOFOBJ; i++) {
        objects[i].Rect.x = randomFloat(0.0, 100.0);
        objects[i].Rect.y = randomFloat(0.0, 100.0);
        objects[i].Rect.width = randomFloat(1.0, 10.0);
        objects[i].Rect.height = randomFloat(1.0, 10.0);
        objects[i].label = rand() % 5 + 1; 
        objects[i].prob = randomFloat(0.0, 1.0);
    }
    
    for (int i = 0; i < NUMOFOBJ; i++) {
        printf("  Probability: %.2f\n", objects[i].prob);
    }
    qsort_descent_inplace(objects, 0, NUMOFOBJ - 1);
    printf("====After====\n");

    for (int i = 0; i < NUMOFOBJ; i++) {
        printf("  Probability: %.2f\n", objects[i].prob);
    }
    return 0;
}