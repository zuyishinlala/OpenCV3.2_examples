#include "./Sources/Object.h"
#include "./Sources/Parameters.h"
#include "./Sources/Input.h"
#include "./Sources/Bbox.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
int cvRound(double value) {return(ceil(value));}

static void swap(struct Object *a, struct Object *b)
{
    struct Object temp = *a;
    *a = *b;
    *b = temp;
}

static void qsort_inplace(struct Object *Objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = Objects[(left + right) / 2].conf;

    while (i <= j)
    {
        while (Objects[i].conf > p)
            i++;

        while (Objects[j].conf < p)
            j--;

        if (i <= j)
        {
            swap(&Objects[i], &Objects[j]);
            i++;
            j--;
        }
    }

    if (left < j)
        qsort_inplace(Objects, left, j);
    if (i < right)
        qsort_inplace(Objects, i, right);
}

float randomFloat(float min, float max)
{
    return ((float)rand() / RAND_MAX) * (max - min) + min;
}

int main(int argc, char **argv)
{
    srand((unsigned int)time(NULL));
    int NUMOFOBJ = 10;
    struct Object objects[NUMOFOBJ];

    for (int i = 0; i < NUMOFOBJ; i++)
    {
        objects[i].label = rand() % 5 + 1;
        objects[i].conf =  randomFloat(1.0, 10.0);
    }
    for(int i = 0 ; i < 10 ; ++i){
        printf("%f,", objects[i].conf);
    }
    printf("\n");

    qsort_inplace(objects, 0, NUMOFOBJ - 1);

    printf("====After====\n");
    for(int i = 0 ; i < 10 ; ++i){
        printf("%f,", objects[i].conf);
    }
    printf("\n");

    return 0;
}