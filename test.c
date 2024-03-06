#include <stdio.h>
#include <string.h>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv/highgui.h>

#include "./Sources/Object.h"
#include "./Sources/Parameters.h"
#include "./Sources/Input.h"
#include "./Sources/Bbox.h"

#define size 10
#include <math.h>
int cvRound(double value) {return(ceil(value));}

void reviseMask(uint8_t* arr){
    memset(arr, 0, sizeof(uint8_t)* size);
    for(int i = 0 ; i < size ; ++i){
        printf("%d, ", arr[i]);
    }
}

int main(){
    static uint8_t  arr[size] = {1, 2, 3, 4, 5};
    for(int i = 0 ; i < size ; ++i){
        printf("%d, ", arr[i]);
    }
    printf("\n");
    printf("=====After=====\n");
    reviseMask(arr);
    return 0;
}