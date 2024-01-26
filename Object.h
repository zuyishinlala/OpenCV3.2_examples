#include<opencv2/imgcodecs/imgcodecs_c.h>
#include<stdio.h>
#include<math.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
int cvRound(double value) {return(ceil(value));}
typedef struct{
    float x, y, width, height; // center_x, center_y, width, height
} Bbox;

typedef struct{
    Bbox Rect;
    int label;
    float prob;
} Object;
