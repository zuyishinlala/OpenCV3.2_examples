#include <stdio.h>
#include "Object.h"

#define ORG_SIZE_HEIGHT 500
#define ORG_SIZE_HEIGHT 500

#define TRAINED_SIZE_HEIGHT 384
#define TRAINED_SIZE_WIDTH 640

#define MASK_SIZE_HEIGHT TRAINED_SIZE_HEIGHT/4
#define MASK_SIZE_WIDTH TRAINED_SIZE_WIDTH/4

#define CONF_THRESHOLD 0.3
#define NMS_THRESHOLD 0.5

#define NUM_CLASSES 80
#define NUM_MASKS 32

#define WIDTH0 TRAINED_SIZE_WIDTH / 8  // 0
#define WIDTH1 TRAINED_SIZE_WIDTH / 16 // 1
#define WIDTH2 TRAINED_SIZE_WIDTH / 32 // 2

#define HEIGHT0 TRAINED_SIZE_HEIGHT / 8
#define HEIGHT1 TRAINED_SIZE_HEIGHT / 16
#define HEIGHT2 TRAINED_SIZE_HEIGHT / 32

#define ROWSIZE HEIGHT0*WIDTH0 + HEIGHT1*WIDTH1 + HEIGHT2*WIDTH2

#define AGNOSTIC 0
#define MULTI_LABEL 0
#define ISSOLO 0
#define ANCHOR_BASED 0

int main(int argc,char** argv){
    float cls_pred[ROWSIZE][NUM_CLASSES]; // class prob

    float reg_pred[ROWSIZE][4]; // position

    float seg_pred[ROWSIZE][1+NUM_MASKS]; // proto net index, mask 

    float masks[NUM_MASKS][MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH];
    /*
    ============================================
        To-do: Read files and store data into arrays
    ============================================    
    */
    sigmoid(&cls_pred, ROWSIZE, NUM_CLASSES);
}

static void sigmoid(float** arr, int rowsize, int colsize)
{
    for(int r = 0 ; r < rowsize; r++){
        for(int c = 0 ; c < colsize; c++){
            arr[r][c] = 1.0f / (1.0f + fast_exp(-arr[r][c]));
        }
    }
}