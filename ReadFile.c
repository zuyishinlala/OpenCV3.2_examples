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

#define WIDTH0 TRAINED_SIZE_WIDTH / 8
#define WIDTH1 TRAINED_SIZE_WIDTH / 16
#define WIDTH2 TRAINED_SIZE_WIDTH / 32

#define HEIGHT0 TRAINED_SIZE_HEIGHT / 8
#define HEIGHT1 TRAINED_SIZE_HEIGHT / 16
#define HEIGHT2 TRAINED_SIZE_HEIGHT / 32

#define AGNOSTIC 0
#define MULTI_LABEL 0
#define ISSOLO 0

int main(int argc,char** argv){
    float cls_pred0[NUM_CLASSES][HEIGHT0][WIDTH0];
    float cls_pred1[NUM_CLASSES][HEIGHT1][WIDTH1];
    float cls_pred2[NUM_CLASSES][HEIGHT2][WIDTH2];

    float reg_pred0[4][HEIGHT0][WIDTH0];
    float reg_pred1[4][HEIGHT1][WIDTH1];
    float reg_pred2[4][HEIGHT2][WIDTH2];

    float seg_pred0[1+NUM_MASKS][HEIGHT0][WIDTH0];
    float seg_pred1[1+NUM_MASKS][HEIGHT1][WIDTH1];
    float seg_pred2[1+NUM_MASKS][HEIGHT2][WIDTH2];

}