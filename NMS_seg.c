#include<stdio.h>
#include"Object.h"

void non_max_suppression_seg(float (*cls_pred)[NUM_CLASSES], float *conf, float conf_thres, float iou_thres,char* classes, int agnostic, int multi_label, int max_det){
    int picked[ROWSIZE] = {0};
}