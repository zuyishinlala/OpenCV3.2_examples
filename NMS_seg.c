#include<stdio.h>
#include "Object.h"
#include "Parameters.h"

void non_max_suppression_seg(float (*cls_pred)[NUM_CLASSES], float (* reg_pos)[4], float *max_prob, int* iscandidate, float conf_thres, float iou_thres, char* classes, int agnostic, int multi_label){
    struct Object candidates[ROWSIZE];
    
    int ValidCandid = 0;

    for(int i = 0 ; i < ROWSIZE ; ++i){
        if(max_prob[i] > conf_thres){
            // init an Object
            ++ValidCandid;
        }
    }
    int max_wh = 4096;                  // maximum box width and height
    int max_nms = 30000;                // maximum number of boxes put into torchvision.ops.nms()
    float time_limit = 10.0f;           // quit the function when nms cost time exceed the limit time.
    //multi_label &= NUM_CLASSES > 1;   // multiple labels per box

    if(multi_label){ // to-do

    }

    if(classes != NULL){ // to-do: only sort labels of this class

    }

}