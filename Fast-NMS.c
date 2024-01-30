#include "Object.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
int cvRound(double value) {return(ceil(value));}

static float intersection_area(const struct Bbox a, const struct Bbox b) {
    float x_overlap = fmax(0, fmin(a.x + a.height / 2, b.x + b.height / 2) - fmax(a.x - a.height / 2, b.x - b.height / 2));
    float y_overlap = fmax(0, fmin(a.y + a.width / 2, b.y + b.width / 2) - fmax(a.y - a.width / 2, b.y - b.width / 2));
    return x_overlap * y_overlap;
}

static void nms_sorted_bboxes(const struct Object* faceobjects, int size, int* picked, float nms_threshold, float conf_threshold) {
    /*
    1. Extract box with obj confidence > conf_threshold and prob > conf_threshold
    2.
    */ 
    if(size == 0) return;
    float areas[size];
    for (int i = 0 ; i < size; i++) { // Calculate areas
        areas[i] = (faceobjects[i].Rect.width) * (faceobjects[i].Rect.height);
    }
    // Fast-NMS start
    // Initialize Matrix 
    float iouMatrix[size][size];
    float colmaxIOU[size];
    // to-do: initialization ?

    // Calculate IOU & record max value for every column(dp)
    for(int r = 0 ; r < size ; r++){
        for(int c = r + 1 ; c < size ; c++){ // Calculate IOU
            float inter_area = intersection_area(faceobjects[r].Rect, faceobjects[c].Rect);
            float union_area = areas[r] + areas[c] - inter_area;
            float iou = inter_area / union_area;
            iouMatrix[r][c] = iou;
            if(iouMatrix[r][c] > colmaxIOU[c]) //dp, update max value for every column
                colmaxIOU[c] = iou;
        }
    }

    // Picked good instances
    for(int i = 0 ; i < size ; i++){
        if(colmaxIOU[i] < nms_threshold)
            picked[i] = 1;
    }
    /*
    for (i = 0 ; i < n ; i++) {
        int keep = 1;
        for (j = 0 ; j < picked_count; j++) {
            float inter_area = intersection_area(faceobjects[i].Rect, faceobjects[picked[j]].Rect);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep) {
            picked[picked_count++] = i;
        }
    }
    */
}

int main(){
    
}