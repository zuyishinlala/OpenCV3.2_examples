#include <stdio.h>
#include "Object.h"
#include "Parameters.h"
#include "Input.h"

// ===================================
// Qsort all objects
// ===================================
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

// ========================================
// Perform Fast-NMS
// ========================================
static float intersection_area(const struct Bbox a, const struct Bbox b) {
    float x_overlap = fmax(0, fmin(a.x + a.height / 2, b.x + b.height / 2) - fmax(a.x - a.height / 2, b.x - b.height / 2));
    float y_overlap = fmax(0, fmin(a.y + a.width / 2, b.y + b.width / 2) - fmax(a.y - a.width / 2, b.y - b.width / 2));
    return x_overlap * y_overlap;
}

static void nms_sorted_bboxes(const struct Object* faceobjects, int size, int* picked, float nms_threshold, float conf_threshold) {
    /*
    - Extract box with obj confidence > conf_threshold and prob > conf_threshold
    - For every prob in 8500 grids: class prob *=  Obj confidence 
    - Find the max object probability and the class index 
    - Extract boxes with class prob > conf_threshold
    - Create an array to store output [ NumOfOutputs, 39]
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


static void non_max_suppression_seg(struct Pred_Input *input, float *max_prob, int max_index, float conf_thres, struct Object *objects, float iou_thres, char *classes, int agnostic, int multi_label)
{
    // Calculate Area for each grid
    float areas[ROWSIZE] = {0};
    for (int i = 0; i < ROWSIZE; ++i)
    {
        areas[i] = input->reg_pred[i][2] * input->reg_pred[i][3]; // width * height
    }

    // Pick good Bboxes
    struct Object candidates[ROWSIZE];
    int CountValidCandid = 0;
    for (int i = 0; i < ROWSIZE; ++i)
    {
        if (max_prob[i] > conf_thres)
        {
            struct Bbox box = {input->reg_pred[i][0], input->reg_pred[i][1], input->reg_pred[i][2], input->reg_pred[i][3]};
            // init an Object
            struct Object obj = {box, i, max_prob[i], &(input->seg_pred[i][0])};
            candidates[CountValidCandid++] = obj;
        }
    }

    int max_wh = 4096;        // maximum box width and height
    int max_nms = 30000;      // maximum number of boxes put into torchvision.ops.nms()
    float time_limit = 10.0f; // quit the function when nms cost time exceed the limit time.
    // multi_label &= NUM_CLASSES > 1;   // multiple labels per box

    if (multi_label)
    { // to-do
    }

    if (classes != NULL)
    { // to-do: only sort labels of this class
    }

    // Sort with confidence
    qsort_inplace(candidates, 0, CountValidCandid - 1);


}