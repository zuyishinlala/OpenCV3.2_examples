#include "./Sources/Object.h"
#include "./Sources/Parameters.h"
#include "./Sources/Input.h"
#include "./Sources/Bbox.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
int cvRound(double value) {return(ceil(value));}
// ===================================
// Find Max class probability for each row
// ===================================

static void max_classpred(float (*cls_pred)[NUM_CLASSES], float *max_predictions, int *class_index)
{
    // Obtain max_prob and the max class index
    for (int i = 0; i < ROWSIZE ; ++i)
    {
        float *predictions = &cls_pred[i][0]; // a pointer to the first column of each row
        float max_pred = 0;
        int max_class_index = -1;

        // iterate all class probability
        for (int class_idx = 0; class_idx < NUM_CLASSES; ++class_idx)
        {
            if (max_pred < predictions[class_idx])
            {
                max_pred = predictions[class_idx];
                max_class_index = class_idx;
            }
        }
        // store max data
        max_predictions[i] = max_pred;
        class_index[i] = max_class_index;
    }
    return;
}



static void swap(struct Object *a, struct Object *b)
{
    struct Object temp = *a;
    *a = *b;
    *b = temp;
}

// ===================================
// Qsort all objects by confidence
// ===================================
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
/*
// ========================================
// Calculate intersection area -- xywh
// ========================================
static float intersection_area(const struct Bbox a, const struct Bbox b) {
    float x_overlap = fmax(0, fmin(a.x + a.height / 2, b.x + b.height / 2) - fmax(a.x - a.height / 2, b.x - b.height / 2));
    float y_overlap = fmax(0, fmin(a.y + a.width / 2, b.y + b.width / 2) - fmax(a.y - a.width / 2, b.y - b.width / 2));
    return x_overlap * y_overlap;
}
*/

// ========================================
// Calculate intersection area -- xyxy(left, top, bottom, right)
// ========================================
float intersection_area(struct Bbox box1, struct Bbox box2) {
    float width = fmin(box1.right, box2.right) - fmax(box1.left, box2.left);
    float height = fmin(box1.bottom, box2.bottom) - fmax(box1.top, box2.top);
    return (width < 0 || height < 0) ?  0 : width * height;
}

static void nms_sorted_bboxes(const struct Object* faceobjects, int size, struct Object* picked_object, int *CountValidDetect) {
    /*
    - Extract box with obj confidence > conf_threshold and prob > conf_threshold
    - For prob in each grids: class prob *=  Obj confidence 
    - Find the max object probability and the class index 
    - Extract boxes with class prob > conf_threshold
    */
    if(size == 0)
        return;
    // Calculated areas
    float areas[ROWSIZE];
    /*
    for (int row_index = 0 ; row_index < size ; row_index++) {
        areas[row_index] = (faceobjects[row_index].Rect.width) * (faceobjects[row_index].Rect.height);
    }
    */
    for (int row_index = 0 ; row_index < size ; row_index++) {
        areas[row_index] = BoxArea(&faceobjects[row_index].Rect);
    }
    // ==============================
    // Fast-NMS
    // ==============================
    float maxIOU[ROWSIZE] = {0.f}; // record max value
    // Calculate IOU & record max value for every column(dp)
    for(int r = 0 ; r < size ; r++){
        for(int c = r + 1 ; c < size ; c++){

            // Calculate IOU
            float inter_area = intersection_area(faceobjects[r].Rect, faceobjects[c].Rect);
            float union_area = areas[r] + areas[c] - inter_area;
            float iou = inter_area / union_area;
            //dp, record max value
            if(iou > maxIOU[c]) 
                maxIOU[c] = iou;
        }
    }

    // Pick good instances
    for(int row_index = 0 ; row_index < size && *CountValidDetect < MAX_DETECTIONS ; row_index++){
        if(maxIOU[row_index] < NMS_THRESHOLD) // keep Object i
            picked_object[ (*CountValidDetect)++] = faceobjects[row_index];
    }
    return;
}

// ========================================
// Perform NMS
// ========================================
static void non_max_suppression_seg(struct Pred_Input *input, char *classes, struct Object *picked_objects, int* CountValidDetect, float conf_threshold)
{
    // Calculate max class and prob for each row
    float max_clsprob[ROWSIZE] = {0};
    int max_class_index[ROWSIZE] = {0};
    max_classpred(input->cls_pred, max_clsprob, max_class_index);

    struct Object candidates[ROWSIZE];
    // Count good Bboxes
    int CountValidCandid = 0;

    for (int row_index = 0; row_index < ROWSIZE; ++row_index)
    {
        if (max_clsprob[row_index] > conf_threshold)
        {
            struct Bbox box = {input->reg_pred[row_index][0], input->reg_pred[row_index][1], input->reg_pred[row_index][2], input->reg_pred[row_index][3]};
            // init an Object
            struct Object obj = {box, max_class_index[row_index], max_clsprob[row_index], &(input->seg_pred[row_index][0])};
            candidates[CountValidCandid++] = obj;
        }
    }

    

    printf("%d Candidates...\n", CountValidCandid);
    int max_wh = 4096;        // maximum box width and height
    int max_nms = 30000;      // maximum number of boxes put into torchvision.ops.nms()
    float time_limit = 10.0f; // quit the function when nms cost time exceed the limit time.
    // multi_label &= NUM_CLASSES > 1;   // multiple labels per box

    if (MULTI_LABEL)
    { // to-do
    }

    if (classes != NULL)
    { // to-do: only sort labels of these classes ( >= 1)
        //GetOnlyClass(classes, CountValidCandid, candidates);
    }

    if(CountValidCandid > ROWSIZE) CountValidCandid = ROWSIZE;
    // Sort with confidence
    qsort_inplace(candidates, 0, CountValidCandid - 1);
    nms_sorted_bboxes(candidates, CountValidCandid, picked_objects, CountValidDetect);
    return;
}

void initializePrediction(struct Pred_Input *prediction) {
    // Seed the random number generator
    srand(time(NULL));
    int n = 0;
    // Initialize cls_pred with random float values between 0 and 1
    for (int i = 0; i < ROWSIZE; i++) {
        prediction->cls_pred[i][0] = 0.2 + ((float)rand() / RAND_MAX / 3);
        prediction->cls_pred[i][1] = 0;
        prediction->cls_pred[i][2] = 0;
        prediction->cls_pred[i][3] = 0;
    }
    
    struct Bbox bbox1;
    bbox1.left = 15.0f;
    bbox1.top = 25.0f;
    bbox1.right = 55.0f;
    bbox1.bottom = 35.0f;

    
    struct Bbox bbox3;
    bbox3.left = 15.0f;
    bbox3.top = 25.0f;
    bbox3.right = 40.0f;
    bbox3.bottom = 40.0f;


    // Initialize reg_pred with random float values between 0 and 100
    for (int i = 0; i < ROWSIZE; i++) {
        for (int j = 0; j < 2; j++) {
            prediction->reg_pred[i][j] = (float)rand() / RAND_MAX * 5; // 0 to 20
            prediction->reg_pred[i][j+2] = prediction->reg_pred[i][j] + 20.f; // 
        }
    }

    // Initialize seg_pred with random float values between 0 and 1
    for (int i = 0; i < ROWSIZE; i++) {
        for (int j = 0; j < NUM_MASKS; j++) {
            prediction->seg_pred[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

int main(){
    struct Pred_Input input;
    initializePrediction(&input);
    printf("======init inputs======\n");

    struct Object ValidDetections[MAX_DETECTIONS]; 
    int NumDetections = 0;

    non_max_suppression_seg(&input, NULL, ValidDetections, &NumDetections, CONF_THRESHOLD);
    printf("NMS Done,Got %d Detections...\n", NumDetections);
}