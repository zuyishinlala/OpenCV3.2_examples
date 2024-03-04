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

#include "Post_NMS.c"
#include <math.h>
int cvRound(double value) {return(ceil(value));}

static void sigmoid(int rowsize, int colsize, float *ptr)
{
    for (int i = 0; i < rowsize * colsize; ++i, ++ptr)
    {
        *ptr = 1.0f / (1.0f + powf(2.71828182846, -*ptr));
    }
}

static void post_regpreds(float (*distance)[4], char *type)
{
    // dist2bbox & generate_anchor in YOLOv6
    int row = 0;
    float stride = 8.f;
    float row_bound = HEIGHT0, col_bound = WIDTH0;
    for (int stride_index = 0; stride_index < 3; ++stride_index)
    {
        for (float anchor_points_r = 0.5; anchor_points_r < row_bound; ++anchor_points_r)
        {
            for (float anchor_points_c = 0.5; anchor_points_c < col_bound; ++anchor_points_c)
            {
                float *data = &distance[row][0]; // left, top, right, bottom

                // lt, rb = torch.split(distance, 2, -1)

                // x1y1 = anchor_points - lt
                data[0] = anchor_points_c - data[0];
                data[1] = anchor_points_r - data[1];

                // x2y2 = anchor_points + rb
                data[2] += anchor_points_c; // anchor_points_c + data[2]
                data[3] += anchor_points_r; // anchor_points_r + data[3]
                
                ++row;
            }
        }
        if (!strcmp(type,"xywh"))
        {
            // c_xy = (x1y1 + x2y2) / 2
            // wh = x2y2 - x1y1
            for (int i = 0; i < ROWSIZE; ++i)
            {
                float x1 = distance[i][0], y1 = distance[i][1], x2 = distance[i][2], y2 = distance[i][3];
                distance[i][0] = (x2 + x1) / 2; // center_x
                distance[i][1] = (y2 + y1) / 2; // center_y
                distance[i][2] = (x2 - x1);     // width
                distance[i][3] = (y2 - y1);     // height
            }
        }
        for (int i = 0; i < ROWSIZE; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                distance[i][j] *= stride;
            }
        }
        row_bound /= 2;
        col_bound /= 2;
        stride *= 2;
    }
}

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
    float width = fminf(box1.right, box2.right) - fmaxf(box1.left, box2.left);
    float height = fminf(box1.bottom, box2.bottom) - fmaxf(box1.top, box2.top);
    return (width < 0 || height < 0) ?  0 : width * height;
}

// ========================================
// Perform NMS
// ========================================
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

static void GetOnlyClass(char* className, int *Candid, struct Object* candidates){
    int ValidCandid = 0;
    for(int i = 0 ; i < *Candid ; ++i){
        if(!strcmp(candidates[i].label, className)){   // If same
            swap(&candidates[i], &candidates[ValidCandid++]);
        }
    }
    *Candid = ValidCandid;
}

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
        GetOnlyClass(classes, CountValidCandid, candidates);
    }
    if(CountValidCandid > ROWSIZE) CountValidCandid = ROWSIZE;

    // Sort with confidence
    qsort_inplace(candidates, 0, CountValidCandid - 1);
    nms_sorted_bboxes(candidates, CountValidCandid, picked_objects, CountValidDetect);
    return;
}


int main(int argc, char **argv)
{
    
    IplImage* Img = cvLoadImage( argv[1], CV_LOAD_IMAGE_COLOR);
    if(!Img){
        printf("---No Img---\n");
        return;
    }
    //IplImage *Img32 = cvCreateImage(cvGetSize(Img), IPL_DEPTH_32F, 3);
    //cvConvertScale(Img, Img32, 1/255.f, 0);


    char* Bboxtype = "xyxy";
    char* classes = NULL;

    // ========================
    // 10 Inputs (9 prediction input + 1 mask input)
    // ========================
    struct Pred_Input input;
    float Mask_Input[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH]; 

    // ========================
    // Init Inputs in Sources/Input.c
    // ========================
    initPredInput(&input, &Mask_Input[0][0], argv);
    
    sigmoid(ROWSIZE, NUM_CLASSES, &input.cls_pred[0][0]);
    
    post_regpreds(input.reg_pred, Bboxtype);
    printf("Post_RegPredictions on reg_preds Done\n");

    // Recorded Detections for NMS
    struct Object ValidDetections[MAX_DETECTIONS]; 
    int NumDetections = 0;

    non_max_suppression_seg(&input, classes, ValidDetections, &NumDetections, CONF_THRESHOLD);
    printf("NMS Done,Got %d Detections...\n", NumDetections);

    // Store Masks Results
    static uint8_t UncropedMask[MAX_DETECTIONS][TRAINED_SIZE_HEIGHT*TRAINED_SIZE_WIDTH] = {0};

    // ========================
    // Post NMS in Post_NMS.c
    // ========================
    handle_proto_test(ValidDetections, Mask_Input, NumDetections, UncropedMask, cvGetSize(Img)); 
    printf("Handled_proto_test for %d predicitons\n", NumDetections);

    // Bounding Box positions to Real place
    rescalebox(ValidDetections, NumDetections, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);

    // Obtain final mask (size : ORG_SIZE_HEIGHT, ORG_SIZE_WIDTH)
    RescaleMaskandDrawLabel(ValidDetections, NumDetections, UncropedMask, &Img);

    // ========================
    // Display Output
    // ========================
    cvNamedWindow("Output", CV_WINDOW_AUTOSIZE);
    cvShowImage("Output", Img);

    // Wait for a key event and close the window
    cvWaitKey(0);
    cvDestroyAllWindows();

    cvReleaseImage(&Img);
}