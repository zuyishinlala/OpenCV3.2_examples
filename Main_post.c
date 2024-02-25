#include <stdio.h>
#include <string.h>

/*
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
*/
#include "Object.h"
#include "Input.h"
#include "Bbox.h"
#include "Parameters.h"
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
        // pred_bboxes *= stride_tensor
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

static float GetPixel(int x, int y, int width, int height, float *Src){
    if(x < 0 || x >= width || y < 0 || y >= height) return 0.f;
    return *(Src + y*width + x);
}

//Align Corner = False
static void GetFinalMask(float *Src, uint8_t *Tar, float Threshold, struct Bbox Rect){

    float src_width = TRAINED_SIZE_WIDTH, src_height = TRAINED_SIZE_HEIGHT;
    float tar_width = ORG_SIZE_WIDTH, tar_height = ORG_SIZE_HEIGHT;

    float r_ratio = src_height / tar_height;
    float c_ratio = src_width / tar_width;
    
    // Perform Binary Threshold only in the Bounding Box Region
    for(int r = Rect.top ; r < Rect.bottom ; ++r){
        for(int c = Rect.left ; c < Rect.right ; ++c, ++Tar){
            float PixelSum = 0.f;

            float dc = (c + 0.5) * c_ratio - 0.5;
            float dr = (r + 0.5) * r_ratio - 0.5;
            int ic = floorf(dc), ir = floorf(dr);

            dr = (dr < 0.f) ? 1.0f : ((dr > src_height - 1.0f) ? 0.f : dr - ir);
            dc = (dc < 0.f) ? 1.0f : ((dc > src_width - 1.0f) ? 0.f : dc - ic);

            PixelSum =     dc *  dr * GetPixel(ic + 1, ir + 1, src_height, src_width, Src) + 
                     (1 - dc) *  dr * GetPixel(    ic, ir + 1, src_height, src_width, Src) +
                      dc * (1 - dr) * GetPixel(ic + 1,     ir, src_height, src_width, Src) +
                (1 - dc) * (1 - dr) * GetPixel(    ic,     ir, src_height, src_width, Src);
            
            *Tar = PixelSum > Threshold ? 255 : 0;
        }
    }
}

static void handle_proto_test(const struct Object* ValidDetections, const float masks[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH], int NumDetections,  uint8_t (* FinalMask)[TRAINED_SIZE_HEIGHT*TRAINED_SIZE_WIDTH])
{
    // Resize mask & Obtain Binary Mask
    // Matrix Multiplication
    /*
    - matrix multiplication. 32 * masks
    - sigmoid
    - reshape to [MASK_SIZE_HEIGHT][MASK_SIZE_WIDTH]
    - bilinear interpolate to [TRAINED_SIZE_HEIGHT][TRAINED_SIZE_WIDTH]
    - crop mask 
    - binary threshold
    */
    float Binary_Thres = 0.5;
    for(int d = 0 ; d < NumDetections ; ++d){
        float* maskcoeffs = ValidDetections[d].maskcoeff;
        float pred_mask[MASK_SIZE_HEIGHT][MASK_SIZE_WIDTH] = {0};
        float* mask_ptr = &pred_mask[0][0];

        // Matrix Multiplication
        for(int i = 0 ; i < MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH ; ++i, ++mask_ptr){
            float Pixel = 0.f;
            for(int c = 0 ; c < NUM_MASKS ; ++c){
                Pixel += maskcoeffs[c] * masks[c][i];
            }
            *mask_ptr = Pixel;
        }
        sigmoid(MASK_SIZE_HEIGHT, MASK_SIZE_WIDTH, &pred_mask[0][0]);
        
        // Crop Mask
        struct Bbox box = ValidDetections[d].Rect;
        int left = fmax(0, floorf(box.left)), top = fmax(0, floorf(box.top));
        int right = fmin(TRAINED_SIZE_WIDTH, ceilf(box.right)), bottom = fmin(TRAINED_SIZE_HEIGHT, ceilf(box.right));

        struct Bbox Bound = {left, top, right, bottom};

        // Bilinear Interpolate + Binary Threshold 
        GetFinalMask(&pred_mask[0][0], &FinalMask[d][0], Binary_Thres, Bound);
    }
}

static inline void getMaskxyxy(int* xyxy, float org_size_w, float org_size_h, float tar_size_w, float tar_size_h){
    float ratio = fminf( org_size_w/tar_size_w, org_size_h/tar_size_h);
    int padding_w = (org_size_w - tar_size_w * ratio) / 2, padding_h = (org_size_h - tar_size_h * ratio) / 2;
    xyxy[0] = padding_w;              // left
    xyxy[1] = padding_h;              // top
    xyxy[2] = org_size_w - padding_w; // right
    xyxy[3] = org_size_h - padding_h; // bottom
    return;
}

static void plot_box_draw_label(){

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
    float width = fmin(box1.right, box2.right) - fmax(box1.left, box2.left);
    float height = fmin(box1.bottom, box2.bottom) - fmax(box1.top, box2.top);
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
    { // to-do: only sort labels of this class
    }

    // Sort with confidence
    qsort_inplace(candidates, 0, CountValidCandid - 1);
    nms_sorted_bboxes(candidates, CountValidCandid, picked_objects, CountValidDetect);
    return;
}


static void RescaleMaskandDrawLabel(const uint8_t (* FinalMask)[TRAINED_SIZE_HEIGHT*TRAINED_SIZE_WIDTH], int NumDetections, IplImage* TrainedImg, int* maskxyxy){
    /*
    Retrieve Real Mask from Original Mask
    ReSize Final Mask
    Draw Label
    */
    for(int i = 0 ; i < NumDetections ; ++i){

        // uint8 to IplImage
        IplImage* SrcMask = cvCreateImageHeader(cvSize(TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT), CV_8UC1, 1);
        cvSetData(SrcMask, FinalMask[i], SrcMask->widthstep);

        //IplImage* CropedMask = cvCreateImage(cvSize(maskxyxy[2] - maskxyxy[0], maskxyxy[3] - maskxyxy[1]), CV_8UC1, 1);

        // ROI Mask Region by using maskxyxy (left, top, right ,bottom)
        CvRect roiRect = cvRect(maskxyxy[0], maskxyxy[1], maskxyxy[2] - maskxyxy[0], maskxyxy[3] - maskxyxy[1]); // (left, top, width, height)
        cvSetImageROI(SrcMask, roiRect);

        //cvCopy(Mask,CropedMask);
        // Obtain Resized Mask
        IplImage* rescaledMask = cvCreateImage(cvSize(TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT), SrcMask->depth, SrcMask->nChannels);
        cvResize(SrcMask, rescaledMask, CV_INTER_LINEAR);

        cvThreshold(rescaledMask, rescaledMask, 200, 255, CV_THRESH_BINARY);
        
        cvSet(TrainedImg, CV_RGB(0, 0, 255), resizedMask);

        cvAddWeighted(TrainedImg, 0.5, rescaledMask, 0.5, 0, TrainedImg);

        // Draw Label and Task

        cvReleaseImage(&SrcMask);
        cvReleaseImage(&resizedMask);
    }
}



int main(int argc, char **argv)
{
    /*
    IplImage* OrgImg = cvLoadImage( argv[1], CV_LOAD_IMAGE_COLOR);
     if(!OrgImg){
        printf("---No Img---\n");
        return;
    }
    */
   
    char* Bboxtype = "xyxy";
    
    // 10 Inputs (9 + 11)
    static struct Pred_Input input;
    static float Mask_Input[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH]; 

    // Read Inputs
    initPredInput(&input, argv);
    
    sigmoid(ROWSIZE, NUM_CLASSES, &input.cls_pred[0][0]);
    
    post_regpreds(input.reg_pred, Bboxtype);
    printf("Post_RegPredictions on reg_preds Done\n");

    // Recorded Detections for NMS
    struct Object ValidDetections[MAX_DETECTIONS]; 
    int NumDetections = 0;

    // Masks Results
    static uint8_t FinalMask[MAX_DETECTIONS][TRAINED_SIZE_HEIGHT*TRAINED_SIZE_WIDTH];

    non_max_suppression_seg(&input, "None", ValidDetections, &NumDetections, CONF_THRESHOLD);
    printf("NMS Done,Got %d Detections...\n", NumDetections);

    int mask_xyxy[4] = {0};             // the real mask in the resized image. left top bottom right
    getMaskxyxy(mask_xyxy, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, ORG_SIZE_WIDTH, ORG_SIZE_HEIGHT);
    printf("Got xyxy of masks\n");

    // Obtain mask with image padding
    handle_proto_test(ValidDetections, Mask_Input, NumDetections, FinalMask);  //[:NumDetections] is the output
    printf("Handled_proto_test for %d predicitons\n", NumDetections);
    //RescaleMask(FinalMask, ValidDetections, mask_xyxy);
    //cvReleaseImage(&OrgImg);
    //RescaleMaskandDrawLabel(FinalMask, NumDetections, img);
}