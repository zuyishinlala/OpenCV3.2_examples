#include <stdio.h>
#include <string.h>
/*
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv/highgui.h>
*/
#include "./Sources/Object.h"
#include "./Sources/Parameters.h"
#include "./Sources/Input.h"
#include "./Sources/Bbox.h"
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

static float GetPixel(int x, int y, int width, int height, float *Src){
    if(x < 0 || x >= width || y < 0 || y >= height) return 0.f;
    return *(Src + y*width + x);
}

//Align Corner = False
static void BilinearInterpolate(float *Src, uint8_t *Tar, float Threshold, struct Bbox Bound, CvSize OrgImgSize){

    float src_width = TRAINED_SIZE_WIDTH, src_height = TRAINED_SIZE_HEIGHT;
    float tar_width = OrgImgSize.width, tar_height = OrgImgSize.height;

    float r_ratio = src_height / tar_height;
    float c_ratio = src_width / tar_width;
    
    // Perform Binary Threshold only in the Bounding Box Region
    for(int r = Bound.top ; r < Bound.bottom ; ++r){
        for(int c = Bound.left ; c < Bound.right ; ++c, ++Tar){
            float PixelSum = 0.f;

            float dr = (r + 0.5) * r_ratio - 0.5;
            float dc = (c + 0.5) * c_ratio - 0.5;
            int  ir = floorf(dr), ic = floorf(dc);

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

static void handle_proto_test(const struct Object* ValidDetections, const float masks[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH], int NumDetections,  uint8_t (* UnCropedMask)[TRAINED_SIZE_HEIGHT*TRAINED_SIZE_WIDTH], CvSize OrgImg_Size)
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
        BilinearInterpolate(&pred_mask[0][0], &UnCropedMask[d][0], Binary_Thres, Bound, OrgImg_Size);
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

// Rescale 
static void rescalebox(struct Object *Detections, const int CountValidDetect, float src_size_w, float src_size_h, float tar_size_w, float tar_size_h){
    float ratio = minf(tar_size_w/src_size_w, tar_size_h/src_size_h);
    float padding_w = (src_size_w - tar_size_w * ratio) / 2, padding_h = (src_size_h - src_size_h * ratio) / 2;

    for(int i = 0 ; i < CountValidDetect ; ++i){
        struct Bbox *Box = &Detections[i].Rect;

        Box->left   = (Box->left - padding_w) / ratio;
        Box->right  = (Box->right - padding_w) / ratio;
        Box->top    = (Box->top - padding_h) / ratio;
        Box->bottom = (Box->bottom - padding_h) / ratio;

        clamp(Box, tar_size_w, tar_size_h);
    }
}

// Plot Label and Bounding Box
static void plot_box_and_label(const int* label, const struct Bbox* box, float mask_transparency, IplImage **mask, IplImage **ImgSrc){
    int boxthickness = 2;
    CvScalar BLUE = CV_RGB(50, 178, 255);
    CvScalar WHITE = CV_RGB(240, 240, 240);

    IplImage* MaskedImg = cvCloneImage(*ImgSrc);
    cvSet(MaskedImg, CV_RGB(0, 0, 255), *mask); //Specify the color

    // Draw Mask
    cvAddWeighted(*ImgSrc, 1.f - mask_transparency, MaskedImg, mask_transparency, 0, *ImgSrc);

    // Draw Bounding Box
    CvPoint tlp = cvPoint(box->left, box->top);
    CvPoint brp = cvPoint(box->right, box->bottom);
    cvRectangle(*ImgSrc, tlp, brp, BLUE, boxthickness, CV_AA, 0);

    // Draw Label
    int baseLine;
    CvSize label_size; 
    CvFont font; // font for text
    cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 0.5, 0.8, 0, 2, CV_AA);

    cvGetTextSize(label, &font, &label_size, &baseLine);
    brp = cvPoint(box->left + label_size.width, box->top + label_size.height + baseLine);
    cvRectangle(*ImgSrc, tlp, brp, BLUE, CV_FILLED, CV_AA, 0);
    cvPutText(*ImgSrc, label, cvPoint(box->left, box->top + label_size.height), &font, WHITE);
    cvReleaseImage(&MaskedImg);
}

// rescale_mask + draw label
static void RescaleMaskandDrawLabel(const struct Object *Detections, int NumDetections, const uint8_t (* UnCropedMask)[TRAINED_SIZE_HEIGHT*TRAINED_SIZE_WIDTH], IplImage** ImgSrc){
    /*
    Retrieve Real Mask of Original Mask
    Resize to Final Mask
    Draw Label
    */
    int mask_xyxy[4] = {0};             // the real mask in the resized image. left top bottom right
    getMaskxyxy(mask_xyxy, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, (*ImgSrc)->width, (*ImgSrc)->height);

    for(int i = 0 ; i < NumDetections ; ++i){

        // uint8_t array to IplImage
        IplImage* SrcMask = cvCreateImageHeader(cvSize(TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT), IPL_DEPTH_8U, 1);   
        cvSetData(SrcMask, UnCropedMask[i], SrcMask->widthStep);

        // ROI Mask Region by using maskxyxy (left, top, right ,bottom)
        CvRect roiRect = cvRect(mask_xyxy[0], mask_xyxy[1], mask_xyxy[2] - mask_xyxy[0], mask_xyxy[3] - mask_xyxy[1]); // (left, top, width, height)
        cvSetImageROI(SrcMask, roiRect);

        // Obtain ROI image
        IplImage* roiImg = cvCreateImage(cvSize(roiRect.width, roiRect.height), SrcMask->depth, 1);
        cvCopy(SrcMask, roiImg, NULL);

        // Obtain Resized Mask
        IplImage* FinalMask = cvCreateImage(cvGetSize(*ImgSrc), roiImg->depth, 1);
        cvResize(roiImg, FinalMask, CV_INTER_LINEAR);

        // Draw Label and Task
        plot_box_and_label(&Detections[i].label, &Detections[i].Rect, MASK_TRANSPARENCY, &FinalMask, ImgSrc);

        cvReleaseImage(&SrcMask);
        cvReleaseImage(&roiImg);
        cvReleaseImage(&FinalMask);
        /*
        // Uncropped Mask (haven't ROI)
        CvMat SrcMask = cvMat( TRAINED_SIZE_WIDTH,  TRAINED_SIZE_HEIGHT, CV_32FC1, UnCropedMask[i]);

        int ROI_WIDTH = mask_xyxy[2] - mask_xyxy[0], ROI_HEIGHT = mask_xyxy[3] - mask_xyxy[1];
        
        // Data to store final Mask
        float FinalMask_data[ORG_SIZE_WIDTH * ORG_SIZE_HEIGHT] = {0.f};
        CvMat FinalMask = cvMat(ORG_SIZE_WIDTH, ORG_SIZE_HEIGHT, CV_32FC1, FinalMask_data);
        */
    }
}

int main(int argc, char **argv)
{
    
    IplImage* Img = cvLoadImage( argv[1], CV_LOAD_IMAGE_COLOR);
    if(!Img){
        printf("---No Img---\n");
        return;
    }
    IplImage *Img32 = cvCreateImage(cvGetSize(Img), IPL_DEPTH_32F, 3);
    cvConvertScale(Img, Img32, 1/255.f, 0);


    char* Bboxtype = "xyxy";
    char* classes = NULL;

    // 10 Inputs (9 prediction input + 1 mask input)
    struct Pred_Input input;
    float Mask_Input[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH]; 

    // Read Inputs
    initPredInput(&input, argv);
    
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

    // Obtain uncroped mask (size : TRAINED_SIZE_HEIGHT*TRAINED_SIZE_WIDTH)
    handle_proto_test(ValidDetections, Mask_Input, NumDetections, UncropedMask, cvGetSize(Img));  //[:NumDetections] is the output
    printf("Handled_proto_test for %d predicitons\n", NumDetections);

    // Bounding Box positions to Real place
    rescalebox(ValidDetections, NumDetections, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);

    // Obtain final mask (size : ORG_SIZE_HEIGHT, ORG_SIZE_WIDTH)
    RescaleMaskandDrawLabel(ValidDetections, NumDetections, UncropedMask, &Img);
    cvReleaseImage(&Img);
    cvReleaseImage(&Img32);
}