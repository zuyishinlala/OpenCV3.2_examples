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

// dist2bbox & generate_anchor in YOLOv6
static void post_regpreds(float (*distance)[4], char *type)
{
    int row = 0;
    float stride = 8.f;
    int row_bound = HEIGHT0, col_bound = WIDTH0;
    int prev_row_index = 0;
    for (int stride_index = 0; stride_index < 3; ++stride_index)
    {
        for (int anchor_points_y = 0 ; anchor_points_y < row_bound; ++anchor_points_y)
        {
            for (int anchor_points_x = 0 ; anchor_points_x < col_bound; ++anchor_points_x)
            {
                float *data = &distance[row][0]; // left, top, right, bottom

                // lt, rb = torch.split(distance, 2, -1)
                // x1y1 = anchor_points - lt
                data[0] = (float)anchor_points_x - data[0] + 0.5f;
                data[1] = (float)anchor_points_y - data[1] + 0.5f;

                // x2y2 = anchor_points + rb
                data[2] += (float)anchor_points_x + 0.5f; // anchor_points_c + data[2]
                data[3] += (float)anchor_points_y + 0.5f; // anchor_points_r + data[3]
                
                ++row;
            }
        }
        if (!strcmp(type,"xywh"))
        {
            // c_xy = (x1y1 + x2y2) / 2
            // wh = x2y2 - x1y1
            for (int i = 0; i < row_bound ; ++i)
            {
                float x1 = distance[i][0], y1 = distance[i][1], x2 = distance[i][2], y2 = distance[i][3];
                distance[i][0] = (x2 + x1) / 2; // center_x
                distance[i][1] = (y2 + y1) / 2; // center_y
                distance[i][2] = (x2 - x1);     // width
                distance[i][3] = (y2 - y1);     // height
            }
        }
        for (int i = prev_row_index ; i < row ; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                distance[i][j] *= stride;
            }
        }
        prev_row_index = row;
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

// ========================================
// Calculate intersection area -- xywh
// ========================================
/*
static float intersection_area(const struct Bbox a, const struct Bbox b) {
    float x_overlap = fmax(0, fmin(a.x + a.height / 2, b.x + b.height / 2) - fmax(a.x - a.height / 2, b.x - b.height / 2));
    float y_overlap = fmax(0, fmin(a.y + a.width / 2, b.y + b.width / 2) - fmax(a.y - a.width / 2, b.y - b.width / 2));
    return x_overlap * y_overlap;
}
*/

// ========================================
// Calculate intersection area -- xyxy(left, top, bottom, right)
// ========================================
static float intersection_area(struct Bbox box1, struct Bbox box2) {
    float width = fminf(box1.right, box2.right) - fmaxf(box1.left, box2.left);
    float height = fminf(box1.bottom, box2.bottom) - fmaxf(box1.top, box2.top);
    return (width < 0 || height < 0) ?  0 : width * height;
}

// ========================================
// Perform NMS
// ========================================
static void nms_sorted_bboxes(const struct Object* faceobjects, int size, struct Object* picked_object, int *CountValidDetect) {
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
/*
static void GetOnlyClass(char* className, int *Candid, struct Object* candidates){
    int ValidCandid = 0;
    for(int i = 0 ; i < *Candid ; ++i){
        if(!strcmp(candidates[i].label, className)){   // If same
            swap(&candidates[i], &candidates[ValidCandid++]);
        }
    }
    *Candid = ValidCandid;
}
*/
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

    printf("Found %d candidates...\n", CountValidCandid);
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

static void CopyMaskCoeffs(float (*DstCoeffs)[32], const int NumDetections, struct Object *ValidDetections){
    for(int i = 0 ; i < NumDetections ; ++i){
        memcpy(DstCoeffs[i], ValidDetections[i].maskcoeff, sizeof(float) * NUM_MASKS);
        ValidDetections[i].maskcoeff = &DstCoeffs[i][0];
    }
}

static void print_data(float *data,int colsize, int size){
    for(int i = 0 ; i < size ; ++i){
        for(int c = 0 ; c < 4 ; ++c){
            printf("%f, ", *(data + i*colsize + c));
        }
        printf("\n");
    }
    printf("\n");
}

static inline void PrintObjectData(int NumDetections, struct Object* ValidDetections){
    for(int i = 0 ; i < NumDetections ; ++i){
        printf("======Index: %d======\n", i);
        printf("Box Position: %f, %f, %f, %f\n",ValidDetections[i].Rect.left, ValidDetections[i].Rect.top, ValidDetections[i].Rect.right, ValidDetections[i].Rect.bottom);
        printf("Confidence: %f \n",ValidDetections[i].conf);
        printf("Label: %d \n",ValidDetections[i].label);
        printf("Mask Coeffs: %f, %f, %f, %f \n", ValidDetections[i].maskcoeff[0], ValidDetections[i].maskcoeff[1], ValidDetections[i].maskcoeff[2], ValidDetections[i].maskcoeff[3]);
    }
    printf("======================\n");
}

// Read Inputs + Pre-process of Inputs + NMS
static inline void PreProcessing(float* Mask_Input, int* NumDetections, struct Object *ValidDetections, float (*MaskCoeffs)[NUM_MASKS], const char** argv, int ImageIndex){
    char* Bboxtype = "xyxy";
    char* classes = NULL;

    struct Pred_Input input;
    // ========================
    // Init Inputs(9 prediction input + 1 mask input) in Sources/Input.c
    // ========================
    initPredInput(&input, Mask_Input, argv, ImageIndex);
    sigmoid(ROWSIZE, NUM_CLASSES, &input.cls_pred[0][0]);
    post_regpreds(input.reg_pred, Bboxtype);

    non_max_suppression_seg(&input, classes, ValidDetections, NumDetections, CONF_THRESHOLD);
    printf("NMS Done,Got %d Detections...\n", *NumDetections);

    CopyMaskCoeffs(MaskCoeffs, *NumDetections, ValidDetections);
    //PrintObjectData(*NumDetections, ValidDetections);
}



// Post NMS(Rescale Mask, Draw Label) in Post_NMS.c
static inline void PostProcessing(const int NumDetections, struct Object *ValidDetections, const float Mask_Input[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH], IplImage* Img, uint8_t* Mask, CvScalar TextColor){
    printf("Drawing Labels and Segments...\n");

    int mask_xyxy[4] = {0};             // the real mask in the resized image. left top bottom right
    getMaskxyxy(mask_xyxy,  TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);
    int mask_size_w = TRAINED_SIZE_WIDTH;
    int mask_size_h = TRAINED_SIZE_HEIGHT;

    for(int i = 0 ; i < NumDetections ; ++i){
        memset(Mask, 0, mask_size_w * mask_size_h * sizeof(uint8_t));
        struct Object* Detect = &ValidDetections[i];
        handle_proto_test(Detect, Mask_Input, Mask);
        rescalebox(&Detect->Rect, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);
        RescaleMaskandDrawMask(Detect, Mask, Img, mask_xyxy);
    }

    int Thickness = (int) fmaxf(roundf((Img->width + Img->height) / 2.f * 0.003f), 2);

    for(int i = NumDetections - 1  ; i > -1 ; --i){
        DrawLabel(ValidDetections[i].Rect, ValidDetections[i].conf, ValidDetections[i].label, Thickness, TextColor, Img);
    }
}

int main(int argc, const char **argv)
{

    FILE *ImageDataFile;
    char NameBuffer[MAX_FILENAME_LENGTH];

    CvScalar TextColor = CV_RGB(255, 255, 255);
    static uint8_t Mask[TRAINED_SIZE_WIDTH * TRAINED_SIZE_HEIGHT] = {0};


    // Open the file for reading
    ImageDataFile = fopen(argv[1], "r");
    if (ImageDataFile == NULL) {
        printf("Error opening file %s\n", argv[1]);
        return 0;
    }
    int ImageCount = 0;
    // Read the string from the file
    while (fgets(NameBuffer, sizeof(NameBuffer), ImageDataFile) != NULL) {
        char FinalDirectory[MAX_FILENAME_LENGTH] = "./Results/result";
        NameBuffer[strcspn(NameBuffer, "\n")] = '\0';

        IplImage* Img = cvLoadImage(NameBuffer, CV_LOAD_IMAGE_COLOR);
        if(!Img){
            printf("%s not found\n", NameBuffer);
            return 0;
        }
        printf("===============Reading Image: %s ===============\n", NameBuffer);
        float Mask_Input[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH];
        float Mask_Coeffs[MAX_DETECTIONS][NUM_MASKS];

        struct Object ValidDetections[MAX_DETECTIONS]; 
        int NumDetections = 0;

        // Preprocessing + NMS 
        PreProcessing(&Mask_Input[0][0], &NumDetections, ValidDetections, Mask_Coeffs, argv, ImageCount);

        // Store Masks Results
        PostProcessing(NumDetections, ValidDetections, Mask_Input, Img, Mask, TextColor);


        // Save Images
        char* BaseName = strrchr(NameBuffer, '/');
        if(BaseName != NULL){
            BaseName++;
        }
        strcat(FinalDirectory, BaseName);
        cvSaveImage(FinalDirectory, Img, 0);
        cvReleaseImage(&Img);
        printf("===============%s Complete===============\n", NameBuffer);
        printf("\n\n");
        ++ImageCount;
    }
    // Close the file
    fclose(ImageDataFile);

    return 0;
}
/* 
Type:
gcc main.c -o T ./Sources/Input.c ./Sources/Bbox.c  `pkg-config --cflags --libs opencv` -lm
./T ./ImgData.txt ./outputs/cls_preds8.txt ./outputs/cls_preds16.txt ./outputs/cls_preds32.txt ./outputs/reg_preds8.txt ./outputs/reg_preds16.txt ./outputs/reg_preds32.txt ./outputs/seg_preds8.txt ./outputs/seg_preds16.txt ./outputs/seg_preds32.txt ./outputs/mask_input.txt
*/