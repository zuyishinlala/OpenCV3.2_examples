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
static inline void PreProcessing(float* Mask_Input, int* NumDetections, struct Object *ValidDetections, float (*MaskCoeffs)[NUM_MASKS], const char** argv){
    char* Bboxtype = "xyxy";
    char* classes = NULL;

    struct Pred_Input input;
    // ========================
    // Init Inputs(9 prediction input + 1 mask input) in Sources/Input.c
    // ========================
    initPredInput(&input, Mask_Input, argv);
    printf("======Init Input Complete======\n");
    sigmoid(ROWSIZE, NUM_CLASSES, &input.cls_pred[0][0]);
    post_regpreds(input.reg_pred, Bboxtype);
    print_data(Mask_Input, MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH, 4);
    //initPredInput_pesudo(&input, Mask_Input, argv);

    non_max_suppression_seg(&input, classes, ValidDetections, NumDetections, CONF_THRESHOLD);
    printf("NMS Done,Got %d Detections...\n", *NumDetections);

    CopyMaskCoeffs(MaskCoeffs, *NumDetections, ValidDetections);
    PrintObjectData(*NumDetections, ValidDetections);
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
        // May cause "SEGMENTATION FAULT" contributed by memory out of bound (> 640 or < 0) in BilinearInterpolation
        handle_proto_test(Detect, Mask_Input, Mask);
        rescalebox(&Detect->Rect, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);
        RescaleMaskandDrawMask(Detect, Mask, Img, mask_xyxy);
    }
    printf("Drew %d Masks...\n", NumDetections);

    int Thickness = (int) fmaxf(roundf((Img->width + Img->height) / 2.f * 0.003f), 2);

    for(int i = NumDetections - 1  ; i > -1 ; --i){
        //String Append Label + Confidence
        char* Label = GetClassName(ValidDetections[i].label);
        char FinalLabel[20];
        strcpy(FinalLabel, Label);
        size_t len = strlen(FinalLabel);
        FinalLabel[len] = ' ';
        sprintf(FinalLabel + len + 1, "%.2f", ValidDetections[i].conf);

        DrawLabel(ValidDetections[i].Rect, ValidDetections[i].label, FinalLabel, Thickness, TextColor, Img);
    }

    printf("Drew %d Lables...\n", NumDetections);
    printf("============Post Processing Complete.============\n");
}


int main(int argc, const char **argv)
{
    IplImage* Img = cvLoadImage( argv[1], CV_LOAD_IMAGE_COLOR);
    if(!Img){
        printf("---No Img---\n");
        return 0;
    }
    CvScalar TextColor = CV_RGB(255, 255, 255);

    float Mask_Input[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH];
    float Mask_Coeffs[MAX_DETECTIONS][NUM_MASKS];

    static uint8_t Mask[TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH] = {0};

    // Recorded Detections for NMS
    struct Object ValidDetections[MAX_DETECTIONS]; 
    int NumDetections = 0;

    // Preprocessing + NMS 
    PreProcessing(&Mask_Input[0][0], &NumDetections, ValidDetections, Mask_Coeffs, argv);
    //PrintObjectData(NumDetections, ValidDetections);

    // Store Masks Results
    PostProcessing(NumDetections, ValidDetections, Mask_Input, Img, Mask, TextColor);
    
    // ========================
    // Display Output
    // ========================
    cvNamedWindow("Final Output", CV_WINDOW_AUTOSIZE);
    cvShowImage("Final Output", Img);
    cvWaitKey(0);
    cvDestroyAllWindows();
    cvSaveImage("Result.jpg", Img, 0);
    cvReleaseImage(&Img);
    printf("Original Image Released.");
    return 0;
}
/*
After NMS output
======Index: 0======
Box Position: 496.098450, 196.839951, 558.655457, 341.662231
Confidence: 0.943630 
Label: 0 
First 3 mask coeffs: -0.002038, 0.035050, -0.093406
======Index: 1======
Box Position: 263.882141, 367.571228, 376.792908, 490.456238
Confidence: 0.913981 
Label: 16 
First 3 mask coeffs: -0.032480, -0.185846, 0.176836
======Index: 2======
Box Position: 483.988342, 342.410767, 586.628845, 431.275635
Confidence: 0.913909 
Label: 16 
First 3 mask coeffs: -0.078615, -0.105044, 0.319571
======Index: 3======
Box Position: 37.144051, 274.736267, 88.445816, 452.130920
Confidence: 0.900443 
Label: 0 
First 3 mask coeffs: 0.048116, -0.035240, -0.068219
======Index: 4======
Box Position: 159.593872, 423.177673, 212.107727, 473.202698
Confidence: 0.889873 
Label: 16 
First 3 mask coeffs: -0.180799, 0.274203, 0.087585
======Index: 5======
Box Position: 101.557365, 361.983704, 172.068848, 440.894104
Confidence: 0.882814 
Label: 16 
First 3 mask coeffs: -0.124123, 0.136503, 0.259379
======Index: 6======
Box Position: 214.535736, 301.623230, 266.269531, 382.887878
Confidence: 0.864283 
Label: 0 
First 3 mask coeffs: -0.112651, 0.309662, -0.076536
======Index: 7======
Box Position: 350.345337, 290.823669, 372.778992, 339.081116
Confidence: 0.788122 
Label: 0 
First 3 mask coeffs: -0.427509, 0.263954, -0.555481
======Index: 8======
Box Position: 336.352234, 317.848511, 362.574158, 371.942871
Confidence: 0.739948 
Label: 0 
First 3 mask coeffs: -0.114220, 0.482586, -0.248823
======Index: 9======
Box Position: 260.837372, 297.721680, 278.164764, 326.434387
Confidence: 0.422207 
Label: 0 
First 3 mask coeffs: -0.073450, 0.230403, -0.755010
======================
*/

/*
tensor([[1.48400e+03, 2.21000e+02, 1.67100e+03, 6.54000e+02, 9.43630e-01, 0.00000e+00],
        [7.89000e+02, 7.31000e+02, 1.12700e+03, 1.09900e+03, 9.13981e-01, 1.60000e+01],
        [1.44700e+03, 6.56000e+02, 1.75400e+03, 9.22000e+02, 9.13909e-01, 1.60000e+01],
        [1.11000e+02, 4.54000e+02, 2.65000e+02, 9.84000e+02, 9.00443e-01, 0.00000e+00]])
*/

/* 
384 * 640
======Index: 0======
Box Position: 496.742889, 72.740685, 558.141663, 212.283295
Confidence: 0.911777 
Label: 0 
Mask Coeffs: 0.000000, 0.036110, -0.072220, 0.409245 
======Index: 1======
Box Position: 37.394318, 148.274445, 88.559937, 321.307251
Confidence: 0.911777 
Label: 0 
Mask Coeffs: 0.048146, -0.012037, -0.048146, 0.457392 
======Index: 2======
Box Position: 484.089264, 214.694946, 586.420532, 300.281067
Confidence: 0.911777 
Label: 16 
Mask Coeffs: -0.060183, -0.144439, 0.288879, 0.132403 
======Index: 3======
Box Position: 264.952698, 241.524338, 376.237488, 356.713837
Confidence: 0.893720 
Label: 16 
Mask Coeffs: -0.043946, -0.131839, 0.219732, 0.076906 
======Index: 4======
Box Position: 159.812622, 293.394318, 212.838806, 340.838806
Confidence: 0.884338 
Label: 16 
Mask Coeffs: -0.144439, 0.373135, 0.072220, -0.252769 
======Index: 5======
Box Position: 214.509766, 174.137222, 266.605682, 251.350800
Confidence: 0.876406 
Label: 0 
Mask Coeffs: -0.096293, 0.300916, -0.036110, 0.000000 
======Index: 6======
Box Position: 101.579491, 233.485794, 173.211365, 309.769104
Confidence: 0.859139 
Label: 16 
Mask Coeffs: -0.132403, 0.132403, 0.300916, -0.060183 
======Index: 7======
Box Position: 350.157379, 163.465744, 372.843903, 211.530426
Confidence: 0.769491 
Label: 0 
Mask Coeffs: -0.427589, 0.277933, -0.545176, -0.096208 
======Index: 8======
Box Position: 337.233521, 189.623123, 361.842621, 243.071045
Confidence: 0.769491 
Label: 0 
Mask Coeffs: -0.160346, 0.491728, -0.277933, -0.256554 
======Index: 9======
Box Position: 261.463196, 170.002548, 278.381958, 198.456833
Confidence: 0.471330 
Label: 0 
Mask Coeffs: -0.085518, 0.224484, -0.801730, -0.459658 
======Index: 10======
Box Position: 498.237335, 71.470840, 557.453003, 127.994904
Confidence: 0.251498 
Label: 0 
Mask Coeffs: -0.085518, 0.352761, 0.299312, -0.577246 
======================

loutputs:
tensor([[1.49900e+03, 2.23000e+02, 1.68800e+03, 6.51000e+02, 9.11777e-01, 0.00000e+00],
        [9.00000e+01, 4.55000e+02, 2.47000e+02, 9.86000e+02, 9.11777e-01, 0.00000e+00],
        [1.46300e+03, 6.56000e+02, 1.77400e+03, 9.24000e+02, 9.11777e-01, 1.60000e+01],
        [7.88000e+02, 7.41000e+02, 1.13000e+03, 1.09400e+03, 8.93720e-01, 1.60000e+01],
        [4.66000e+02, 9.00000e+02, 6.28000e+02, 1.04600e+03, 8.84338e-01, 1.60000e+01],
        [6.33000e+02, 5.34000e+02, 7.93000e+02, 7.71000e+02, 8.76406e-01, 0.00000e+00],
        [2.87000e+02, 7.16000e+02, 5.07000e+02, 9.50000e+02, 8.59139e-01, 1.60000e+01],
        [1.05000e+03, 5.01000e+02, 1.11900e+03, 6.49000e+02, 7.69491e-01, 0.00000e+00],
        [1.01000e+03, 5.82000e+02, 1.08500e+03, 7.46000e+02, 7.69491e-01, 0.00000e+00],
        [7.77000e+02, 5.22000e+02, 8.29000e+02, 6.09000e+02, 4.71330e-01, 0.00000e+00],
        [1.50400e+03, 2.19000e+02, 1.68500e+03, 3.93000e+02, 2.51498e-01, 0.00000e+00]])
        
tensor([[ 0.00000,  0.03611, -0.07222,  0.40925],
        [ 0.04815, -0.01204, -0.04815,  0.45739],
        [-0.08426, -0.09629,  0.32499,  0.16851],
        [-0.04395, -0.13184,  0.21973,  0.07691],
        [-0.14444,  0.37314,  0.07222, -0.25277],
        [-0.09629,  0.30092, -0.03611,  0.00000],
        [-0.13240,  0.13240,  0.30092, -0.06018],
        [-0.42759,  0.27793, -0.54518, -0.09621],
        [-0.16035,  0.49173, -0.27793, -0.25655],
        [-0.08552,  0.22448, -0.80173, -0.45966],
        [-0.08552,  0.35276,  0.29931, -0.57725]])

*/