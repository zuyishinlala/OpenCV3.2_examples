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

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

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
static int nextPowerOf2(int n) {
    int exponent = 0;
    while ((1 << exponent) <= n) {
        exponent++;
    }
    return exponent;
}

// Perform Sorting + NMS
static void non_max_suppression_seg(struct Pred_Input *input, char *classes, struct Object *picked_objects, int* CountValidDetect, float conf_threshold)
{
    // Calculate max class and prob for each row
    float max_clsprob[ROWSIZE] = {0};
    int max_class_index[ROWSIZE] = {0};
    max_classpred(input->cls_pred, max_clsprob, max_class_index);

    struct Object candidates[ROWSIZE * NUM_CLASSES];
    int max_wh = 4096;        // maximum box width and height
    int max_nms = 30000;      // maximum number of boxes put into torchvision.ops.nms()
    float time_limit = 10.0f; // quit the function when nms cost time exceed the limit time.
    // multi_label &= NUM_CLASSES > 1;   // multiple labels per box
    
    // Count good Bboxes
    int CountValidCandid = 0;

    int shift_num = nextPowerOf2(ROWSIZE);
    int bitwise_num = (1 << shift_num) - 1;
    
    if(AGNOSTIC)
    {
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
    }else{
        if(MULTI_LABEL){
            for (int row_index = 0; row_index < ROWSIZE; ++row_index)
            {
                for(int class = 0 ; class < NUM_CLASSES ; ++class){
                    if(input->cls_pred[row_index][class] > conf_threshold){
                        float enlarge_factor = class * max_wh;
                        struct Bbox box = {input->reg_pred[row_index][0] + enlarge_factor, 
                                        input->reg_pred[row_index][1] + enlarge_factor,
                                        input->reg_pred[row_index][2] + enlarge_factor, 
                                        input->reg_pred[row_index][3] + enlarge_factor};

                        int new_row_index = row_index | (class << shift_num);
                        // Temporary obj
                        struct Object obj = {box, new_row_index, max_clsprob[row_index], &(input->seg_pred[row_index][0])};
                        candidates[CountValidCandid++] = obj;
                    }
                }
            }
        }else{
            for (int row_index = 0; row_index < ROWSIZE; ++row_index)
            {
                if (max_clsprob[row_index] > conf_threshold)
                {
                    float enlarge_factor = max_class_index[row_index] * max_wh;
                    struct Bbox box = {input->reg_pred[row_index][0] + enlarge_factor, 
                                    input->reg_pred[row_index][1] + enlarge_factor,
                                    input->reg_pred[row_index][2] + enlarge_factor, 
                                    input->reg_pred[row_index][3] + enlarge_factor};
                    // Temporary obj
                    struct Object obj = {box, row_index, max_clsprob[row_index], &(input->seg_pred[row_index][0])};
                    candidates[CountValidCandid++] = obj;
                }
            }
        }
    }
    
    printf("Found %d candidates...\n", CountValidCandid);

    
    if (classes != NULL)
    {   //to-do: only sort labels of these classes ( >= 1)
        //GetOnlyClass(classes, CountValidCandid, candidates);
    }

    // Sort with confidence
    qsort_inplace(candidates, 0, CountValidCandid - 1);
    if(CountValidCandid > max_nms) CountValidCandid = max_nms;
    nms_sorted_bboxes(candidates, CountValidCandid, picked_objects, CountValidDetect);

    if(!AGNOSTIC){
        for(int i = 0 ; i < *CountValidDetect ; ++i){
            int row_index = picked_objects[i].label;
            int real_class = max_class_index[row_index];
            if(MULTI_LABEL){
                real_class = row_index >> shift_num;
                row_index = row_index & bitwise_num;
            }
            struct Bbox real_box = {input->reg_pred[row_index][0], input->reg_pred[row_index][1], input->reg_pred[row_index][2], input->reg_pred[row_index][3]};
            picked_objects[i].label = real_class;
            picked_objects[i].Rect = real_box;
        }
    }
    return;
}

static void CopyMaskCoeffs(float (*DstCoeffs)[NUM_MASKS], const int NumDetections, struct Object *ValidDetections){
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
static inline void PostProcessing(const int NumDetections, struct Object *ValidDetections, const float Mask_Input[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH], IplImage* Img, uint8_t(* Mask)[TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH], CvScalar TextColor){
    printf("Drawing Labels and Segments...\n");

    int mask_xyxy[4] = {0};             // the real mask in the resized image. left top bottom right
    getMaskxyxy(mask_xyxy,  TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);
    
    for(int i = 0 ; i < NumDetections ; ++i){
        memset(Mask[i], 0,  sizeof(uint8_t) * TRAINED_SIZE_WIDTH * TRAINED_SIZE_HEIGHT);
        struct Object* Detect = &ValidDetections[i];
        handle_proto_test(Detect, Mask_Input,Mask[i]);
        rescalebox(&Detect->Rect, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);
    }

    int Thickness = (int) fmaxf(roundf((Img->width + Img->height) / 2.f * 0.003f), 2);

    for(int i = NumDetections - 1 ; i > -1 ; --i){
        struct Object* Detect = &ValidDetections[i];
        RescaleMaskandDrawMask(Detect->label, Mask[i], Img, mask_xyxy);
        DrawLabel(Detect->Rect, Detect->conf, Detect->label, Thickness, TextColor, Img);
    }
}


// Post NMS(Rescale Mask, Draw Label) in Post_NMS.c
static inline void PostProcessingSaveMask(const int NumDetections, struct Object *ValidDetections, const float Mask_Input[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH],
                                          IplImage* Img, uint8_t(* Mask)[TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH], CvScalar TextColor,
                                        uint8_t(* OverLapMask)[TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH], int* HASMASKS){
    printf("Drawing Labels and Segments...\n");

    int mask_xyxy[4] = {0};             // the real mask in the resized image. left top bottom right
    getMaskxyxy(mask_xyxy,  TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);
    
    for(int i = 0 ; i < NumDetections ; ++i){
        memset( Mask[i], 0,  sizeof(uint8_t) * TRAINED_SIZE_WIDTH * TRAINED_SIZE_HEIGHT);
        struct Object* Detect = &ValidDetections[i];
        const int Class_Index = Detect->label;
        handle_proto_test(Detect, Mask_Input, Mask[i]);
        rescalebox(&Detect->Rect, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);
    }
    
    // OverLap Masks
    for(int i = 0 ; i < NumDetections ; ++i){
        int Label = ValidDetections[i].label;
        for(int r = 0 ; r < TRAINED_SIZE_HEIGHT ; ++r){
            for(int c = 0 ; c < TRAINED_SIZE_WIDTH ; ++c){
                OverLapMask[Label][r*TRAINED_SIZE_WIDTH + c] |= Mask[i][r*TRAINED_SIZE_WIDTH + c];
            }
        }
        HASMASKS[Label] = 1;
    }

    int Thickness = (int) fmaxf(roundf((Img->width + Img->height) / 2.f * 0.003f), 2);
    
    for(int i = 0 ; i < NUM_CLASSES ; ++i){
        if(HASMASKS[i]){
            RescaleMaskandDrawMask(i, OverLapMask[i], Img, mask_xyxy);
        }
    }

    for(int i = NumDetections - 1 ; i > -1 ; --i){
        struct Object* Detect = &ValidDetections[i];
        DrawLabel(Detect->Rect, Detect->conf, Detect->label, Thickness, TextColor, Img);
    }
    
}

static void CreateDirectory(const char *directoryPath){
    struct stat st;
    // Check if directory exists
    if (stat(directoryPath, &st) == -1) {
        // Directory doesn't exist, create it
        if (mkdir(directoryPath, 0777) != 0) {
            printf("Failed to create %s\n", directoryPath);
            return;
        }
        printf("Directory %s created successfully.\n", directoryPath);
    }
}

static void SavePosition(char* Directory, char* baseFileName, const int NumDetections, struct Object *Detections){

    size_t directoryLen = strlen(Directory);
    size_t fileNameLen = strlen(baseFileName);
    if (directoryLen + fileNameLen + 5 > MAX_FILENAME_LENGTH) {
        printf("File path is too long.\n");
        return;
    }
    char cur_directory[MAX_FILENAME_LENGTH];
    strcpy(cur_directory, Directory);

    // Append the file name and extension to filePath
    strcat(cur_directory, baseFileName);
    strcat(cur_directory, ".txt");
    FILE* fileptr;
    fileptr = fopen( cur_directory, "w");

    for (int i = 0; i < NumDetections ; i++) {
        fprintf(fileptr, "%f %f %f %f %f %d\n", 
                Detections[i].Rect.left, Detections[i].Rect.top, Detections[i].Rect.right, Detections[i].Rect.bottom, Detections[i].conf,
                Detections[i].label);
    }

    fclose(fileptr);
    printf("Write Positions into %s.\n", cur_directory);
}

static void SaveResultImage(char* Directory, char* baseFileName, IplImage* Img){
    size_t directoryLen = strlen(Directory);
    size_t fileNameLen = strlen(baseFileName);
    if (directoryLen + fileNameLen + 5 > MAX_FILENAME_LENGTH) {
        printf("File path is too long.\n");
        return;
    }
    char cur_directory[MAX_FILENAME_LENGTH];
    strcpy(cur_directory, Directory);

    strcat(cur_directory, baseFileName);
    strcat(cur_directory, ".jpg");
    cvSaveImage(cur_directory, Img, 0);
    printf("Save predicted image into %s.\n", cur_directory);
}


static void SaveMask(char* Directory, char* baseFileName, uint8_t (* OverLapMask)[TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH], int* HASMASKS, IplImage* Img){

    size_t directoryLen = strlen(Directory);
    size_t fileNameLen = strlen(baseFileName);
    if (directoryLen + fileNameLen + 5 > MAX_FILENAME_LENGTH) {
        printf("File path is too long.\n");
        return;
    }
    char cur_directory[MAX_FILENAME_LENGTH];
    strcpy(cur_directory, Directory);
    strcat(cur_directory, baseFileName);
    // strcat(cur_directory, "/");
    //CreateDirectory(cur_directory);
    size_t length = strlen(cur_directory);
    strcat(cur_directory, ".jpg");

    int mask_xyxy[4] = {0};
    getMaskxyxy(mask_xyxy, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);
    IplImage* OverLapImg = cvCreateImage(cvGetSize(Img), IPL_DEPTH_8U, 1);
    cvZero(OverLapImg);
    for(int i = 0 ; i < NUM_MASKS ; ++i){
        if(HASMASKS[i]){
            char int2char[20];
            // Convert integer to string
            // sprintf(int2char, "%d", i);
            // strcpy(cur_directory + length, int2char);
            // strcpy(cur_directory + length + strlen(int2char), ".jpg");

            IplImage* SrcMask = cvCreateImageHeader(cvSize(TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT), IPL_DEPTH_8U, 1);   
            cvSetData(SrcMask, OverLapMask[i], SrcMask->widthStep);

            // ROI Mask Region by using maskxyxy
            CvRect roiRect = cvRect(mask_xyxy[0], mask_xyxy[1], mask_xyxy[2] - mask_xyxy[0], mask_xyxy[3] - mask_xyxy[1]); // (left, top, width, height)
            cvSetImageROI(SrcMask, roiRect);
            
            // Obtain ROI image
            IplImage* roiImg = cvCreateImage(cvSize(roiRect.width, roiRect.height), SrcMask->depth, 1);
            cvCopy(SrcMask, roiImg, NULL);

            // Obtain Resized Mask
            IplImage* FinalMask = cvCreateImage(cvGetSize(Img), roiImg->depth, 1);
            cvResize(roiImg, FinalMask, CV_INTER_LINEAR);

            //cvSaveImage(cur_directory, FinalMask, 0);
            //printf("Saved Mask at: %s\n", cur_directory);
            cvOr(OverLapImg, FinalMask, OverLapImg, NULL);

            cvReleaseImage(&SrcMask);
            cvReleaseImage(&roiImg);
            cvReleaseImage(&FinalMask);
        }
    }
    // char overlap_directory[MAX_FILENAME_LENGTH];
    // strcpy(overlap_directory, Directory);
    // strcat(overlap_directory, "OverLapMask/");
    // CreateDirectory(overlap_directory);
    // strcat(overlap_directory, baseFileName);
    // strcat(overlap_directory, ".jpg");
    // printf("Save OverLap Image at %s\n", overlap_directory);
    cvSaveImage(cur_directory, OverLapImg, 0);
    cvReleaseImage(&OverLapImg);
}

void extractBaseName(const char *filepath, char *basename) {
    // Find the position of the last '/'
    const char *last_slash_position = strrchr(filepath, '/');
    if (last_slash_position == NULL) {
        // No '/' found, use the beginning of the filepath
        last_slash_position = filepath;
    } else {
        // Move past the '/'
        last_slash_position++;
    }

    const char *dot_position = strrchr(last_slash_position, '.');
    if (dot_position == NULL) {
        dot_position = filepath + strlen(filepath);
    }

    // Calculate the length of the substring between '/' and '.'
    size_t length = dot_position - last_slash_position;

    // Copy the substring between '/' and '.' to basename
    strncpy(basename, last_slash_position, length);
    basename[length] = '\0'; // Null-terminate the string
}


int main(int argc, const char **argv)
{

    FILE *ImageDataFile;
    char NameBuffer[MAX_FILENAME_LENGTH];
    int ImageCount = 0;

    CvScalar TextColor = CV_RGB(255, 255, 255);
    static uint8_t Mask[MAX_DETECTIONS][TRAINED_SIZE_WIDTH * TRAINED_SIZE_HEIGHT] = {0};
    static uint8_t OverLapMask[NUM_CLASSES][TRAINED_SIZE_WIDTH * TRAINED_SIZE_HEIGHT] = {0};

    char* PredictionDirectory = "./Prediction";
    CreateDirectory(PredictionDirectory);

    char* subResultDirectory = "/Results/";
    char ResultDirectory[MAX_FILENAME_LENGTH];   
    strcpy(ResultDirectory, PredictionDirectory);
    strcat(ResultDirectory, subResultDirectory);          // ./Prediction/Masks/
    CreateDirectory(ResultDirectory); 

    char* subMaskDirectory = "/Masks/";
    char MaskDirectory[MAX_FILENAME_LENGTH];

    char* subPositionDirectory = "/Position/";
    char PositionDirectory[MAX_FILENAME_LENGTH];

    if(SAVEMASK){

        strcpy(MaskDirectory, PredictionDirectory);
        strcat(MaskDirectory, subMaskDirectory);          // ./Prediction/Masks/
        CreateDirectory(MaskDirectory);

        strcpy(PositionDirectory, PredictionDirectory);
        strcat(PositionDirectory, subPositionDirectory);  // ./Prediction/Position/
        CreateDirectory(PositionDirectory);
    }
    
    // Open ImgData.txt that stores Image Directories
    ImageDataFile = fopen(argv[1], "r");
    if (ImageDataFile == NULL) {
        printf("Error opening file %s\n", argv[1]);
        return 0;
    }

    // Read the string from a .txt File
    while (fgets(NameBuffer, sizeof(NameBuffer), ImageDataFile) != NULL) {
        NameBuffer[strcspn(NameBuffer, "\n")] = '\0';
        IplImage* Img = cvLoadImage(NameBuffer, CV_LOAD_IMAGE_COLOR);
        if(!Img){
            printf("%s not found\n", NameBuffer);
            continue;
        }

        if(SAVEMASK){
            memset( OverLapMask, 0, sizeof(uint8_t) * NUM_CLASSES * TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH);
        }
        

        printf("===============Reading Image: %s ===============\n", NameBuffer);
        float Mask_Input[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH];
        float Mask_Coeffs[MAX_DETECTIONS][NUM_MASKS];

        struct Object ValidDetections[MAX_DETECTIONS]; 
        int NumDetections = 0;

        int HASMASKS[NUM_CLASSES] = {0};
        // Preprocessing + NMS 
        PreProcessing(&Mask_Input[0][0], &NumDetections, ValidDetections, Mask_Coeffs, argv, ImageCount);

        // Store Masks Results
        if(SAVEMASK){
            PostProcessingSaveMask(NumDetections, ValidDetections, Mask_Input, Img, Mask, TextColor, OverLapMask, HASMASKS);
        }else{
            PostProcessing(NumDetections, ValidDetections, Mask_Input, Img, Mask, TextColor);
        }
        printf("============Drawing Mask and Labels Complete============\n");
        
       // Saving Data
        char BaseName[MAX_FILENAME_LENGTH];
        extractBaseName(NameBuffer, BaseName);
        if(SAVEMASK){
            SaveMask( MaskDirectory, BaseName, OverLapMask, HASMASKS, Img);
            SavePosition( PositionDirectory, BaseName, NumDetections, ValidDetections);
        }
        SaveResultImage( ResultDirectory, BaseName, Img);

        cvReleaseImage(&Img);
        printf("===============Saved Image Complete===============\n");
        printf("\n\n");
        ++ImageCount;
    }
    printf("All Images are read\n");
    // Close the file
    fclose(ImageDataFile);

    return 0;
}
/* 
================================================================================================================================================================
Type:
gcc main.c -o T ./Sources/Input.c ./Sources/Bbox.c  `pkg-config --cflags --libs opencv` -lm
./T ./ImgData.txt ./outputs/cls_preds8.txt ./outputs/cls_preds16.txt ./outputs/cls_preds32.txt ./outputs/reg_preds8.txt ./outputs/reg_preds16.txt ./outputs/reg_preds32.txt ./outputs/seg_preds8.txt ./outputs/seg_preds16.txt ./outputs/seg_preds32.txt ./outputs/mask_input.txt
================================================================================================================================================================
*/