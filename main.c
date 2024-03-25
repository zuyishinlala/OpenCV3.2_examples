#include <stdio.h>
#include <string.h>

#include "./Sources/Object.h"
#include "./Sources/Input.h"
#include "./Sources/Output.h"
#include "./Sources/Bbox.h"
#include "./Sources/Parameters.h"


#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <time.h>

#include "Post_NMS.c"

// dist2bbox & generate_anchor in YOLOv6
static void post_regpreds(float (*distance)[4], char *type)
{
    float stride = 8.f;
    int row_bound = HEIGHT0, col_bound = WIDTH0;

    for (int stride_index = 0 ; stride_index < 3 ; ++stride_index, row_bound /= 2, col_bound /= 2, stride *= 2){

        int end_row_index = row_bound * col_bound;

        #pragma omp parallel for collapse(2) schedule(static, CHUNKSIZE)
        for (int anchor_points_y = 0 ; anchor_points_y < row_bound ; ++anchor_points_y)
        {
            for (int anchor_points_x = 0 ; anchor_points_x < col_bound ; ++anchor_points_x)
            {
                int row = anchor_points_y * col_bound + anchor_points_x;
                float *data = &distance[row][0]; // left, top, right, bottom

                // x1y1 = anchor_points - lt
                data[0] = anchor_points_x - data[0];
                data[1] = anchor_points_y - data[1];

                // x2y2 = anchor_points + rb
                data[2] += anchor_points_x; // anchor_points_c + data[2]
                data[3] += anchor_points_y; // anchor_points_r + data[3]

                for (int j = 0 ; j < 4 ; ++j){
                    distance[row][j] = (distance[row][j] + 0.5f) * stride;
                }
            }
        }
        distance += end_row_index;
    }
        /*
        if (!strcmp(type, "xywh"))
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
        */
    //}
}

// ===================================
// Find Max class probability for each row
// ===================================
static void max_classpred(float (*cls_pred)[NUM_CLASSES], float *max_predictions, int *class_index)
{
    // Obtain max_prob and the max class index
    #pragma omp parallel for schedule(static, CHUNKSIZE)
    for (int i = 0; i < ROWSIZE ; ++i)
    {
        float *predictions = &cls_pred[i][0];
        float max_pred = 0;
        int max_class_index = -1;

        for (int class_idx = 0; class_idx < NUM_CLASSES; ++class_idx)
        {
            if (max_pred < predictions[class_idx])
            {
                max_pred = predictions[class_idx];
                max_class_index = class_idx;
            }
        }

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

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_inplace(Objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_inplace(Objects, i, right);
        }
    }
}

int compare_objects(const void *a, const void *b) {
    const struct Object *obj1 = (const struct Object *)a;
    const struct Object *obj2 = (const struct Object *)b;

    // Sort in descending order based on 'conf'
    if (obj1->conf < obj2->conf) return 1;
    if (obj1->conf > obj2->conf) return -1;
    return 0;
}

// ========================================
// Calculate intersection area -- xyxy(left, top, bottom, right)
// ========================================
static float intersection_area(struct Bbox box1, struct Bbox box2)
{
    float width = fminf(box1.right, box2.right) - fmaxf(box1.left, box2.left);
    float height = fminf(box1.bottom, box2.bottom) - fmaxf(box1.top, box2.top);
    return (width < 0 || height < 0) ? 0 : width * height;
}
// ========================================
// Perform NMS
// ========================================
static void nms_sorted_bboxes(const struct Object *faceobjects, int size, struct Object *picked_object, int *CountValidDetect)
{
    if (size == 0)
        return;
    // Calculated areas
    float areas[ROWSIZE];

    #pragma omp parallel for schedule(static)
    for (int row_index = 0; row_index < size; row_index++)
    {
        areas[row_index] = BoxArea(&faceobjects[row_index].Rect);
    }

    // ==============================
    // Fast-NMS
    // ==============================
    float maxIOU[ROWSIZE] = {0.f}; // record max value

    #pragma omp parallel for schedule(dynamic)
    for (int cur_index = 0 ; cur_index < size ; ++cur_index) {
        float max_IOU = 0.f;
        for (int c = 0 ; c < cur_index ; ++c) {
            // Calculate IOU
            float inter_area = intersection_area(faceobjects[cur_index].Rect, faceobjects[c].Rect);
            float union_area = areas[cur_index] + areas[c] - inter_area;
            float iou = inter_area / union_area;
            if(iou > max_IOU){
                max_IOU = iou;
            }
        }
        maxIOU[cur_index] = max_IOU;
    }
    // Pick good instances
    for (int row_index = 0; row_index < size && *CountValidDetect < MAX_DETECTIONS; row_index++){
        if (maxIOU[row_index] < NMS_THRESHOLD) // keep Object i
            picked_object[(*CountValidDetect)++] = faceobjects[row_index];
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
static int nextPowerOf2(int n)
{
    int exponent = 0;
    while ((1 << exponent) <= n)
    {
        exponent++;
    }
    return exponent;
}

// Perform Sorting + NMS
static void non_max_suppression_seg(struct Pred_Input *input, char *classes, struct Object *picked_objects, int *CountValidDetect, const float conf_threshold)
{
    // Calculate max class and prob for each row
    float max_clsprob[ROWSIZE] = {0};
    int max_class_index[ROWSIZE] = {0};
    max_classpred(input->cls_pred, max_clsprob, max_class_index);

    // Count good Bboxes
    int CountValidCandid = 0;
    struct Object candidates[ROWSIZE * NUM_CLASSES];
    
    int max_wh = 4096;        // maximum box width and height
    int max_nms = 30000;      // maximum number of boxes put into torchvision.ops.nms()
    float time_limit = 10.0f; // quit the function when nms cost time exceed the limit time.
    // multi_label &= NUM_CLASSES > 1;   // multiple labels per box


    int shift_num = nextPowerOf2(ROWSIZE);
    int bitwise_num = (1 << shift_num) - 1;

    if (AGNOSTIC){
        #pragma omp parallel for schedule(static, CHUNKSIZE)
        for (int row_index = 0; row_index < ROWSIZE; ++row_index)
        {
            if (max_clsprob[row_index] > conf_threshold)
            {
                struct Bbox box = {input->reg_pred[row_index][0], input->reg_pred[row_index][1], input->reg_pred[row_index][2], input->reg_pred[row_index][3]};
                // init an Object
                struct Object obj = {box, max_class_index[row_index], max_clsprob[row_index], &(input->seg_pred[row_index][0])};
                
                #pragma omp critical
                {
                    candidates[CountValidCandid++] = obj;
                }
            }
        }
    }
    else
    {
        if (MULTI_LABEL)
        {
            #pragma omp parallel for collapse(2)
            for (int row_index = 0; row_index < ROWSIZE; ++row_index)
            {
                for (int class = 0; class < NUM_CLASSES; ++class)
                {
                    if (input->cls_pred[row_index][class] > conf_threshold)
                    {
                        float enlarge_factor = class * max_wh;
                        struct Bbox box = {input->reg_pred[row_index][0] + enlarge_factor,
                                           input->reg_pred[row_index][1] + enlarge_factor,
                                           input->reg_pred[row_index][2] + enlarge_factor,
                                           input->reg_pred[row_index][3] + enlarge_factor};

                        int new_row_index = row_index | (class << shift_num);
                        // Temporary obj
                        struct Object obj = {box, new_row_index, max_clsprob[row_index], &(input->seg_pred[row_index][0])};
                        #pragma omp critical
                        {
                            candidates[CountValidCandid++] = obj;
                        }
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for
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
                    #pragma omp critical
                    {
                        candidates[CountValidCandid++] = obj;
                    }
                }
            }
        }
    }

    printf("Found %d candidates...\n", CountValidCandid);

    if (classes != NULL)
    { // to-do: only sort labels of these classes ( >= 1)
      // GetOnlyClass(classes, CountValidCandid, candidates);
    }

    // Sort with confidence
    qsort_inplace(candidates, 0, CountValidCandid - 1);
    //qsort( candidates, CountValidCandid, sizeof(candidates[0]), compare_objects);
    if (CountValidCandid > max_nms)
        CountValidCandid = max_nms;
    nms_sorted_bboxes(candidates, CountValidCandid, picked_objects, CountValidDetect);

    if (!AGNOSTIC){
        for (int i = 0; i < *CountValidDetect ; ++i)
        {
            int row_index = picked_objects[i].label;
            int real_class = max_class_index[row_index];
            if (MULTI_LABEL)
            {
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

static void CopyMaskCoeffs(float (*DstCoeffs)[NUM_MASKS], const int NumDetections, struct Object *ValidDetections)
{   
    //#pragma omp parallel for
    for (int i = 0; i < NumDetections; ++i)
    {
        memcpy(DstCoeffs[i], ValidDetections[i].maskcoeff, sizeof(float) * NUM_MASKS);
        ValidDetections[i].maskcoeff = &DstCoeffs[i][0];
    }
}

static inline void PrintObjectData(int NumDetections, struct Object *ValidDetections)
{
    for (int i = 0; i < NumDetections; ++i)
    {
        printf("======Index: %d======\n", i);
        printf("Box Position: %f, %f, %f, %f\n", ValidDetections[i].Rect.left, ValidDetections[i].Rect.top, ValidDetections[i].Rect.right, ValidDetections[i].Rect.bottom);
        printf("Confidence: %f \n", ValidDetections[i].conf);
        printf("Label: %d \n", ValidDetections[i].label);
        printf("Mask Coeffs: %f, %f, %f, %f \n", ValidDetections[i].maskcoeff[0], ValidDetections[i].maskcoeff[1], ValidDetections[i].maskcoeff[2], ValidDetections[i].maskcoeff[3]);
    }
    printf("======================\n");
}

// Read Inputs + Pre-process of Inputs + NMS
static inline void PreProcessing(float *Mask_Input, int *NumDetections, struct Object *ValidDetections, float (*MaskCoeffs)[NUM_MASKS], const char **argv, int ImageIndex)
{
    char *Bboxtype = "xyxy";
    char *classes = NULL;
    struct Pred_Input input;

    clock_t start = clock();
    // ========================
    // Init Inputs(9 prediction input + 1 mask input) in Sources/Input.c
    // ========================
    initPredInput(&input, Mask_Input, argv, ImageIndex);
    clock_t init_End = clock();

    sigmoid(ROWSIZE, NUM_CLASSES, &input.cls_pred[0][0]);
    post_regpreds(input.reg_pred, Bboxtype);
    clock_t PreProcess = clock();

    non_max_suppression_seg(&input, classes, ValidDetections, NumDetections, CONF_THRESHOLD);
    clock_t NMS = clock();

    printf("NMS Done,Got %d Detections...\n", *NumDetections);

    CopyMaskCoeffs(MaskCoeffs, *NumDetections, ValidDetections);
    clock_t Copied = clock();

    printf("==============Time Spend==============\n");
    printf("-- Init Input time %.6f\n", (double)(init_End - start) / CLOCKS_PER_SEC * 1000);
    printf("-- PreProcess time %.6f\n", (double)(PreProcess - init_End) / CLOCKS_PER_SEC * 1000);
    printf("-- NMS %.6f\n", (double)(NMS - PreProcess) / CLOCKS_PER_SEC * 1000);
    printf("-- Copied time %.6f\n", (double)(Copied - NMS) / CLOCKS_PER_SEC * 1000);
    // PrintObjectData(*NumDetections, ValidDetections);
}

// Post NMS(Rescale Mask, Draw Label) in Post_NMS.c
static inline void PostProcessing(struct Output* output, const float (* Mask_Input)[NUM_MASKS], IplImage *Img, uint8_t (*Mask)[TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH], CvScalar TextColor)
{
    //printf("Drawing Labels and Segments...\n");
    const int NumDetections = output->NumDetections;
    struct Object* ValidDetections = output->detections;

    clock_t start = clock();
    int mask_xyxy[4] = {0};             // the real mask in the resized image. left top bottom right
    getMaskxyxy(mask_xyxy,  TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);

    omp_set_nested(1);

    #pragma omp parallel for 
    for(int i = 0 ; i < NumDetections ; ++i){
        memset(Mask[i], 0, sizeof(uint8_t) * TRAINED_SIZE_WIDTH * TRAINED_SIZE_HEIGHT); // 2 / Detection
        struct Object* Detect = &ValidDetections[i];
        handle_proto_test( Detect, Mask_Input, Mask[i]); // 9 / Detection
        rescalebox(&Detect->Rect, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);
    }
    omp_set_nested(0);
    clock_t handle_proto_test_time = clock();

    int Thickness = (int)fmaxf(roundf((Img->width + Img->height) / 2.f * 0.003f), 2);
    
    #pragma omp parallel for 
    for (int i = NumDetections - 1; i > -1; --i){
        //printf("Thread Num:%d, Index: %d\n", omp_get_thread_num(), i);
        struct Object *Detect = &ValidDetections[i];
        RescaleMask( &output->Masks[i], Mask[i], Img, mask_xyxy);
        //DrawMask( Detect->label, MASK_TRANSPARENCY, output->Masks[i], Img);
        //DrawLabel( Detect->Rect, Detect->conf, Detect->label, Thickness, TextColor, Img);
    }
    clock_t draw_masklabel_time = clock();

    printf("-- handle_proto_test time per detection %.6f\n", (double)(handle_proto_test_time - start) / CLOCKS_PER_SEC * 1000 / (double)NumDetections);
    printf("-- Total Handle_proto_test:%.6f\n", (double)(handle_proto_test_time - start) / CLOCKS_PER_SEC * 1000);
    printf("-- Draw Masks & Labels time per detection %.6f\n", (double)(draw_masklabel_time - handle_proto_test_time) / CLOCKS_PER_SEC * 1000 / (double)NumDetections);
    printf("-- Total Draw Masks & Labels time  %.6f\n", (double)(draw_masklabel_time - handle_proto_test_time) / CLOCKS_PER_SEC * 1000);
    printf("==============Time Spend End==============\n");
}
/*
// Post NMS(Rescale Mask, Draw Label) in Post_NMS.c
static inline void PostProcessingDrawByMask(const int NumDetections, struct Object *ValidDetections, const float (*Mask_Input)[NUM_MASKS],
                                          IplImage *Img, uint8_t (*Mask)[TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH], CvScalar TextColor,
                                          uint8_t (*OverLapMask)[TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH], int *HASMASKS)
{
    printf("Drawing Labels and Segments...\n");

    int mask_xyxy[4] = {0}; // the real mask in the resized image. left top bottom right
    getMaskxyxy(mask_xyxy, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);

    for (int i = 0; i < NumDetections; ++i)
    {
        memset(Mask[i], 0, sizeof(uint8_t) * TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH); 
        struct Object *Detect = &ValidDetections[i];
        const int Class_Index = Detect->label;
        handle_proto_test(Detect, Mask_Input, Mask[i]);
        rescalebox(&Detect->Rect, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);
    }

    // OverLap Masks
    for (int i = 0; i < NumDetections; ++i)
    {
        int Label = ValidDetections[i].label;
        for (int r = 0; r < TRAINED_SIZE_HEIGHT; ++r)
        {
            for (int c = 0; c < TRAINED_SIZE_WIDTH; ++c)
            {
                OverLapMask[Label][r * TRAINED_SIZE_WIDTH + c] |= Mask[i][r * TRAINED_SIZE_WIDTH + c];
            }
        }
        HASMASKS[Label] = 1;
    }

    int Thickness = (int)fmaxf(roundf((Img->width + Img->height) / 2.f * 0.003f), 2);

    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        if (HASMASKS[i])
        {
            RescaleMask(i, OverLapMask[i], Img, mask_xyxy);
        }
    }

    for (int i = NumDetections - 1; i > -1; --i)
    {
        struct Object *Detect = &ValidDetections[i];
        DrawLabel(Detect->Rect, Detect->conf, Detect->label, Thickness, TextColor, Img);
    }
}
*/

static void CreateDirectory(const char *directoryPath)
{
    struct stat st;
    // Check if directory exists
    if (stat(directoryPath, &st) == -1)
    {
        // Directory doesn't exist, create it
        if (mkdir(directoryPath, 0777) != 0)
        {
            printf("Failed to create %s\n", directoryPath);
            return;
        }
        printf("Directory %s created successfully.\n", directoryPath);
    }
}

static void SavePosition(char *Directory, char *baseFileName, const int NumDetections, struct Object *Detections)
{

    size_t directoryLen = strlen(Directory);
    size_t fileNameLen = strlen(baseFileName);
    if (directoryLen + fileNameLen + 5 > MAX_FILENAME_LENGTH)
    {
        printf("File path is too long.\n");
        return;
    }
    char cur_directory[MAX_FILENAME_LENGTH];
    strcpy(cur_directory, Directory);

    // Append the file name and extension to filePath
    strcat(cur_directory, baseFileName);
    strcat(cur_directory, ".txt");
    FILE *fileptr;
    fileptr = fopen(cur_directory, "w");

    for (int i = 0; i < NumDetections; i++)
    {
        fprintf(fileptr, "%f %f %f %f %f %d\n",
                Detections[i].Rect.left, Detections[i].Rect.top, Detections[i].Rect.right, Detections[i].Rect.bottom, Detections[i].conf,
                Detections[i].label);
    }

    fclose(fileptr);
    printf("Write Positions into %s.\n", cur_directory);
}

static void SaveResultImage(char *Directory, char *baseFileName, IplImage *Img)
{
    size_t directoryLen = strlen(Directory);
    size_t fileNameLen = strlen(baseFileName);
    if (directoryLen + fileNameLen + 5 > MAX_FILENAME_LENGTH)
    {
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

static void SaveMask(char *Directory, char *baseFileName, const struct Output *output, IplImage *Img)
{

    size_t directoryLen = strlen(Directory);
    size_t fileNameLen = strlen(baseFileName);
    if (directoryLen + fileNameLen + 5 > MAX_FILENAME_LENGTH)
    {
        printf("File path is too long.\n");
        return;
    }
    char cur_directory[MAX_FILENAME_LENGTH];
    strcpy(cur_directory, Directory);
    strcat(cur_directory, baseFileName);
    
    size_t length = strlen(cur_directory);
    strcat(cur_directory, ".jpg");

    const int NumDetections = output->NumDetections;

    IplImage *OverLapImg = cvCreateImage(cvGetSize(Img), IPL_DEPTH_8U, 1);
    cvZero(OverLapImg);
    for (int i = 0; i < NumDetections ; ++i){
        cvOr(OverLapImg, output->Masks[i], OverLapImg, NULL);
    }
    cvSaveImage(cur_directory, OverLapImg, 0);
    cvReleaseImage(&OverLapImg);
}

static void extractBaseName(const char *filepath, char *basename)
{
    // Find the position of the last '/'
    const char *last_slash_position = strrchr(filepath, '/');
    if (last_slash_position == NULL)
    {
        // No '/' found, use the beginning of the filepath
        last_slash_position = filepath;
    }
    else
    {
        // Move past the '/'
        last_slash_position++;
    }

    const char *dot_position = strrchr(last_slash_position, '.');
    if (dot_position == NULL)
    {
        dot_position = filepath + strlen(filepath);
    }

    // Calculate the length of the substring between '/' and '.'
    size_t length = dot_position - last_slash_position;

    // Copy the substring between '/' and '.' to basename
    strncpy(basename, last_slash_position, length);
    basename[length] = '\0'; // Null-terminate the string
}

static void AppendandCreateDirectory(char *directory, char *foldername, char *ResultDirectory)
{
    strcpy(ResultDirectory, directory);
    strcat(ResultDirectory, foldername);
    CreateDirectory(ResultDirectory);
}

int main(int argc, const char **argv)
{
    FILE *ImageDataFile;
    char NameBuffer[MAX_FILENAME_LENGTH];
    int ImageCount = 0;
    CvScalar TextColor = CV_RGB(255, 255, 255);
    static uint8_t Mask[MAX_DETECTIONS][TRAINED_SIZE_WIDTH * TRAINED_SIZE_HEIGHT] = {0};

    char *PredictionDirectory = "./Prediction";
    CreateDirectory(PredictionDirectory);

    char *subResultDirectory = "/Results/";
    char ResultDirectory[MAX_FILENAME_LENGTH];
    AppendandCreateDirectory(PredictionDirectory, subResultDirectory, ResultDirectory);

    char *subMaskDirectory = "/Masks/";
    char MaskDirectory[MAX_FILENAME_LENGTH];

    char *subPositionDirectory = "/Position/";
    char PositionDirectory[MAX_FILENAME_LENGTH];

    if (SAVEMASK)
    {
        AppendandCreateDirectory(PredictionDirectory, subMaskDirectory, MaskDirectory);
        AppendandCreateDirectory(PredictionDirectory, subPositionDirectory, PositionDirectory);
    }

    // Open ImgData.txt that stores Image Directories
    ImageDataFile = fopen(argv[1], "r");
    if (ImageDataFile == NULL)
    {
        printf("Error opening file %s\n", argv[1]);
        return 0;
    }

    // Read the string from a .txt File
    while (fgets(NameBuffer, sizeof(NameBuffer), ImageDataFile) != NULL && ImageCount < READIMAGE_LIMIT){\
        struct Output output;
        NameBuffer[strcspn(NameBuffer, "\n")] = '\0';
        IplImage *Img = cvLoadImage(NameBuffer, CV_LOAD_IMAGE_COLOR);
        if (!Img){
            printf("%s not found\n", NameBuffer);
            continue;
        }
        init_Output( &output, Img->width, Img->depth);

        printf("===============Reading Image: %s ===============\n", NameBuffer);
        float Mask_Coeffs[MAX_DETECTIONS][NUM_MASKS];
        float Mask_Input[MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH][NUM_MASKS];
        struct Object ValidDetections[MAX_DETECTIONS];
        int NumDetections = 0;

        // Read Input + Process Input + NMS
        PreProcessing(&Mask_Input[0][0], &NumDetections, ValidDetections, Mask_Coeffs, argv, ImageCount);

        output.NumDetections = NumDetections;
        memcpy(output.detections, ValidDetections, sizeof(struct Object) * NumDetections);
        // Store Masks Results
        PostProcessing( &output, Mask_Input, Img, Mask, TextColor);

        printf("============Drawing Mask and Labels Complete============\n");

        // Saving Data
        char BaseName[MAX_FILENAME_LENGTH];
        extractBaseName(NameBuffer, BaseName);
        
        if (SAVEMASK){
            SaveMask(MaskDirectory, BaseName, &output, Img);
            SavePosition(PositionDirectory, BaseName, output.NumDetections, output.detections);
            printf("===============Saved Masks Complete===============\n");
        }

        SaveResultImage(ResultDirectory, BaseName, Img);
        releaseAllMasks(&output);
        cvReleaseImage(&Img);
        printf("===============Saved Image Complete===============\n");
        printf("\n\n");
        ++ImageCount;
    }
    printf("All Images are read\n");
    fclose(ImageDataFile);
    return 0;
}
/*
================================================================================================================================================================
Type:
gcc main.c -o T ./Sources/Input.c ./Sources/Bbox.c  `pkg-config --cflags --libs opencv` -lm
time ./T ./ImgData.txt ./outputs/cls_preds8.txt ./outputs/cls_preds16.txt ./outputs/cls_preds32.txt ./outputs/reg_preds8.txt ./outputs/reg_preds16.txt ./outputs/reg_preds32.txt ./outputs/seg_preds8.txt ./outputs/seg_preds16.txt ./outputs/seg_preds32.txt ./outputs/mask_input.txt
================================================================================================================================================================
*/

/*

X
OpenMP
Mask (32 * (MASK_WIDTH*MASK_HEIGHT))
real    0m0.477s
user    0m0.463s
sys     0m0.064s

5 Images
real    0m4.675s
user    0m4.763s
sys     0m0.109s

OpenMP 
- sigmoid, mask matrix multiplication
real    0m0.511s
user    0m0.662s
sys     0m0.052s

- sigmoid, mask matrix multiplication, Calculate IOU
real    0m0.514s
user    0m0.651s
sys     0m0.061s

- sigmoid, mask matrix multiplication, Calculate IOU, max_classpred
real    0m0.528s
user    0m0.697s
sys     0m0.040s

- sigmoid, mask matrix multiplication, max_classpred, Generate_Anchor
real    0m0.514s
user    0m0.689s
sys     0m0.024s

- sigmoid, mask matrix multiplication, max_classpred, Generate_Anchor, Calculate IOU
real    0m0.482s
user    0m0.611s
sys     0m0.061s


- sigmoid, mask matrix multiplication, max_classpred, Generate_Anchor, Calculate IOU, Calculate Masks output
real    0m0.437s
user    0m0.618s
sys     0m0.061s

real    0m0.481s
user    0m0.644s
sys     0m0.036s
-- Init Input time 301.988000
-- PreProcess time 3.056000
-- NMS 1.131000
-- Copied time 0.006000
-- handle_proto_test time per detection 7.542636
-- Draw Masks & Labels time per detection 14.104909
*/