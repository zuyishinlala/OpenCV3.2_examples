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
    // float stride = 8.f;
    int row_bound = HEIGHT0, col_bound = WIDTH0;
    const int row_start[3] = {0, HEIGHT0 * WIDTH0, HEIGHT0 * WIDTH0 + HEIGHT1 * WIDTH1};
    const int row_bound_arr[3] = {HEIGHT0, HEIGHT1, HEIGHT2};
    const int col_bound_arr[3] = { WIDTH0,  WIDTH1,  WIDTH2};
    const float strides[3] = {8.f, 16.f, 32.f};

    #pragma omp parallel for
    for (int stride_index = 0 ; stride_index < 3 ; ++stride_index){

        //int end_row_index = row_bound * col_bound;
        int row_bound = row_bound_arr[stride_index];
        int col_bound = col_bound_arr[stride_index];
        float stride = strides[stride_index];
    
        #pragma omp parallel for collapse(2) schedule(static)
        for (int anchor_points_y = 0 ; anchor_points_y < row_bound ; ++anchor_points_y)
        {   
            //printf("Thread: %d Stride Index: %d\n", omp_get_num_threads(),  stride_index);
            for (int anchor_points_x = 0 ; anchor_points_x < col_bound ; ++anchor_points_x)
            {
                int row = anchor_points_y * col_bound + anchor_points_x + row_start[stride_index];
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
        //distance += end_row_index;
    }
    int min_block_size = WIDTH2 * HEIGHT2;

    // #pragma omp parallel for schedule(static, CHUNKSIZE)
    // for(int i = 0 ; i < ROWSIZE ; ++i){
    //     int block_index = i / min_block_size;
    //     int stride_index = (block_index < 16) ? 0 : (block_index < 19) ? 1 : 2;
    //     int block_size_w = col_bound_arr[stride_index];
    //     int relative_block_index = i - row_start[stride_index];
    //     float stride = strides[stride_index];

    //     int anchor_points_x = relative_block_index % block_size_w;
    //     int anchor_points_y = relative_block_index / block_size_w;

    //     float *data = &distance[i][0]; // left, top, right, bottom
    //     // x1y1 = anchor_points - lt
    //     data[0] = anchor_points_x - data[0];
    //     data[1] = anchor_points_y - data[1];

    //     // x2y2 = anchor_points + rb
    //     data[2] += anchor_points_x ; // anchor_points_c + data[2]
    //     data[3] += anchor_points_y ; // anchor_points_r + data[3]

    //     for (int j = 0 ; j < 4 ; ++j){
    //         distance[i][j] = (distance[i][j] + 0.5f) * stride;
    //     }
    // }
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
    #pragma omp parallel for schedule(static)
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
    for (int i = 0; i < size && *CountValidDetect < MAX_DETECTIONS; i++){
        if (maxIOU[i] < NMS_THRESHOLD) // keep Object i
            picked_object[(*CountValidDetect)++] = faceobjects[i];
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
        #pragma omp parallel for schedule(static)
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
            #pragma omp parallel for collapse(2) schedule(static)
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
            #pragma omp parallel for schedule(static)
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
    if (CountValidCandid > max_nms) CountValidCandid = max_nms;
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
static double PreProcessing(float *Mask_Input, int *NumDetections, struct Object *ValidDetections, float (*MaskCoeffs)[NUM_MASKS], const char **argv, int ImageIndex)
{
    char *Bboxtype = "xyxy";
    char *classes = NULL;
    struct Pred_Input input;

    double start = omp_get_wtime( );
    // ========================
    // Init Inputs(9 prediction input + 1 mask input) in Sources/Input.c
    // ========================
    initPredInput(&input, Mask_Input, argv, ImageIndex);
    double init_End = omp_get_wtime( );
    sigmoid(ROWSIZE, NUM_CLASSES, &input.cls_pred[0][0]);
    post_regpreds(input.reg_pred, Bboxtype);
    double PreProcess =  omp_get_wtime( );

    non_max_suppression_seg(&input, classes, ValidDetections, NumDetections, CONF_THRESHOLD);
    double NMS = omp_get_wtime( );

    printf("NMS Done,Got %d Detections...\n", *NumDetections);

    CopyMaskCoeffs(MaskCoeffs, *NumDetections, ValidDetections);
    double Copied =  omp_get_wtime( );

    // printf("==============Time Spend==============\n");
    // printf("-- Init Input time %.6f ms\n", (init_End - start) * 1000);
    printf("-- PreProcess time %.6f ms\n", (PreProcess - init_End)  * 1000);
    printf("-- NMS %.6f ms\n", (NMS - PreProcess)  * 1000);
    printf("-- Copied mask coeffs time %.6f ms\n", (Copied - NMS) * 1000);
    // PrintObjectData(*NumDetections, ValidDetections);
    return Copied - init_End;
}

// Post NMS(Rescale Mask, Draw Label) in Post_NMS.c
static double PostProcessing(struct Output* output, const float (* Mask_Input)[NUM_MASKS], IplImage *Img, uint8_t (*Mask)[TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH], CvScalar TextColor)
{
    //printf("Drawing Labels and Segments...\n");
    const int NumDetections = output->NumDetections;
    struct Object* ValidDetections = output->detections;

    double start = omp_get_wtime();
    int mask_xyxy[4] = {0};             // the real mask in the resized image. left top bottom right
    getMaskxyxy(mask_xyxy,  TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);

    //#pragma omp parallel for schedule(static)
    for(int i = 0 ; i < NumDetections ; ++i){
        memset(Mask[i], 0, sizeof(uint8_t) * TRAINED_SIZE_WIDTH * TRAINED_SIZE_HEIGHT); // 2 / Detection
        struct Object* Detect = &ValidDetections[i];
        handle_proto_test( Detect, Mask_Input, Mask[i], MASK_THRESHOLD); // 9 / Detection
        rescalebox(&Detect->Rect, TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT, Img->width, Img->height);
    }

    double handle_proto_test_time = omp_get_wtime();

    int Thickness = (int)fmaxf(roundf((Img->width + Img->height) / 2.f * 0.003f), 2);

    #pragma omp parallel for ordered
    for (int i = NumDetections - 1; i > -1; --i){
        //printf("Thread Num:%d, Index: %d\n", omp_get_thread_num(), i);
        struct Object *Detect = &ValidDetections[i];
        RescaleMask( &output->Masks[i], Mask[i], Img, mask_xyxy);
        DrawMask( Detect->label, MASK_TRANSPARENCY, output->Masks[i], Img);
        DrawLabel( Detect->Rect, Detect->conf, Detect->label, Thickness, TextColor, Img);
    }
    double draw_masklabel_time = omp_get_wtime();

    printf("-- Handle_proto_test Avg: %.6f ms, Total:%.6f ms\n",(handle_proto_test_time - start)/(double)NumDetections * 1000, (handle_proto_test_time - start)  * 1000);
    printf("-- Rescale Mask Time: Avg:%.6f ms, Total: %.6f ms\n", (draw_masklabel_time - handle_proto_test_time) /(double)NumDetections  * 1000, (draw_masklabel_time - handle_proto_test_time) * 1000);
    return draw_masklabel_time - start;
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

    IplImage *OverLapImg = cvCreateImage(cvGetSize(Img), Img->depth, 1);
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
    //static uint8_t OverLapMask[NUM_CLASSES][TRAINED_SIZE_WIDTH * TRAINED_SIZE_HEIGHT] = {0};

    char *PredictionDirectory = "./Prediction";
    CreateDirectory(PredictionDirectory);

    char *subResultDirectory = "/Results/";
    char ResultDirectory[MAX_FILENAME_LENGTH];
    AppendandCreateDirectory(PredictionDirectory, subResultDirectory, ResultDirectory);

    char *subMaskDirectory = "/Masks/";
    char MaskDirectory[MAX_FILENAME_LENGTH];

    char *subPositionDirectory = "/Position/";
    char PositionDirectory[MAX_FILENAME_LENGTH];

    char *subMaskperClassDirectory = "/MaskperClass/";
    char MaskperClassDirectory[MAX_FILENAME_LENGTH];


    // Open ImgData.txt that stores Image Directories
    ImageDataFile = fopen(argv[1], "r");
    if (ImageDataFile == NULL)
    {
        printf("Error opening file %s\n", argv[1]);
        return 0;
    }

    // Read the string from a .txt File
    while (fgets(NameBuffer, sizeof(NameBuffer), ImageDataFile) != NULL && ImageCount < READIMAGE_LIMIT){
        struct Output output;
        NameBuffer[strcspn(NameBuffer, "\n")] = '\0';
        IplImage *Img = cvLoadImage(NameBuffer, CV_LOAD_IMAGE_COLOR);
        if (!Img){
            printf("%s not found\n", NameBuffer);
            continue;
        }
        init_Output( &output, Img->width, Img->depth);
        printf("*********************************************\n");
        printf("**************Reading Image: %s**************\n", NameBuffer);
        float Mask_Coeffs[MAX_DETECTIONS][NUM_MASKS];
        float Mask_Input[MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH][NUM_MASKS];
        struct Object ValidDetections[MAX_DETECTIONS];
        int NumDetections = 0;

        // Read Input + Process Input + NMS
        double preprocess_without_init_time = PreProcessing(&Mask_Input[0][0], &NumDetections, ValidDetections, Mask_Coeffs, argv, ImageCount);

        output.NumDetections = NumDetections;
        memcpy(output.detections, ValidDetections, sizeof(struct Object) * NumDetections);
        // Store Masks Results
        double post_process_time = PostProcessing( &output, Mask_Input, Img, Mask, TextColor);

        printf("================================================\n\n");
        printf("Total Compute time without init_input & drawing: %.6fms!!!!!!!\n", (preprocess_without_init_time + post_process_time) * 1000);
        printf("\n================================================\n");

        printf("=========Drawing Mask and Labels Complete=========\n");

        // Saving Data
        char BaseName[MAX_FILENAME_LENGTH];
        extractBaseName(NameBuffer, BaseName);
        
        if (SAVEMASK){
            AppendandCreateDirectory(PredictionDirectory, subMaskDirectory, MaskDirectory);
            AppendandCreateDirectory(PredictionDirectory, subPositionDirectory, PositionDirectory);
            SaveMask(MaskDirectory, BaseName, &output, Img);
            SavePosition(PositionDirectory, BaseName, output.NumDetections, output.detections);
            printf("===============Saved Masks Complete===============\n");
        }
        /*
        if(SAVEPERMASK){
            AppendandCreateDirectory(PredictionDirectory, subMaskperClassDirectory, MaskperClassDirectory);
            SaveperMask(MaskperClassDirectory, OverLapMask, &output);
        }
        */
        SaveResultImage(ResultDirectory, BaseName, Img);
        releaseAllMasks(&output);
        cvReleaseImage(&Img);
        printf("**************Saved Result Complete**************\n");
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
gcc main.c -o T ./Sources/Input.c ./Sources/Bbox.c ./Sources/Output.c  `pkg-config --cflags --libs opencv` -lm -fopenmp
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

===============Reading Image: ./Cars2/11.jpg===============
Reading Prediction Input...
Found 30 candidates...
NMS Done,Got 5 Detections...
==============Time Spend==============
-- Init Input time 108.527363 ms
-- PreProcess time 0.218211 ms
-- NMS 0.092603 ms
-- Copied mask coeffs time 0.003226 ms
-- Handle_proto_test Avg: 1.032674 ms, Total:5.163371 ms
-- Draw Time: Avg:0.496093 ms, Total: 2.480467 ms
================================================

Total Compute time without init_input & drawing: 7.957878ms!!!!!!!

================================================
=========Drawing Mask and Labels Complete=========
Save predicted image into ./Prediction/Results/11.jpg.
5 Masks released
===============Saved Image Complete===============


===============Reading Image: ./Cars2/10.jpg===============
Reading Prediction Input...
Found 4 candidates...
NMS Done,Got 2 Detections...
==============Time Spend==============
-- Init Input time 215.656760 ms
-- PreProcess time 0.113110 ms
-- NMS 0.052805 ms
-- Copied mask coeffs time 0.002918 ms
-- Handle_proto_test Avg: 0.784779 ms, Total:1.569558 ms
-- Draw Time: Avg:1.200579 ms, Total: 2.401158 ms
================================================

Total Compute time without init_input & drawing: 4.139549ms!!!!!!!

================================================
=========Drawing Mask and Labels Complete=========
Save predicted image into ./Prediction/Results/10.jpg.
2 Masks released
===============Saved Image Complete===============


===============Reading Image: ./Cars2/5.jpg===============
Reading Prediction Input...
Found 20 candidates...
NMS Done,Got 3 Detections...
==============Time Spend==============
-- Init Input time 321.717361 ms
-- PreProcess time 0.130215 ms
-- NMS 0.043756 ms
-- Copied mask coeffs time 0.001969 ms
-- Handle_proto_test Avg: 0.583689 ms, Total:1.751067 ms
-- Draw Time: Avg:0.851249 ms, Total: 2.553746 ms
================================================

Total Compute time without init_input & drawing: 4.480753ms!!!!!!!

================================================
=========Drawing Mask and Labels Complete=========
Save predicted image into ./Prediction/Results/5.jpg.
3 Masks released
===============Saved Image Complete===============


===============Reading Image: ./Cars2/6.jpg===============
Reading Prediction Input...
Found 65 candidates...
NMS Done,Got 11 Detections...
==============Time Spend==============
-- Init Input time 427.527671 ms
-- PreProcess time 26.778343 ms
-- NMS 0.186088 ms
-- Copied mask coeffs time 0.007072 ms
-- Handle_proto_test Avg: 1.529826 ms, Total:16.828082 ms
-- Draw Time: Avg:1.877413 ms, Total: 20.651547 ms
================================================

Total Compute time without init_input & drawing: 64.451132ms!!!!!!!

================================================
=========Drawing Mask and Labels Complete=========
Save predicted image into ./Prediction/Results/6.jpg.
11 Masks released
===============Saved Image Complete===============


===============Reading Image: ./Cars2/13.jpg===============
Reading Prediction Input...
Found 2 candidates...
NMS Done,Got 1 Detections...
==============Time Spend==============
-- Init Input time 542.863286 ms
-- PreProcess time 1.571224 ms
-- NMS 0.044202 ms
-- Copied mask coeffs time 0.004234 ms
-- Handle_proto_test Avg: 0.435098 ms, Total:0.435098 ms
-- Draw Time: Avg:0.551168 ms, Total: 0.551168 ms
================================================

Total Compute time without init_input & drawing: 2.605926ms!!!!!!!

================================================
=========Drawing Mask and Labels Complete=========
Save predicted image into ./Prediction/Results/13.jpg.
1 Masks released
===============Saved Image Complete===============

*/