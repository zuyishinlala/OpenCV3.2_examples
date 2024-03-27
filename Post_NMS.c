#include "./Sources/Object.h"
#include "./Sources/Parameters.h"
#include "./Sources/Input.h"
#include "./Sources/Bbox.h"

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv/highgui.h>
#include <omp.h>

#include <math.h>
int cvRound(double value) {return(ceil(value));}
float cvRoundf(float value) {return roundf(value);}

static char* names[] = {
        "drivable", "alternatives", "line", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

static unsigned int hex_colors[20] = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A, 0x92CC17, 0x3DDB86, 0x1A9334, 0x00D4BB,
                                          0x2C99A8, 0x00C2FF, 0x344593, 0x6473FF, 0x0018EC, 0x8438FF, 0x520085, 0xCB38FF, 0xFF95C8, 0xFF37C7};

static inline void sigmoid(int rowsize, int colsize, float *ptr)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rowsize; ++i) {
        for (int j = 0; j < colsize ; ++j) {
            int index = i * colsize + j;
            ptr[index] = 1.0f / (1.0f + expf(-ptr[index]));
        }
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

// Obtain Uncropped Mask
static void handle_proto_test(struct Object* obj, const float (*masks)[NUM_MASKS], uint8_t* UncroppedMask, float Binary_Thres)
{
    // Resize mask & Obtain Binary Mask
    // Matrix Multiplication
    float* maskcoeffs = obj->maskcoeff;
    struct Bbox box = obj->Rect;

    float pred_mask[MASK_SIZE_WIDTH * MASK_SIZE_HEIGHT] = {0};
    
    #pragma omp parallel for collapse(2) schedule(static, CHUNKSIZE)
    for (int i = 0; i < MASK_SIZE_HEIGHT; ++i) {
        for (int j = 0; j < MASK_SIZE_WIDTH; ++j) {
            float Pixel = 0.f;
            const int index = i * MASK_SIZE_WIDTH + j;
            for (int c = 0; c < NUM_MASKS; ++c) {
                Pixel += maskcoeffs[c] * masks[index][c];
            }
            pred_mask[index] = Pixel;
        }
    }
    
    sigmoid(MASK_SIZE_HEIGHT, MASK_SIZE_WIDTH, pred_mask);
    
    IplImage* SrcMask = cvCreateImageHeader(cvSize(MASK_SIZE_WIDTH, MASK_SIZE_HEIGHT), IPL_DEPTH_32F, 1); 
    cvSetData(SrcMask, pred_mask, SrcMask->widthStep);

    IplImage* SrcMask_Resized = cvCreateImage(cvSize(TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT), SrcMask->depth, 1); 
    cvResize(SrcMask, SrcMask_Resized, CV_INTER_LINEAR);
    
    // ROI Mask
    IplImage* ZeroMask = cvCreateImage(cvSize(TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT), SrcMask->depth, 1);
    cvZero(ZeroMask);
    cvRectangle(ZeroMask, cvPoint(box.left, box.top), cvPoint(box.right, box.bottom), cvScalar(1, 1, 1, 0), CV_FILLED, CV_AA, 0);

    // Keep the Mask only in the ROI region
    cvAnd(SrcMask_Resized, ZeroMask, SrcMask_Resized, NULL);

    // Thresholding
    cvThreshold(SrcMask_Resized, SrcMask_Resized, Binary_Thres, 1.f, CV_THRESH_BINARY);

    // float2uint8_t mask
    IplImage* SrcMask_uint8 = cvCreateImage(cvSize(TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT), IPL_DEPTH_8U, 1);
    cvConvertScale(SrcMask_Resized, SrcMask_uint8, 255, 0);

    memcpy(UncroppedMask, SrcMask_uint8->imageData, sizeof(uint8_t) * TRAINED_SIZE_WIDTH * TRAINED_SIZE_HEIGHT);

    cvReleaseImage(&SrcMask);
    cvReleaseImage(&SrcMask_Resized);
    cvReleaseImage(&ZeroMask);
    cvReleaseImage(&SrcMask_uint8);
}

// Rescale Bbox: Bounding Box positions to Real place
static void rescalebox(struct Bbox *Box, float src_size_w, float src_size_h, float tar_size_w, float tar_size_h){
    float ratio = fminf( src_size_w/tar_size_w, src_size_h/tar_size_h);
    
    float padding_w = (src_size_w - tar_size_w * ratio) / 2;
    float padding_h = (src_size_h - tar_size_h * ratio) / 2;

    Box->left   = (Box->left - padding_w) / ratio;
    Box->right  = (Box->right - padding_w) / ratio;
    Box->top    = (Box->top - padding_h) / ratio;
    Box->bottom = (Box->bottom - padding_h) / ratio;

    clamp(Box, tar_size_w, tar_size_h);

    Box->left = cvRoundf(Box->left);
    Box->right = cvRoundf(Box->right);
    Box->top = cvRoundf(Box->top);
    Box->bottom = cvRoundf(Box->bottom);
}

static char* GetClassName(int ClassIndex){
    return names[ClassIndex];
}

static CvScalar Generate_Color(int ClassIndex){
    int ColorIndex = ClassIndex % 20;
    return CV_RGB((hex_colors[ColorIndex] >> 16) & 0xFF,  (hex_colors[ColorIndex] >> 8) & 0xFF, hex_colors[ColorIndex] & 0xFF);
}

// Plot Label and Bounding Box
static void DrawMask(const int ClassLabel, float mask_transparency, IplImage* mask, IplImage* ImgSrc){
    CvScalar COLOR = Generate_Color(ClassLabel);

    // Draw Mask
    COLOR.val[0] *= mask_transparency;
    COLOR.val[1] *= mask_transparency;
    COLOR.val[2] *= mask_transparency;

    cvAddS(ImgSrc, COLOR, ImgSrc, mask);
}

static void DrawLabel(const struct Bbox box, const float conf, const int ClassLabel, int boxthickness, CvScalar TextColor, IplImage* ImgSrc){
    //String Append Label + Confidence
    char* Label = GetClassName(ClassLabel);
    char FinalLabel[20];
    strcpy(FinalLabel, Label);
    size_t len = strlen(FinalLabel);
    FinalLabel[len] = ' ';
    sprintf(FinalLabel + len + 1, "%.2f", conf);

    CvScalar COLOR = Generate_Color(ClassLabel);
    int left = (int)box.left, top = (int)box.top;
    // Draw Bounding Box
    CvPoint tlp = cvPoint(left , top);
    CvPoint brp = cvPoint((int)box.right, (int)box.bottom);
    cvRectangle(ImgSrc, tlp, brp, COLOR, boxthickness, CV_AA, 0);

    int baseLine;
    CvSize label_size; 
    CvFont font; // font for text
    cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1.f, 1.f, 0, boxthickness / 2, CV_AA);

    cvGetTextSize(FinalLabel, &font, &label_size, &baseLine);

    brp.x = left + label_size.width;
    brp.y = top;
    tlp.y -= (label_size.height+baseLine);

    // Draw Background
    cvRectangle(ImgSrc, tlp, brp, COLOR, CV_FILLED, CV_AA, 0);
    // Draw Label
    cvPutText(ImgSrc, FinalLabel, cvPoint(left, top - baseLine), &font, TextColor);
}

static void RescaleMask(IplImage** Dst, uint8_t* UnCropedMask, IplImage* ImgSrc, int* mask_xyxy) {
    // uint8_t array to IplImage
    IplImage* SrcMask = cvCreateImageHeader(cvSize(TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT), IPL_DEPTH_8U, 1);   
    cvSetData(SrcMask, UnCropedMask, SrcMask->widthStep);

    // ROI Mask Region by using maskxyxy
    CvRect roiRect = cvRect(mask_xyxy[0], mask_xyxy[1], mask_xyxy[2] - mask_xyxy[0], mask_xyxy[3] - mask_xyxy[1]); // (left, top, width, height)
    cvSetImageROI(SrcMask, roiRect);
    
    // Obtain ROI image
    IplImage* roiImg = cvCreateImage(cvSize(roiRect.width, roiRect.height), SrcMask->depth, 1);
    cvCopy(SrcMask, roiImg, NULL);

    // Obtain Resized Mask
    *Dst = cvCreateImage(cvGetSize(ImgSrc), SrcMask->depth, 1);
    cvResize(roiImg, *Dst, CV_INTER_LINEAR);

    /*
    cvNamedWindow("Resized Mask", CV_WINDOW_AUTOSIZE);
    cvShowImage("Resized Mask", *Dst);

    cvWaitKey(0);
    cvDestroyAllWindows();
    */
    // Draw Label and Task (int label to string)
    //DrawMask(Label, MASK_TRANSPARENCY, FinalMask, ImgSrc);

    cvReleaseImageHeader(&SrcMask);
    cvReleaseImage(&roiImg);
}