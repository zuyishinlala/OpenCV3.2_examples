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

static void sigmoid(int rowsize, int colsize, float *ptr)
{
    for (int i = 0; i < rowsize * colsize ; ++i, ++ptr)
    {
        *ptr = 1.0f / (1.0f + powf(2.71828182846, -*ptr));
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

static float GetPixel(int x, int y, int width, int height, float *Src){
    if(x < 0 || x >= width || y < 0 || y >= height) return 0.f;
    return *(Src + y*width + x);
}

//Align Corner = False
static void BilinearInterpolate(float *Src, uint8_t *Tar, float Threshold, struct Bbox Bound){

    float src_width = MASK_SIZE_WIDTH, src_height = MASK_SIZE_HEIGHT;
    float tar_width = TRAINED_SIZE_WIDTH, tar_height = TRAINED_SIZE_HEIGHT;

    float r_ratio = src_height / tar_height;
    float c_ratio = src_width / tar_width;

    clamp(&Bound, tar_width, tar_height);

    // Perform Binary Threshold only in the Bounding Box Region
    for(int r = (int)Bound.top ; r <  (int)Bound.bottom ; ++r){
        for(int c =  (int)Bound.left ; c <  (int)Bound.right ; ++c){
            float PixelSum = 0.f;
            
            float dr = (r + 0.5) * r_ratio - 0.5;
            float dc = (c + 0.5) * c_ratio - 0.5;
            float ir = floorf(dr), ic = floorf(dc);

            dr = (dr < 0.f) ? 1.0f : ((dr > src_height - 1.0f) ? 0.f : dr - ir);
            dc = (dc < 0.f) ? 1.0f : ((dc > src_width - 1.0f) ? 0.f : dc - ic);

            PixelSum =     dc *  dr * GetPixel(ic + 1, ir + 1, src_height, src_width, Src) + 
                     (1 - dc) *  dr * GetPixel(ic    , ir + 1, src_height, src_width, Src) +
                      dc * (1 - dr) * GetPixel(ic + 1, ir    , src_height, src_width, Src) +
                (1 - dc) * (1 - dr) * GetPixel(ic    , ir    , src_height, src_width, Src);
            *(Tar + r * TRAINED_SIZE_WIDTH + c) = (PixelSum > Threshold) ? 255 : 0;
        }
    }
}

// Obtain Uncropped Mask
static void handle_proto_test(struct Object* obj, const float masks[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH], uint8_t* UncroppedMask)
{
    // Resize mask & Obtain Binary Mask
    // Matrix Multiplication
    float* maskcoeffs = obj->maskcoeff;
    struct Bbox box = obj->Rect;
   
    float Binary_Thres = 0.5f;
    float pred_mask[MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH] = {0};

    // Obtain Uncropped Mask (size 160*160)
    for(int i = 0 ; i < MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH ; ++i){
        float Pixel = 0.f;
        for(int c = 0 ; c < NUM_MASKS ; ++c){
            Pixel += maskcoeffs[c] * masks[c][i];
        }
        pred_mask[i] = Pixel;
    }

    sigmoid(MASK_SIZE_HEIGHT, MASK_SIZE_WIDTH, pred_mask);
    // Bilinear Interpolate + Binary Threshold 
    // Obtain Uncropped Mask (size 640 * 640)
    BilinearInterpolate(pred_mask, UncroppedMask, Binary_Thres, box);
}

// Rescale Bbox: Bounding Box positions to Real place
static void rescalebox(struct Bbox *Box, float src_size_w, float src_size_h, float tar_size_w, float tar_size_h){
    float ratio = fminf(src_size_w/ tar_size_w, src_size_h/tar_size_h);
    
    float padding_w = (src_size_w - tar_size_w * ratio) / 2;
    float padding_h = (src_size_h - tar_size_h * ratio) / 2;

    Box->left   = (Box->left - padding_w) / ratio;
    Box->right  = (Box->right - padding_w) / ratio;
    Box->top    = (Box->top - padding_h) / ratio;
    Box->bottom = (Box->bottom - padding_h) / ratio;

    clamp(Box, tar_size_w, tar_size_h);
}

static char* GetClassName(int ClassIndex){
    char* names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee"};
    return names[ClassIndex];
}

static CvScalar Generate_Color(int ClassIndex){
    const unsigned int hex_colors[20] = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A, 0x92CC17, 0x3DDB86, 0x1A9334, 0x00D4BB,
                                          0x2C99A8, 0x00C2FF, 0x344593, 0x6473FF, 0x0018EC, 0x8438FF, 0x520085, 0xCB38FF, 0xFF95C8, 0xFF37C7};
    int ColorIndex = ClassIndex % 20;
    return CV_RGB((hex_colors[ColorIndex] >> 16) & 0xFF,  (hex_colors[ColorIndex] >> 8) & 0xFF, hex_colors[ColorIndex] & 0xFF);
}

// Plot Label and Bounding Box
static void DrawMask(const int ClassLabel, float mask_transparency, IplImage* mask, IplImage* ImgSrc){
    CvScalar COLOR = Generate_Color(ClassLabel);
    // IplImage* MaskedImg = cvCloneImage(ImgSrc);
    // cvSet(MaskedImg, COLOR, mask); //Specify the color

    // cvNamedWindow("Final Output", CV_WINDOW_AUTOSIZE);
    // cvShowImage("Final Output", MaskedImg);
    // cvWaitKey(0);
    // cvDestroyAllWindows();
    // Draw Mask
    COLOR.val[0] *= mask_transparency;
    COLOR.val[1] *= mask_transparency;
    COLOR.val[2] *= mask_transparency;
    //cvAddWeighted(ImgSrc, 1.f - mask_transparency, MaskedImg, mask_transparency, 0, ImgSrc);
    cvAddS(ImgSrc, COLOR, ImgSrc, mask);
    //cvReleaseImage(&MaskedImg);
}

static void DrawLabel(const struct Bbox box, const int ClassLabel, const char* label, int boxthickness, CvScalar TextColor, IplImage* ImgSrc){
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

    cvGetTextSize(label, &font, &label_size, &baseLine);

    brp.x = left + label_size.width;
    brp.y = top;
    tlp.y -= (label_size.height+baseLine);

    // Draw Background
    cvRectangle(ImgSrc, tlp, brp, COLOR, CV_FILLED, CV_AA, 0);
    // Draw Label
    cvPutText(ImgSrc, label, cvPoint(left, top - baseLine), &font, TextColor);
}

static void RescaleMaskandDrawMask(struct Object* obj, uint8_t* UnCropedMask, IplImage* ImgSrc, int* mask_xyxy){

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
    IplImage* FinalMask = cvCreateImage(cvGetSize(ImgSrc), roiImg->depth, 1);
    cvResize(roiImg, FinalMask, CV_INTER_LINEAR);

    // Draw Label and Task (int label to string)
    DrawMask(obj->label, MASK_TRANSPARENCY, FinalMask, ImgSrc);

    cvReleaseImage(&SrcMask);
    cvReleaseImage(&roiImg);
    cvReleaseImage(&FinalMask);
}