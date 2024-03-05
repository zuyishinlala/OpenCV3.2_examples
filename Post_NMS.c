#include "./Sources/Object.h"
#include "./Sources/Parameters.h"
#include "./Sources/Input.h"
#include "./Sources/Bbox.h"

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv/highgui.h>

#include <math.h>
int cvRound(double value) {return(ceil(value));}

static void sigmoid(int rowsize, int colsize, float *ptr)
{
    for (int i = 0; i < rowsize * colsize ; ++i, ++ptr)
    {
        *ptr = 1.0f / (1.0f + powf(2.71828182846, -*ptr));
    }
}

static float GetPixel(int x, int y, int width, int height, float *Src){
    if(x < 0 || x >= width || y < 0 || y >= height) return 0.f;
    return *(Src + y*width + x);
}

//Align Corner = False
static void BilinearInterpolate(float *Src, uint8_t *Tar, float Threshold, struct Bbox Bound){

    float src_width = MASK_SIZE_WIDTH, src_height = MASK_SIZE_HEIGHT;
    float tar_width = TRAINED_SIZE_WIDTH, tar_height = TRAINED_SIZE_WIDTH;

    float r_ratio = src_height / tar_height;
    float c_ratio = src_width / tar_width;
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
                     (1 - dc) *  dr * GetPixel(    ic, ir + 1, src_height, src_width, Src) +
                      dc * (1 - dr) * GetPixel(ic + 1,     ir, src_height, src_width, Src) +
                (1 - dc) * (1 - dr) * GetPixel(    ic,     ir, src_height, src_width, Src);
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
    float pred_mask[MASK_SIZE_HEIGHT][MASK_SIZE_WIDTH] = {0};
    float* mask_ptr = &pred_mask[0][0];

    // Obtain Uncropped Mask (size 160*160)
    for(int i = 0 ; i < MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH ; ++i, ++mask_ptr){
        float Pixel = 0.f;
        for(int c = 0 ; c < NUM_MASKS ; ++c){
            Pixel += maskcoeffs[c] * masks[c][i];
        }
        *mask_ptr = Pixel;
    }

    sigmoid(MASK_SIZE_HEIGHT, MASK_SIZE_WIDTH, &pred_mask[0][0]);
    // Bilinear Interpolate + Binary Threshold 
    // Obtain Uncropped Mask (size 640 * 640)
    BilinearInterpolate(&pred_mask[0][0], UncroppedMask, Binary_Thres, box);
}

// Rescale Bbox: Bounding Box positions to Real place
static void rescalebox(struct Bbox *Box, float src_size_w, float src_size_h, float tar_size_w, float tar_size_h){
    float ratio = fminf(src_size_w/ tar_size_w, src_size_h/tar_size_h);
    float padding_w = (src_size_w - tar_size_w * ratio) / 2, padding_h = (src_size_h - tar_size_h * ratio) / 2;
    Box->left   = (Box->left - padding_w) / ratio;
    Box->right  = (Box->right - padding_w) / ratio;
    Box->top    = (Box->top - padding_h) / ratio;
    Box->bottom = (Box->bottom - padding_h) / ratio;

    clamp(Box, tar_size_w, tar_size_h);
}

// Plot Label and Bounding Box
static void plot_box_and_label(const char* label, const struct Bbox* box, float mask_transparency, IplImage** mask, IplImage** ImgSrc){
    int boxthickness = 2;
    CvScalar BLUE = CV_RGB(50, 178, 255);
    CvScalar WHITE = CV_RGB(240, 240, 240);

    IplImage* MaskedImg = cvCloneImage(*ImgSrc);
    cvSet(MaskedImg, CV_RGB(0, 0, 255), *mask); //Specify the color

    // Draw Mask
    cvAddWeighted(*ImgSrc, 1.f - mask_transparency, MaskedImg, mask_transparency, 0, *ImgSrc);

    // Draw Bounding Box
    CvPoint tlp = cvPoint((int)box->left, (int)box->top);
    CvPoint brp = cvPoint((int)box->right, (int)box->bottom);
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

static inline void getMaskxyxy(int* xyxy, float org_size_w, float org_size_h, float tar_size_w, float tar_size_h){
    float ratio = fminf( org_size_w/tar_size_w, org_size_h/tar_size_h);
    int padding_w = (org_size_w - tar_size_w * ratio) / 2, padding_h = (org_size_h - tar_size_h * ratio) / 2;
    xyxy[0] = padding_w;              // left
    xyxy[1] = padding_h;              // top
    xyxy[2] = org_size_w - padding_w; // right
    xyxy[3] = org_size_h - padding_h; // bottom
    return;
}

//Obtain final mask (size : ORG_SIZE_HEIGHT, ORG_SIZE_WIDTH) and draw label
void RescaleMaskandDrawLabel(struct Object* obj, uint8_t* UnCropedMask, IplImage** ImgSrc, int* mask_xyxy){
    
    // uint8_t array to IplImage
    IplImage* SrcMask = cvCreateImageHeader(cvSize(TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT), IPL_DEPTH_8U, 1);   
    cvSetData(SrcMask, UnCropedMask, TRAINED_SIZE_WIDTH);

    // ROI Mask Region by using maskxyxy
    CvRect roiRect = cvRect(mask_xyxy[0], mask_xyxy[1], mask_xyxy[2] - mask_xyxy[0], mask_xyxy[3] - mask_xyxy[1]); // (left, top, width, height)
    cvSetImageROI(SrcMask, roiRect);
    
    // Obtain ROI image
    IplImage* roiImg = cvCreateImage(cvSize(roiRect.width, roiRect.height), SrcMask->depth, 1);
    cvCopy(SrcMask, roiImg, NULL);

    // Obtain Resized Mask
    IplImage* FinalMask = cvCreateImage(cvGetSize(*ImgSrc), roiImg->depth, 1);
    cvResize(roiImg, FinalMask, CV_INTER_LINEAR);

    // Draw Label and Task (int label to string)
    // plot_box_and_label("Pesudo Label", &obj->Rect, MASK_TRANSPARENCY, &FinalMask, ImgSrc);

    cvReleaseImage(&SrcMask);
    cvReleaseImage(&roiImg);
    cvReleaseImage(&FinalMask);
}