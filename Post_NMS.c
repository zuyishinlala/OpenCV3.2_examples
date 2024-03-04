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
    for (int i = 0; i < rowsize * colsize; ++i, ++ptr)
    {
        *ptr = 1.0f / (1.0f + powf(2.71828182846, -*ptr));
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

static void handle_proto_test(int NumDetections, const struct Object* ValidDetections, const float masks[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH],   uint8_t (* UnCropedMask)[TRAINED_SIZE_HEIGHT*TRAINED_SIZE_WIDTH], CvSize OrgImg_Size)
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

// Rescale Bbox: Bounding Box positions to Real place
static void rescalebox(const int NumDetections, struct Object *Detections,  float src_size_w, float src_size_h, float tar_size_w, float tar_size_h){
    float ratio = minf(tar_size_w/src_size_w, tar_size_h/src_size_h);
    float padding_w = (src_size_w - tar_size_w * ratio) / 2, padding_h = (src_size_h - src_size_h * ratio) / 2;

    for(int i = 0 ; i < NumDetections ; ++i){
        struct Bbox *Box = &Detections[i].Rect;

        Box->left   = (Box->left - padding_w) / ratio;
        Box->right  = (Box->right - padding_w) / ratio;
        Box->top    = (Box->top - padding_h) / ratio;
        Box->bottom = (Box->bottom - padding_h) / ratio;

        clamp(Box, tar_size_w, tar_size_h);
    }
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
static void RescaleMaskandDrawLabel( int NumDetections, const struct Object *Detections, const uint8_t (* UnCropedMask)[TRAINED_SIZE_HEIGHT*TRAINED_SIZE_WIDTH], IplImage** ImgSrc){
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
        cvSetData(SrcMask, UnCropedMask[i], TRAINED_SIZE_WIDTH);

        // ROI Mask Region by using maskxyxy (left, top, right ,bottom)
        CvRect roiRect = cvRect(mask_xyxy[0], mask_xyxy[1], mask_xyxy[2] - mask_xyxy[0], mask_xyxy[3] - mask_xyxy[1]); // (left, top, width, height)
        cvSetImageROI(SrcMask, roiRect);

        // Obtain ROI image
        IplImage* roiImg = cvCreateImage(cvSize(roiRect.width, roiRect.height), SrcMask->depth, 1);
        cvCopy(SrcMask, roiImg, NULL);

        // Obtain Resized Mask
        IplImage* FinalMask = cvCreateImage(cvGetSize(*ImgSrc), roiImg->depth, 1);
        cvResize(roiImg, FinalMask, CV_INTER_LINEAR);

        // Draw Label and Task (int label to string)
        plot_box_and_label("Pesudo Label", &Detections[i].Rect, MASK_TRANSPARENCY, &FinalMask, ImgSrc);

        cvReleaseImage(&SrcMask);
        cvReleaseImage(&roiImg);
        cvReleaseImage(&FinalMask);
    }
}