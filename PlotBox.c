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
#include <math.h>
int cvRound(double value) {return(ceil(value));}

// Plot Label and Bounding Box
static void plot_box_and_label(const char* label, const struct Bbox* box, float mask_transparency, IplImage **mask, IplImage **ImgSrc){
    int boxthickness = 2;
    CvScalar BLUE = CV_RGB(50, 178, 255);
    CvScalar WHITE = CV_RGB(240, 240, 240);

    IplImage* MaskedImg = cvCloneImage(*ImgSrc);
    cvSet(MaskedImg, CV_RGB(0, 0, 255), *mask); //Specify the color
    
    printf("Mask Depth and channel is: %d %d\n", MaskedImg->depth, MaskedImg->nChannels);
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

int main(int argc, char** argv){
    IplImage* Img = cvLoadImage( argv[1], CV_LOAD_IMAGE_COLOR);
    if(!Img){
        printf("---No Img---\n");
        return 0;
    }
    //IplImage* ImgResized = cvCreateImage(cvSize(WIDTH, HEIGHT), Img->depth, Img->nChannels);
    //cvResize(Img, ImgResized, CV_INTER_LINEAR);
    // Size is not restricted to IplImage size. It can be bigger!
    uint8_t data[TRAINED_SIZE_HEIGHT * TRAINED_SIZE_WIDTH * 3] = {0};

    for (int row = TRAINED_SIZE_HEIGHT/2 ; row < TRAINED_SIZE_HEIGHT ; ++row) {
        for (int col = 0; col < TRAINED_SIZE_WIDTH * 3; ++col) {
            data[row * TRAINED_SIZE_WIDTH + col] = 255; 
        }
    }

    IplImage* SrcMask = cvCreateImageHeader(cvSize(TRAINED_SIZE_WIDTH, TRAINED_SIZE_HEIGHT), IPL_DEPTH_8U, 1);   
    cvSetData(SrcMask, data, TRAINED_SIZE_WIDTH);
    
    cvNamedWindow("SrcMask", CV_WINDOW_AUTOSIZE);
    cvShowImage("SrcMask", SrcMask);

    // Wait for a key event and close the window
    cvWaitKey(0);
    cvDestroyAllWindows();

    // ROI Mask Region by using maskxyxy (left, top, right ,bottom)
    CvRect roiRect = cvRect(120, 120, 120, 120); // (left, top, width, height)
    cvSetImageROI(SrcMask, roiRect);

    // Obtain ROI image
    IplImage* roiImg = cvCreateImage(cvSize(roiRect.width, roiRect.height), SrcMask->depth, 1);
    cvCopy(SrcMask, roiImg, NULL);
    
    // Obtain Resized Mask
    IplImage* FinalMask = cvCreateImage(cvGetSize(Img), roiImg->depth, 1);
    cvResize(roiImg, FinalMask, CV_INTER_LINEAR);
    
    struct Bbox bound;
    bound.left = 100.f;
    bound.top = 100.f;
    bound.right = 200.f;
    bound.bottom = 200.f;

    // Draw Label and Task (int label to string)
    plot_box_and_label("Pesudo Label", &bound, MASK_TRANSPARENCY, &FinalMask, &Img);


    //IplImage* Mask = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE); // Mask
    //cvThreshold(Mask, Mask, 200, 255, CV_THRESH_BINARY);
    

    // Vertical
    //cvLine(Img, cvPoint(bound.left, bound.top), cvPoint(bound.left, Img->height), CV_RGB(200, 0, 0), 2, CV_AA, 0);

    // Horizontal
    //cvLine(Img, cvPoint(bound.left, bound.top), cvPoint(Img->width, bound.top), CV_RGB(200, 0, 0), 2, CV_AA, 0);
    
    cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Image", Img);

    cvNamedWindow("Mask", CV_WINDOW_AUTOSIZE);
    cvShowImage("Mask", roiImg);

    // Wait for a key event and close the window
    cvWaitKey(0);
    cvDestroyAllWindows();
    
    cvReleaseImage(&SrcMask);
    cvReleaseImage(&roiImg);
    cvReleaseImage(&FinalMask);
    cvReleaseImage(&Img);
    //cvReleaseImage(&ImgResized);
}