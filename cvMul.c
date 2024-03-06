#include<opencv2/imgcodecs/imgcodecs_c.h>
#include<stdio.h>
#include<math.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
int cvRound(double value) {return(ceil(value));}

int main(int argc, char** argv) {
    IplImage* img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
    if (!img) {
        printf("---No Img---");
        return 0;
    }

    IplImage *im32 = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 3);
    cvConvertScale(img, im32, 1/255.f, 0);

    // Create mask - this should be a separate image, not just a grayscale version of the original
    IplImage* Mask = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    IplImage* Mask32 = cvCreateImage(cvGetSize(Mask), IPL_DEPTH_32F, 1);

    cvConvertScale(Mask, Mask32, 1/255.f, 0);
    //cvThreshold(Mask32, Mask32, 0.5f, 1.f, CV_THRESH_BINARY_INV);
    // Apply image processing to create your mask from the original image
    IplImage* CLR = cvCreateImage(cvGetSize(Mask32), IPL_DEPTH_32F, 3);
    CvScalar RED = cvScalar(0.2f, 0.8f, 0.45, 0.0);
    // Fill the image with the specified color (RGB: 200, 100, 50)
    cvSet(CLR, RED, NULL);

    cvNamedWindow("MaskImg", CV_WINDOW_AUTOSIZE);
    cvShowImage("MaskImg", Mask32); // Display     
    cvWaitKey(0);

    IplImage* ExtendMask = cvCreateImage(cvGetSize(Mask32), IPL_DEPTH_32F, 3);
    cvMerge(Mask32, Mask32, Mask32, NULL, ExtendMask);

    // cvMerge();
    cvMul(CLR, ExtendMask, ExtendMask, 1);
    
    cvNamedWindow("Src_Img", CV_WINDOW_AUTOSIZE);
    cvShowImage("Src_Img", im32); // Display 

    cvNamedWindow("BlendedMask", CV_WINDOW_AUTOSIZE);
    cvShowImage("BlendedMask", ExtendMask); // Display 

    cvWaitKey(0);
    cvDestroyAllWindows();

    cvAddWeighted(im32, 0.8, ExtendMask, 0.2, 0, im32);

    cvNamedWindow("FinalImg", CV_WINDOW_AUTOSIZE);
    cvShowImage("FinalImg", im32); // Display 

    cvWaitKey(0);
    cvDestroyAllWindows();

    // Release resources
    cvReleaseImage(&img);
    cvReleaseImage(&im32);
    cvReleaseImage(&Mask);
    cvReleaseImage(&Mask32);
    cvReleaseImage(&CLR);
    cvReleaseImage(&ExtendMask);

    return 0;
}
