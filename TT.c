#include <opencv/cv.h>
#include <stdio.h>
#include <opencv/cxcore.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
int cvRound(double value) {return(ceil(value));}

#define HEIGHT 300
#define WIDTH 800

int main(int argc, char** argv) {
    uint8_t data[HEIGHT * WIDTH] = {0};

    // Fill the data array
    for(int row = 150 ; row < HEIGHT ; ++row){
        for(int col = 0 ; col < WIDTH ; ++col){
            data[row * WIDTH + col] = 255;
        }
    }

    // Create an IplImage header
    IplImage* img = cvCreateImageHeader(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1);
    // Set the data pointer
    cvSetData(img, data, WIDTH); // or simply WIDTH * sizeof(uchar)

    cvNamedWindow("Orginal Image", CV_WINDOW_AUTOSIZE);

    // Show the image in the window
    cvShowImage("Orginal Image", img);

    cvWaitKey(0);

    // Destroy the window
    cvDestroyWindow("Orginal Image");

    IplImage* roiImg = cvCreateImage(cvSize(100, 100), img->depth, img->nChannels);

    // Copy the ROI from the source image to the destination image

    printf("Before\n===%d, %d===\n", img->width, img->height);
    CvRect roiRect = cvRect(100, 100, 100, 100); // (x(left), y(top), width, height)
    
    // Create an ROI (region of interest) from the original image
    cvSetImageROI(img, roiRect);

    cvCopy(img, roiImg, NULL);

    printf("After\n===%d, %d===\n", roiImg->width, roiImg->height);
    // Create a window
    cvNamedWindow("Display Image", CV_WINDOW_AUTOSIZE);

    // Show the image in the window
    cvShowImage("Display Image", roiImg);


    // Wait for a key press to close the window
    cvWaitKey(0);
    // Destroy the window
    cvDestroyWindow("Display Image");

    // Release the image header
    cvReleaseImageHeader(&img);
    cvReleaseImage(&img);
    cvReleaseImage(&roiImg);
    return 0;
}