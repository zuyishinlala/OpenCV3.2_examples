#include <opencv/cv.h>
#include <stdio.h>
#include <opencv/cxcore.h>
#include <math.h>
#include<opencv2/imgcodecs/imgcodecs_c.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
int cvRound(double value) {return(ceil(value));}
#define ROW 500
#define COL 200
int main() {
    // Example float 2D array
    float data [ROW * COL] = {0};
    
    for(int row = 50 ; row < ROW ; ++row){
        for(int col = 0 ; col < COL ; ++col){
            data[row*COL + col] = 0.8f;
        }
    }
    CvMat m = cvMat(ROW, COL, CV_32FC1, data);

    // Creates a buffer, no data inside. MUST need to initalize data
    IplImage* img = cvCreateImageHeader(cvSize(0, 0), 32, 3); 
    cvGetImage(&m, img);
    printf("Image Size %d %d", img->width, img->height);
    printf("Image Channel %d", img->nChannels);

    cvNamedWindow("Display Image", CV_WINDOW_AUTOSIZE);

    // Show the image in the window
    cvShowImage("Display Image", img);

    // Wait for a key press to close the window
    cvWaitKey(0);

    // Destroy the window
    cvDestroyWindow("Display Image");

    //printf("Buffer Size %d %d", buff->width, buff->height);
    

    //cvReleaseImage(&buff);
    cvReleaseImage(&img);
    return 0;
}
