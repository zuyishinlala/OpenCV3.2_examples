#include<opencv2/imgcodecs/imgcodecs_c.h>
#include<stdio.h>
#include<math.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
int cvRound(double value) {return(ceil(value));}

int main(int argc,char** argv){ 
    CvMat* cvMatImage = cvLoadImageM("your_image.jpg", CV_LOAD_IMAGE_COLOR);

    // Convert CvMat* to IplImage*
    // IplImage* iplImageHeader = cvGetImage(cvMatImage,);
    
    // Now you can use iplImage as an IplImage*

    // Don't forget to release the CvMat* if it was allocated
    cvReleaseMat(&cvMatImage);
} 