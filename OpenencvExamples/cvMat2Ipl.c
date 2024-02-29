#include<opencv2/imgcodecs/imgcodecs_c.h>
#include<stdio.h>
#include<math.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
int cvRound(double value) {return(ceil(value));}

int main(int argc,char ** argv) {
    //IplImage* img = cvLoadImage( argv[1], CV_LOAD_IMAGE_COLOR);
    CvMat* img2 = cvLoadImageM( argv[1], CV_LOAD_IMAGE_ANYCOLOR);
    if(!img2){
        printf("---No Img---");
        return 0;
    }

    cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Image", img2);
    
    cvWaitKey(0);
    cvDestroyAllWindows();
    cvReleaseMat(&img2);
    return 0;
}
