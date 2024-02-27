#include<opencv2/imgcodecs/imgcodecs_c.h>
#include<stdio.h>
#include<math.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
int cvRound(double value) {return(ceil(value));}
/*
    binary mask
    RGB img
    Color you want to blend
    Add Weight 
*/
#define HEIGHT 300
#define WIDTH 300

int main(int argc,char** argv){ 

    float data[2][HEIGHT * WIDTH] = {0};
    
    for(int row = 150 ; row < HEIGHT ; ++row){
        for(int col = 0 ; col < WIDTH ; ++col){
            data[0][row * WIDTH + col] = 1.f;
        }
    }

    IplImage* SrcMask = cvCreateImageHeader(cvSize(WIDTH, HEIGHT), IPL_DEPTH_32F, 1);   
    cvSetData(SrcMask, data[0], SrcMask->widthStep);
    

    cvNamedWindow("SrcMask", CV_WINDOW_AUTOSIZE);
    cvShowImage("SrcMask", SrcMask);

    cvWaitKey(0);
    cvDestroyAllWindows();

    cvReleaseImage(&SrcMask);
} 