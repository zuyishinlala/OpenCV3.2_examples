#include<opencv2/imgcodecs/imgcodecs_c.h>
#include<stdio.h>
#include<math.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
int cvRound(double value) {return(ceil(value));}

void DisplayImg(IplImage* img){
    cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Image", img);

    // Wait for a key event and close the window
    cvWaitKey(0);
    cvDestroyAllWindows();
}   

int main(int argc,char** argv){ 
    IplImage* img = cvLoadImage( argv[1], CV_LOAD_IMAGE_COLOR);
    if(!img){
        printf("---No Img---");
        return 0;
    }
    DisplayImg(img);
    
   
    // Release the image
    cvReleaseImage(&img);
} 