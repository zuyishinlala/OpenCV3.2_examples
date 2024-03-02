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
int main(int argc,char** argv){ 
    IplImage* img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!img){
        printf("---No Img---");
        return 0;
    }

    CvMat* Mask = cvLoadImageM(argv[1], CV_LOAD_IMAGE_GRAYSCALE); // Mask
    cvThreshold(Mask, Mask, 200, 255, CV_THRESH_BINARY);             

    IplImage* MaskedImg = cvCloneImage(img);
    cvSet(MaskedImg, CV_RGB(0, 0, 255), Mask); //Specify the color
    
    cvNamedWindow("Org", CV_WINDOW_AUTOSIZE);
    cvShowImage("Org", img); // Masked Img

    cvNamedWindow("Image", CV_WINDOW_AUTOSIZE); 
    cvShowImage("Image", Mask);

    cvAddWeighted(img, 0.5, MaskedImg, 0.5, 0, img);
    cvNamedWindow("Image2", CV_WINDOW_AUTOSIZE);
    cvShowImage("Image2", img); // Masked Img

    cvWaitKey(0);
    cvDestroyAllWindows();
    cvReleaseImage(&img);
    cvReleaseMat(&Mask);
    cvReleaseImage(&MaskedImg);
} 