#include<opencv2/imgcodecs/imgcodecs_c.h>
#include<stdio.h>
#include<math.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
int cvRound(double value) {return(ceil(value));}

void DrawPoly(IplImage* img,CvPoint** pts,int *npts,int NumOfPo,CvScalar FilledColor){
    /*
    If the functioned use at the same time, overlayed area will not fill color.
    If called function seperately, overlayed area will fill color.
    */
    cvFillPoly(img, pts, npts, NumOfPo, FilledColor, CV_AA, 0);
    return;
}

int main(int argc,char** argv){ 
    CvScalar BLUE = CV_RGB( 0, 10, 200); 
    CvScalar GREEN = CV_RGB( 0, 200, 10);
    IplImage* img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!img){
        printf("---No Img---");
        return 0;
    }
    CvPoint greenRect[] = {cvPoint(200, 200), cvPoint(300, 200), cvPoint(300, 300), cvPoint(200, 300)};
    CvPoint* pts_array[] = {greenRect};
    int npts[] = {sizeof(greenRect)/sizeof(CvPoint)};
    int Len = sizeof(npts) / sizeof(int);

    DrawPoly(img, pts_array, npts, Len, GREEN);

    CvPoint greenTriangle[] = {cvPoint(100, 100), cvPoint(300, 100), cvPoint(200, 300)};
    pts_array[0] = greenTriangle;
    npts[0] = sizeof(greenTriangle)/sizeof(CvPoint);
    Len = sizeof(npts) / sizeof(int);
    DrawPoly(img, pts_array, npts, Len, BLUE);

    cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Image", img);
    // Wait for a key event and close the window
    cvWaitKey(0);
    cvDestroyAllWindows();
   
    // Release the image
    cvReleaseImage(&img);
} 