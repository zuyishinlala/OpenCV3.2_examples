#include <opencv2/highgui/highgui_c.h>  // 高层图形界面（C 接口）
#include <opencv2/imgproc/imgproc_c.h>  // 图像处理（C 接口）
#include <opencv2/imgcodecs/imgcodecs_c.h>  // 图像编解码（C 接口）
#include <stdio.h>
#include <math.h>
int cvRound(double value) {return(ceil(value));}

// Rewrite YOLOv6.cpp draw_label Function
void draw_label(IplImage* input_image, const char* label, int left, int top)
{
    int baseLine;
    CvSize label_size; 
    CvPoint tlc, brc;
    // Create Font
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.8, 0, 2, CV_AA);

    CvScalar BLACK = CV_RGB(0, 0, 0);
    CvScalar BLUE = CV_RGB(50, 178, 255);
    CvScalar WHITE = CV_RGB(240, 240, 240);

    cvGetTextSize(label, &font, &label_size, &baseLine);
    //top = max(top, label_size->height);
    top = (label_size.height > top) ? label_size.height : top;
    // Top left corner.
    tlc = cvPoint(left, top);
    // Bottom right corner.
    brc = cvPoint(left + label_size.width, top + label_size.height + baseLine);

    // Draw blue rectangle.
    cvRectangle(input_image, tlc, brc, BLUE, CV_FILLED, CV_AA, 0);

    // Put the label on the black rectangle.
    // cvPutText: The cvPoint is the BottomLeft Corner
    cvPutText(input_image, label, cvPoint(left,label_size.height + top), &font, WHITE);
}

int main(int argc, char** argv){
    IplImage* img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!img){
        printf("---No Img---");
        return 0;
    }
    /*
    CvPoint startPointV = cvPoint(100, 0);
    CvPoint endPointV = cvPoint(100, 500);
    cvLine(img, startPointV, endPointV, CV_RGB(200, 0, 0), 2, CV_AA, 0);

    CvPoint startPointH = cvPoint(0, 100);
    CvPoint endPointH = cvPoint(500, 100);
    cvLine(img, startPointH, endPointH, CV_RGB(200, 0, 0), 2, CV_AA, 0);
    */  
    CvScalar BLUE = CV_RGB(50, 178, 255);

    CvPoint tlp = cvPoint(110, 40);
    CvPoint brp = cvPoint(430, 360);
    cvRectangle(img, tlp, brp, BLUE, 2, CV_AA, 0);
    const char* str = "Sun Flower";
    // para(110, 40) is the top left of every Label
    draw_label( img, str, tlp.x, tlp.y);

    cvNamedWindow("Labeled", CV_WINDOW_AUTOSIZE);
    cvShowImage("Labeled", img);

    cvWaitKey(0);

    cvReleaseImage(&img);
}