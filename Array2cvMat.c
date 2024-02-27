
#include <opencv/cv.h>
#include <stdio.h>
#include <opencv/cxcore.h>
#include <math.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
int cvRound(double value) {return(ceil(value));}

#define HEIGHT 300
#define WIDTH 300

#define ORG_WIDTH 200
#define ORG_HEIGHT 587
int main(int argc, char** argv) {
    // Example float 2D array
    float data[HEIGHT * WIDTH] = {0};
    float datadouble[HEIGHT * WIDTH * 4] = {0};
    float data3h[HEIGHT * WIDTH][3] = {0};

    IplImage* img = cvCreateImage(cvSize(ORG_WIDTH, ORG_HEIGHT), IPL_DEPTH_32F, 3);

    CvScalar COLOR = cvScalar(0.0, 0., 200.0, 0); // B, G, R if 32F [0:1] if 8U [0:255]
    cvSet(img, COLOR, NULL);

    cvNamedWindow("Org_Img", CV_WINDOW_AUTOSIZE);
    cvShowImage("Org_Img", img);

    cvWaitKey(0);
    cvDestroyAllWindows();

    cvReleaseImage(&img);
    return 0;
    for (int row = 150; row < HEIGHT; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            data[row * WIDTH + col] = 1.f;
        }
    }

    CvMat m = cvMat( WIDTH, HEIGHT, CV_32FC1, data);

    CvMat m_3h = cvMat( WIDTH, HEIGHT, CV_32FC3, data3h);

    // Adjust the destination size to be larger 
    CvMat dst = cvMat(WIDTH * 2, HEIGHT * 2, CV_32FC1, datadouble);

    cvMerge(&m, &m, &m, NULL, &m_3h);

    cvMul(img, &m_3h, &m_3h, 1);

    printf("type m: %d, type dst: %d\n", m.type, dst.type);

    cvResize(&m, &dst, CV_INTER_LINEAR);

    cvNamedWindow("Org_Img", CV_WINDOW_AUTOSIZE);
    cvShowImage("Org_Img", &m_3h);

    cvNamedWindow("Dst_Img", CV_WINDOW_AUTOSIZE);
    cvShowImage("Dst_Img", &dst);

    cvWaitKey(0);
    cvDestroyAllWindows();
    
    cvReleaseImage(&img);
    return 0;
}