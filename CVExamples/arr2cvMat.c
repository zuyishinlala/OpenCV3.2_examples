
#include <opencv/cv.h>
#include <stdio.h>
#include <opencv/cxcore.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
int cvRound(double value) {return(ceil(value));}

#define HEIGHT 300
#define WIDTH 300

#define ORG_WIDTH 200
#define ORG_HEIGHT 587
int main(int argc, char** argv) {
    // Example float 2D array
    uint8_t data[HEIGHT * WIDTH * 3] = {0};
    uint8_t data_resized[ORG_WIDTH*ORG_HEIGHT*3] = {0};


    for (int row = HEIGHT/2; row < HEIGHT; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            data[row * WIDTH + col] = 200;
        }
    }
    CvMat* m_ptr = NULL;
    CvMat* m_ptr_100 = NULL;
    CvMat* m_ptr_resized = NULL;
    CvMat m = cvMat( WIDTH, HEIGHT, CV_8UC1, data);
    m_ptr = &m;

    CvMat m_3h = cvMat( ORG_WIDTH, ORG_HEIGHT, CV_8UC1, data_resized);
    m_ptr_100 = &m_3h;
    /*
    // Adjust the destination size to be larger 
    CvMat dst = cvMat(WIDTH * 2, HEIGHT * 2, CV_32FC1, datadouble);
    
    cvMerge(&m, &m, &m, NULL, &m_3h);

    cvMul(img, &m_3h, &m_3h, 1);

    printf("type m: %d, type dst: %d\n", m.type, dst.type);

    cvResize(&m, &dst, CV_INTER_LINEAR);
    */
    uint8_t datas[50*50] = {0};
    CvMat m_100x100 = cvMat(100, 100, CV_8UC1, datas);
    cvGetSubRect( &m, &m_100x100, cvRect(110, 110, 100, 100));
    m_ptr_100 = &m_100x100;

    cvNamedWindow("Org_Img_100", CV_WINDOW_AUTOSIZE);
    cvShowImage("Org_Img_100", &m_100x100);

    cvResize(&m, &m_3h, CV_INTER_LINEAR);



    cvWaitKey(0);
    cvDestroyAllWindows();

    cvReleaseMat(&m_ptr);
    cvReleaseMat(&m_ptr_100);
    cvReleaseMat(&m_ptr_resized);
    return 0;
}