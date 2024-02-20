#include<opencv/cv.h>
#include<opencv/highgui.h>
#include<opencv2/imgcodecs/imgcodecs_c.h>
#include<math.h>
#include<stdio.h>
int cvRound(double value) {return(ceil(value));}

int main(int argc,char** argv) {
    // Read an image
    IplImage* image = cvLoadImage("./Images/img1.jpg", CV_LOAD_IMAGE_COLOR);
    if (!image) {
        printf("Error: Couldn't load image.\n");
        return -1;
    }

    // Define the region of interest (ROI) coordinates
    CvRect roiRect = cvRect(100, 100, 200, 150); // (x, y, width, height)

    // Create an ROI (region of interest) from the original image
    cvSetImageROI(image, roiRect);

    // Create a copy of the ROI
    IplImage* roiImage = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
    cvCopy(image, roiImage, NULL);

    // Reset the ROI to the full image size
    cvResetImageROI(image);

    // Display the original image and the ROI
    cvNamedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("ROI Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Original Image", image);
    cvShowImage("ROI Image", roiImage);
    cvWaitKey(0);

    // Release the images
    cvReleaseImage(&image);
    cvReleaseImage(&roiImage);

    return 0;
}
