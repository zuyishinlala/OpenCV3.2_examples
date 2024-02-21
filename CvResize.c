#include <stdio.h>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "Object.h"
#include "Input.h"
#include "Bbox.h"

#include <math.h>
int cvRound(double value) {return(ceil(value));}

int main(int argc,char ** argv) {
    // Load the image
    IplImage* image = cvLoadImage(argv[1], CV_LOAD_IMAGE_UNCHANGED);

    // Check if the image is loaded successfully
    if(!image) {
        printf("Error: Could not read the image.\n");
        return -1;
    }

    // Define the new dimensions
    int newWidth = 300;
    int newHeight = 200;

    // Create an image for the resized result
    IplImage* resizedImage = cvCreateImage(cvSize(newWidth, newHeight), image->depth, image->nChannels);

    // Resize the image
    cvResize(image, resizedImage, CV_INTER_LINEAR);

    // Show the original and resized images
    cvNamedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Resized Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Original Image", image);
    cvShowImage("Resized Image", resizedImage);

    // Wait for a key press
    cvWaitKey(0);

    // Release the images
    cvReleaseImage(&image);
    cvReleaseImage(&resizedImage);

    // Destroy the windows
    cvDestroyAllWindows();

    return 0;
}
