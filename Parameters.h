#ifndef PARAMETERS_H
#define PARAMETERS_H

#define NUM_CLASSES 30
#define NUM_MASKS 32

// image size 
#define ORG_SIZE_HEIGHT 500
#define ORG_SIZE_WIDTH 500

#define TRAINED_SIZE_HEIGHT 384
#define TRAINED_SIZE_WIDTH 640

#define MASK_SIZE_HEIGHT TRAINED_SIZE_HEIGHT/4
#define MASK_SIZE_WIDTH TRAINED_SIZE_WIDTH/4

#define WIDTH0 TRAINED_SIZE_WIDTH/8  // 0
#define WIDTH1 TRAINED_SIZE_WIDTH/16 // 1
#define WIDTH2 TRAINED_SIZE_WIDTH/32 // 2

#define HEIGHT0 TRAINED_SIZE_HEIGHT/8
#define HEIGHT1 TRAINED_SIZE_HEIGHT/16
#define HEIGHT2 TRAINED_SIZE_HEIGHT/32

#define ROWSIZE HEIGHT0*WIDTH0 + HEIGHT1*WIDTH1 + HEIGHT2*WIDTH2

#define MAX_DETECTIONS 200
#define CONF_THRESHOLD 0.3
#define NMS_THRESHOLD 0.5

// Bool
#define AGNOSTIC 0
#define MULTI_LABEL 0
#define ISSOLO 0
#define ANCHOR_BASED 0

// OpenCV parameters for drawing
#define MASK_TRANSPARENCY 0.2 // 0 to 1

#endif