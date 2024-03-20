#ifndef PARAMETERS_H
#define PARAMETERS_H

#define NUM_CLASSES 80
#define NUM_MASKS 32

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

#define MAX_DETECTIONS 50
#define CONF_THRESHOLD 0.25f
#define NMS_THRESHOLD 0.45f

#define MAX_FILENAME_LENGTH 256

// Bool
#define AGNOSTIC 0     // True: we do class-independent nms.   False: different class would do nms respectively.
#define MULTI_LABEL 0  // True: one box can have multi labels. False: one box only have one label.
#define SAVEMASK 0     // Save Mask & positions & results, otherwise only save results

// OpenCV parameters for drawing
#define MASK_TRANSPARENCY 0.8f // 0 to 1
#define READIMAGE_LIMIT 10

#endif