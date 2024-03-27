#ifndef PARAMETERS_H
#define PARAMETERS_H

#define NUM_CLASSES 3
#define NUM_MASKS 32


#define ORG_IMAGE_SIZE 1920 * 1080

#define TRAINED_SIZE_HEIGHT 288
#define TRAINED_SIZE_WIDTH 512

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
#define MASK_THRESHOLD 0.45f

#define MAX_FILENAME_LENGTH 256

// Bool
#define AGNOSTIC 0     // True: All detections do nms toegether  False: different class would do nms respectively
#define MULTI_LABEL 0  // True: 1 anchor box can have > 1 labels False: one box only have one label.
#define SAVEMASK 0     // True: Save Mask & positions & results  False: only save results
#define SAVEPERMASK 0  // True: Save Image per Class             False: Don't Save

#define CHUNKSIZE 18
#define NUMOFTHREADS 4 // Num of cores in CPU

// OpenCV parameters for drawing
#define MASK_TRANSPARENCY 0.8f // 0 to 1
#define READIMAGE_LIMIT 10
#endif