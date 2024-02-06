#ifndef OBJECTS_H
#define OBJECTS_H

#define ORG_SIZE_HEIGHT 500
#define ORG_SIZE_HEIGHT 500

#define TRAINED_SIZE_HEIGHT 384
#define TRAINED_SIZE_WIDTH 640

#define MASK_SIZE_HEIGHT TRAINED_SIZE_HEIGHT/4
#define MASK_SIZE_WIDTH TRAINED_SIZE_WIDTH/4

#define CONF_THRESHOLD 0.3
#define NMS_THRESHOLD 0.5

#define NUM_CLASSES 80
#define NUM_MASKS 32

#define WIDTH0 TRAINED_SIZE_WIDTH / 8  // 0
#define WIDTH1 TRAINED_SIZE_WIDTH / 16 // 1
#define WIDTH2 TRAINED_SIZE_WIDTH / 32 // 2

#define HEIGHT0 TRAINED_SIZE_HEIGHT / 8
#define HEIGHT1 TRAINED_SIZE_HEIGHT / 16
#define HEIGHT2 TRAINED_SIZE_HEIGHT / 32

#define ROWSIZE HEIGHT0*WIDTH0 + HEIGHT1*WIDTH1 + HEIGHT2*WIDTH2

#define AGNOSTIC 0
#define MULTI_LABEL 0
#define ISSOLO 0
#define ANCHOR_BASED 0

struct Bbox {
    float x, y, width, height; // center_x, center_y, width, height
};

struct Object {
    float confidence;   // obj confidence score
    struct Bbox Rect;   // Bounding Box
    int label;          // classification label
    float prob;         // classification max prob score
    float** Mask;       // Final Binary Mask
};

#endif  // OBJECTS_H
