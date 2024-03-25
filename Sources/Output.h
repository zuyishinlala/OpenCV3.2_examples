#include <opencv/cv.h>
#include "Object.h"
#include "Parameters.h"

struct Output {
    struct Object detections[MAX_DETECTIONS];
    IplImage* Masks[MAX_DETECTIONS];
    int NumDetections;
    int ORG_IMAGE_WIDTH;
    int ORG_IMAGE_HEIGHT;
};

// init output
void init_Output(struct Output* output, int width, int height);
void releaseAllMasks(struct Output* output);