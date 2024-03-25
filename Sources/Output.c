#include "Output.h"

void init_Output(struct Output* output, int width, int height){
    output->ORG_IMAGE_WIDTH = width;
    output->ORG_IMAGE_HEIGHT = height;
}

void releaseAllMasks(struct Output* output){
    for(int i = 0 ; i < output->NumDetections ; ++i){
        cvReleaseImage(&output->Masks[i]);
    }
    printf("%d Masks released\n", output->NumDetections);
    output->NumDetections = 0;
    //memset(output->detections, 0, sizeof(Object) * MAX_DETECTIONS);
}