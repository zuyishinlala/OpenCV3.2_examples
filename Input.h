#ifndef INPUTS_H
#define INPUTS_H
#include "Parameters.h"

struct Pred_Input{
    float cls_pred[ROWSIZE][NUM_CLASSES]; // class prob
    float reg_pred[ROWSIZE][4];           // Bbox distance. Stored in stride order: 8, 16, 32
    float seg_pred[ROWSIZE][NUM_MASKS];   // mask coefficients
};

void initPredInput(struct Pred_Input*, char**);
#endif // OBJECTS_H
