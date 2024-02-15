#ifndef INPUTS_H
#define INPUTS_H
#include "Parameters.h"

struct Pred_Input{
    // data[0]
    float cls_pred[ROWSIZE][NUM_CLASSES]; // class prob
    float reg_pred[ROWSIZE][4];           // position: [ROWSIZE][0:2] lt, [ROWSIZE][2:4] br, Stored in stride order: 8, 16, 32
    // data[1]
    // float masks[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH];
    // data[2]
    float seg_pred[ROWSIZE][NUM_MASKS]; // mask coefficients
};

// void initInput();
#endif // OBJECTS_H
