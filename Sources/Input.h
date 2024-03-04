#ifndef INPUTS_H
#define INPUTS_H
#include <stdio.h>
#include <stdlib.h>
#include "Parameters.h"

struct Pred_Input{
    float cls_pred[ROWSIZE][NUM_CLASSES]; // class prob
    float reg_pred[ROWSIZE][4];           // Bbox distance. Stored in stride order: 8, 16, 32
    float seg_pred[ROWSIZE][NUM_MASKS];   // mask coefficients
};

void initPredInput(struct Pred_Input*, float*,const char**);
void initPredInput_pesudo(struct Pred_Input*, float*,const char**);
void ReadMaskInput(float*, int, int,const char*);
void ReadFile(float*, int, int,const  char*);
#endif // OBJECTS_H
