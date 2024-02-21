#include "Input.h"
#include <stdio.h>
#include <stdlib.h>
void ReadFile(float* Dst, int RowSize, int ColSize,  char* FileName){
    FILE *file;
    file = fopen( FileName, "r");

    if (file == NULL) {
        printf("Error opening file %s\n", FileName);
        return;
    }

    for(int c = 0 ; c < ColSize ; ++c){
        for(int r = 0 ; r < RowSize ; ++r){
            fscanf( file, "%f", Dst+r*ColSize+c);
        }
    }
    
    fclose(file);
}

void initPredInput(struct Pred_Input* input, char** argv){
    /*
    ================================================
      To-do: Read 9 files and store data into arrays
    ================================================
    */
    float* cls_ptr = &input->cls_pred[0][0];
    float* reg_ptr = &input->reg_pred[0][0];
    float* seg_ptr = &input->seg_pred[0][0];
    
    // Cls_Predictions
    ReadFile(cls_ptr, WIDTH0*HEIGHT0, NUM_CLASSES, argv[0]);

    cls_ptr += WIDTH0*HEIGHT0*NUM_CLASSES;
    ReadFile(&input->cls_pred, WIDTH1*HEIGHT1, NUM_CLASSES, argv[1]);

    cls_ptr += WIDTH1*HEIGHT1*NUM_CLASSES;
    ReadFile(&input->cls_pred, WIDTH2*HEIGHT2, NUM_CLASSES, argv[2]);

    // Reg_Predictions
    ReadFile(&input->reg_pred, WIDTH0*HEIGHT0, 4, argv[3]);

    reg_ptr += WIDTH0*HEIGHT0*4;
    ReadFile(&input->reg_pred, WIDTH1*HEIGHT1, 4, argv[4]);

    reg_ptr += WIDTH1*HEIGHT1*4;
    ReadFile(&input->reg_pred, WIDTH2*HEIGHT2, 4, argv[5]);

    // Seg_Predictions
    ReadFile(&input->seg_pred, WIDTH0*HEIGHT0, NUM_MASKS, argv[6]);

    seg_ptr += WIDTH0*HEIGHT0*NUM_MASKS;
    ReadFile(&input->seg_pred, WIDTH1*HEIGHT1, NUM_MASKS, argv[7]);

    seg_ptr += WIDTH1*HEIGHT1*NUM_MASKS;
    ReadFile(&input->seg_pred, WIDTH2*HEIGHT2, NUM_MASKS, argv[8]);
}