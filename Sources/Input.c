#include "Input.h"

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
    printf("=====Reading Prediction Input...   =====\n");
    
    float* cls_ptr = &input->cls_pred[0][0];
    float* reg_ptr = &input->reg_pred[0][0];
    float* seg_ptr = &input->seg_pred[0][0];
    
    // Cls_Predictions
    ReadFile(cls_ptr, WIDTH0*HEIGHT0, NUM_CLASSES, argv[0]);

    cls_ptr += WIDTH0*HEIGHT0*NUM_CLASSES;
    ReadFile(cls_ptr, WIDTH1*HEIGHT1, NUM_CLASSES, argv[1]);

    cls_ptr += WIDTH1*HEIGHT1*NUM_CLASSES;
    ReadFile(cls_ptr, WIDTH2*HEIGHT2, NUM_CLASSES, argv[2]);

    // Reg_Predictions
    ReadFile(reg_ptr, WIDTH0*HEIGHT0, 4, argv[3]);

    reg_ptr += WIDTH0*HEIGHT0*4;
    ReadFile(reg_ptr, WIDTH1*HEIGHT1, 4, argv[4]);

    reg_ptr += WIDTH1*HEIGHT1*4;
    ReadFile(reg_ptr, WIDTH2*HEIGHT2, 4, argv[5]);

    // Seg_Predictions
    ReadFile(seg_ptr, WIDTH0*HEIGHT0, NUM_MASKS, argv[6]);

    seg_ptr += WIDTH0*HEIGHT0*NUM_MASKS;
    ReadFile(seg_ptr, WIDTH1*HEIGHT1, NUM_MASKS, argv[7]);
 
    seg_ptr += WIDTH1*HEIGHT1*NUM_MASKS;
    ReadFile(seg_ptr, WIDTH2*HEIGHT2, NUM_MASKS, argv[8]);
}