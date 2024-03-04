#include "Input.h"

void ReadFile(float* Dst, int RowSize, int ColSize, const char* FileName){
    FILE *file = fopen( FileName, "r");

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

void ReadMaskInput(float* mask, int RowSize, int ColSize, const char *FileName) {
    FILE *file = fopen(FileName, "r");

    if (file == NULL) {
        printf("Error opening file %s\n", FileName);
        return;
    }

    for (int i = 0; i < RowSize; i++) {
        for (int j = 0; j < ColSize; j++) {
            if (fscanf(file, "%f", &mask[i * ColSize + j]) != 1) {
                printf("Error reading from file %s\n", FileName);
                fclose(file);
                return;
            }
        }
    }

    fclose(file);
}


void initPredInput(struct Pred_Input* input, float* mask_ptr, const char** argv){
    printf("=====Reading Prediction Input...   =====\n");
    
    float* cls_ptr = &input->cls_pred[0][0];
    float* reg_ptr = &input->reg_pred[0][0];
    float* seg_ptr = &input->seg_pred[0][0];
    
    // Cls_Predictions
    ReadFile(cls_ptr, WIDTH0*HEIGHT0, NUM_CLASSES, argv[2]);

    cls_ptr += WIDTH0*HEIGHT0*NUM_CLASSES;
    ReadFile(cls_ptr, WIDTH1*HEIGHT1, NUM_CLASSES, argv[3]);

    cls_ptr += WIDTH1*HEIGHT1*NUM_CLASSES;
    ReadFile(cls_ptr, WIDTH2*HEIGHT2, NUM_CLASSES, argv[4]);

    // Reg_Predictions
    ReadFile(reg_ptr, WIDTH0*HEIGHT0, 4, argv[5]);

    reg_ptr += WIDTH0*HEIGHT0*4;
    ReadFile(reg_ptr, WIDTH1*HEIGHT1, 4, argv[6]);

    reg_ptr += WIDTH1*HEIGHT1*4;
    ReadFile(reg_ptr, WIDTH2*HEIGHT2, 4, argv[7]);

    // Seg_Predictions
    ReadFile(seg_ptr, WIDTH0*HEIGHT0, NUM_MASKS, argv[8]);

    seg_ptr += WIDTH0*HEIGHT0*NUM_MASKS;
    ReadFile(seg_ptr, WIDTH1*HEIGHT1, NUM_MASKS, argv[9]);
 
    seg_ptr += WIDTH1*HEIGHT1*NUM_MASKS;
    ReadFile(seg_ptr, WIDTH2*HEIGHT2, NUM_MASKS, argv[10]);

    ReadMaskInput(mask_ptr, NUM_MASKS, MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH, argv[11]);
}