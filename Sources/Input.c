#include "Input.h"

static void ReadFile(float* Dst, int RowSize, int ColSize, const char* FileName, int ImageIndex){
    FILE *file = fopen( FileName, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", FileName);
        return;
    }

    for(int i = 0 ; i < ImageIndex ; ++i){
        for(int c = 0 ; c < ColSize ; ++c){
            for(int r = 0 ; r < RowSize ; ++r){
                float num;
                if (fscanf( file, "%f", &num) != 1) {
                    printf("Error reading from file %s\n", FileName);
                    return;
                }
            }
        }
    }
    
    for(int c = 0 ; c < ColSize ; ++c){
        for(int r = 0 ; r < RowSize ; ++r){
            if (fscanf( file, "%f", Dst+r*ColSize+c) != 1) {
                printf("Error reading from file %s\n", FileName);
                printf("Error at Index: %d\n", r*ColSize+c);
                fclose(file);
                return;
            }
        }
    }
    fclose(file);
}

static void ReadMaskInput(float* mask, int RowSize, int ColSize, const char *FileName, int ImageIndex) {
    FILE *file = fopen(FileName, "r");

    if (file == NULL) {
        printf("Error opening file %s\n", FileName);
        return;
    }

    for(int i = 0 ; i < ImageIndex ; ++i){
        for(int c = 0 ; c < ColSize ; ++c){
            for(int r = 0 ; r < RowSize ; ++r){
                float num ;
                if (fscanf( file, "%f", &num) != 1) {
                    printf("Error reading from file %s\n", FileName);
                    return;
                }
            }
        }
    }
    
    for (int i = 0; i < RowSize; i++) {
        for (int j = 0; j < ColSize; j++, mask++) {
            if (fscanf(file, "%f", mask) != 1) {
                printf("Error reading from file %s\n", FileName);
                fclose(file);
                return;
            }
        }
    }

    fclose(file);
}
/*
void initPredInput_pesudo(struct Pred_Input* input, float* mask_ptr, const char** argv){
    float* cls_ptr = &input->cls_pred[0][0];
    float* reg_ptr = &input->reg_pred[0][0];
    float* seg_ptr = &input->seg_pred[0][0];

    ReadMaskInput(reg_ptr, ROWSIZE, 4, argv[2]);
    ReadMaskInput(cls_ptr, ROWSIZE, NUM_CLASSES, argv[3]);
    ReadMaskInput(seg_ptr, ROWSIZE, NUM_MASKS, argv[4]);
    ReadMaskInput(mask_ptr, NUM_MASKS, MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH, argv[5]);
}
*/
void initPredInput(struct Pred_Input* input, float* mask_ptr, const char** argv, int ImageIndex){
    printf("Reading Prediction Input...\n");
    float* cls_ptr = &input->cls_pred[0][0];
    float* reg_ptr = &input->reg_pred[0][0];
    float* seg_ptr = &input->seg_pred[0][0];
    
    // Cls_Predictions
    ReadFile(cls_ptr, WIDTH0*HEIGHT0, NUM_CLASSES, argv[2], ImageIndex);

    cls_ptr += WIDTH0*HEIGHT0*NUM_CLASSES;
    ReadFile(cls_ptr, WIDTH1*HEIGHT1, NUM_CLASSES, argv[3], ImageIndex);

    cls_ptr += WIDTH1*HEIGHT1*NUM_CLASSES;
    ReadFile(cls_ptr, WIDTH2*HEIGHT2, NUM_CLASSES, argv[4], ImageIndex);

    // Reg_Predictions
    ReadFile(reg_ptr, WIDTH0*HEIGHT0, 4, argv[5], ImageIndex);

    reg_ptr += WIDTH0*HEIGHT0*4;
    ReadFile(reg_ptr, WIDTH1*HEIGHT1, 4, argv[6], ImageIndex);

    reg_ptr += WIDTH1*HEIGHT1*4;
    ReadFile(reg_ptr, WIDTH2*HEIGHT2, 4, argv[7], ImageIndex);

    // Seg_Predictions
    ReadFile(seg_ptr, WIDTH0*HEIGHT0, NUM_MASKS, argv[8], ImageIndex);

    seg_ptr += WIDTH0*HEIGHT0*NUM_MASKS;
    ReadFile(seg_ptr, WIDTH1*HEIGHT1, NUM_MASKS, argv[9], ImageIndex);
 
    seg_ptr += WIDTH1*HEIGHT1*NUM_MASKS;
    ReadFile(seg_ptr, WIDTH2*HEIGHT2, NUM_MASKS, argv[10], ImageIndex);

    //ReadMaskInput(mask_ptr, NUM_MASKS, MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH, argv[11], ImageIndex);
    ReadFile(mask_ptr, MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH, NUM_MASKS, argv[11], ImageIndex);
}

/*
  gcc main.c -o T ./Sources/Input.c ./Sources/Bbox.c  `pkg-config --cflags --libs opencv` -lm
 ./T ./ImgData.txt ./outputs/cls_preds8.txt ./outputs/cls_preds16.txt ./outputs/cls_preds32.txt ./outputs/reg_preds8.txt ./outputs/reg_preds16.txt ./outputs/reg_preds32.txt ./outputs/seg_preds8.txt ./outputs/seg_preds16.txt ./outputs/seg_preds32.txt ./outputs/mask_input.txt
*/