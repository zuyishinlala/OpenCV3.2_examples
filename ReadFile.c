#include <stdio.h>
#include <math.h>
#include "Object.h"
#include "Parameters.h"

static void sigmoid(float (*arr)[NUM_CLASSES], int rowsize, int colsize)
{
    for(int r = 0 ; r < rowsize; r++){
        for(int c = 0 ; c < colsize; c++){
            arr[r][c] = 1.0f / (1.0f + powf(2.71828182846, -arr[r][c]));
        }
    }
}

static void post_regpreds(float (*distance)[4], char* type){
    // dist2bbox & generate_anchor in YOLOv6
    int row = 0;
    float stride = 8.f;
    for(int stride_index = 0 ; stride_index < 3 ; ++stride_index){
        float row_bound, col_bound;
        switch(stride_index){
            case 0:
                row_bound = HEIGHT0;
                col_bound = WIDTH0;
                break;
            case 1:
                row_bound = HEIGHT1;
                col_bound = WIDTH1;
                break;
            case 2:
                row_bound = HEIGHT2;
                col_bound = WIDTH2;
                break;
            default:
                printf("stride_index out of bound\n");
                return;
        }        
        for(float anchor_points_r = 0.5 ; anchor_points_r < row_bound ; ++anchor_points_r){
            for(float anchor_points_c = 0.5 ; anchor_points_c < col_bound ; ++anchor_points_c){
                //lt, rb = torch.split(distance, 2, -1)
                //no need to perform
                
                float* data = &distance[row][0]; // left, top, bottom, right

                // x1y1 = anchor_points - lt
                data[0] = anchor_points_c - data[0];
                data[1] = anchor_points_r - data[1];
                // x2y2 = anchor_points + rb
                data[2] += anchor_points_c;         // anchor_points_c + data[2]
                data[3] += anchor_points_r;         // anchor_points_r + data[3]
                ++row;
            }
        }
        if(type == "xywh"){
            // c_xy = (x1y1 + x2y2) / 2
            // wh = x2y2 - x1y1
            for(int i = 0 ; i < ROWSIZE ; ++i){
                float x1 = distance[i][0], y1 = distance[i][1], x2 = distance[i][2], y2 = distance[i][3];
                distance[i][0] = (x2 + x1)/2; // center_x
                distance[i][1] = (y2 + y1)/2; // center_y
                distance[i][2] = (x2 - x1);   // width
                distance[i][3] = (y2 - y1);   // height
            }
        }
        // pred_bboxes *= stride_tensor
        for(int i = 0 ; i < ROWSIZE ; ++i){
            for(int j = 0 ; j < 4 ; ++j){
                distance[i][j] *= stride;
            }
        }
        stride *= 2;
    }
}

static void handle_proto_test(){
    // Resize to the size of trained Image & Obtain Binary Mask
    // Matrix Multiplication
}

static void rescale_mask(){
    // Obtain Original Mask for Orginial Image
}

static void max_classpred(float (*cls_pred)[NUM_CLASSES], float* max_predictions, int* class_index){
    // Obtain max_prob and the max class index
    for(int i = 0 ; i < ROWSIZE ; ++i){
        float* predictions = &cls_pred[i][0]; // a pointer to the first column of each row
        float max_pred = 0;
        int max_class_index = -1;
        for(int class_idx = 0 ; class_idx < NUM_CLASSES ; ++class_idx){
            if(max_pred < predictions[class_idx]){
                max_pred = predictions[class_idx];
                max_class_index = class_idx;
            }
        }
        max_predictions[i] = max_pred;
        class_index[i] = max_class_index;
    }
}

static void xyxy2xywh(){

}

int main(int argc,char** argv){
    
    // passed data from npu
    // data[0]
    float cls_pred[ROWSIZE][NUM_CLASSES]; // class prob
    float reg_pred[ROWSIZE][4]; // position: [ROWSIZE][0:2] lt, [ROWSIZE][2:4] br, Stored in stride order: 8, 16, 32
    // data[1] 
    float masks[NUM_MASKS][MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH]; 
    // data[2]
    float seg_pred[ROWSIZE][NUM_MASKS]; // mask coefficients
    // passed data from npu

    float max_clsprob[ROWSIZE] = {0};
    int max_class_index[ROWSIZE] = {0};

    // For NMS
    struct Object ValidDetections[MAX_DETECTIONS]; // Object position is stored as xywh
    int CountValidDetect = 0;
    /*
    ================================================
        To-do: Read files and store data into arrays
    ================================================
    */
    max_classpred(cls_pred, max_clsprob, max_class_index);
    sigmoid(cls_pred, ROWSIZE, NUM_CLASSES);
    post_regpreds(reg_pred, "xyxy"); 
}
