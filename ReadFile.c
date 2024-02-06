#include <stdio.h>
#include <math.h>
#include "Object.h"

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

        if(type == "xywh"){
            for(float anchor_points_r = 0.5 ; anchor_points_r < row_bound ; ++anchor_points_r){
                for(float anchor_points_c = 0.5 ; anchor_points_c < col_bound ; ++anchor_points_c){
                    //lt, rb = torch.split(distance, 2, -1)
                    //no need to perform
                    
                    float* data = &distance[row][0]; // left, top, bottom, right
                    
                    // x1y1 = anchor_points - lt
                    // x2y2 = anchor_points + rb
                    float x1 = anchor_points_c - data[0];
                    float y1 = anchor_points_r - data[1];
                    float x2 = anchor_points_c + data[2];
                    float y2 = anchor_points_r + data[3];

                    // c_xy = (x1y1 + x2y2) / 2
                    // wh = x2y2 - x1y1
                    // pred_bboxes *= stride_tensor
                    data[0] = (x2 + x1) / 2 * stride; // center_x
                    data[1] = (y2 + y1) / 2 * stride; // center_y
                    data[2] = (x2 - x1) * stride;   // width
                    data[3] = (y2 - y1) * stride;   // height
                    ++row;
                }
            }
        }else{ // if type == "xyxy"
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

int main(int argc,char** argv){
    float cls_pred[ROWSIZE][NUM_CLASSES]; // class prob

    float reg_pred[ROWSIZE][4]; // position: [ROWSIZE][0:2] lt, [ROWSIZE][2:4] br, Stored in stride order: 8, 16, 32

    float seg_pred[ROWSIZE][1+NUM_MASKS]; // proto net index, mask

    float masks[NUM_MASKS][MASK_SIZE_HEIGHT*MASK_SIZE_WIDTH]; 
    /*
    ================================================
        To-do: Read files and store data into arrays
    ================================================
    */

    sigmoid(cls_pred, ROWSIZE, NUM_CLASSES);
    post_regpreds(reg_pred, "xywh");
}
