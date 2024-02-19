#include <stdio.h>
#include <math.h>
#include "Object.h"
#include "Input.h"


static void sigmoid(int rowsize, int colsize, float *ptr)
{
    for (int i = 0; i < rowsize * colsize; ++i, ++ptr)
    {
        *ptr = 1.0f / (1.0f + powf(2.71828182846, -*ptr));
    }
}

static void post_regpreds(float (*distance)[4], char *type)
{
    // dist2bbox & generate_anchor in YOLOv6
    int row = 0;
    float stride = 8.f;
    for (int stride_index = 0; stride_index < 3; ++stride_index)
    {
        float row_bound, col_bound;
        switch (stride_index)
        {
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
        for (float anchor_points_r = 0.5; anchor_points_r < row_bound; ++anchor_points_r)
        {
            for (float anchor_points_c = 0.5; anchor_points_c < col_bound; ++anchor_points_c)
            {
                // lt, rb = torch.split(distance, 2, -1)
                // no need to perform
                float *data = &distance[row][0]; // left, top, right, bottom

                // x1y1 = anchor_points - lt
                data[0] = anchor_points_c - data[0];
                data[1] = anchor_points_r - data[1];
                // x2y2 = anchor_points + rb
                data[2] += anchor_points_c; // anchor_points_c + data[2]
                data[3] += anchor_points_r; // anchor_points_r + data[3]
                ++row;
            }
        }
        if (type == "xywh")
        {
            // c_xy = (x1y1 + x2y2) / 2
            // wh = x2y2 - x1y1
            for (int i = 0; i < ROWSIZE; ++i)
            {
                float x1 = distance[i][0], y1 = distance[i][1], x2 = distance[i][2], y2 = distance[i][3];
                distance[i][0] = (x2 + x1) / 2; // center_x
                distance[i][1] = (y2 + y1) / 2; // center_y
                distance[i][2] = (x2 - x1);     // width
                distance[i][3] = (y2 - y1);     // height
            }
        }
        // pred_bboxes *= stride_tensor
        for (int i = 0; i < ROWSIZE; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                distance[i][j] *= stride;
            }
        }
        stride *= 2;
    }
}


static void handle_proto_test(const struct Object* ValidDetections, const int masks[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH], float (*FinalMask)[TRAINED_SIZE_HEIGHT][TRAINED_SIZE_WIDTH], int NumDetections)
{
    // Resize mask & Obtain Binary Mask
    // Matrix Multiplication
    /*
    - matrix multiplication. 32 * masks
    - sigmoid
    - reshape to [MASK_SIZE_HEIGHT][MASK_SIZE_WIDTH]
    - bilinear interpolate to [TRAINED_SIZE_HEIGHT][TRAINED_SIZE_WIDTH]
    - crop mask 
    - binary threshold
    */
    for(int d = 0 ; d < NumDetections ; ++d){
        const struct Object Detection = ValidDetections[d];

        float *maskcoeffs = Detection.maskcoeff;
        float pred_mask[MASK_SIZE_HEIGHT][MASK_SIZE_WIDTH] = {0};
        float* ptr = &pred_mask[0][0];

        // Matrix Multiplication
        for(int i = 0 ; i < MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH ; ++i, ++ptr){
            for(int c = 0 ; c < NUM_MASKS ; ++c){
                *ptr += maskcoeffs[c] * masks[c][i];
            }
        }

        sigmoid(MASK_SIZE_HEIGHT, MASK_SIZE_WIDTH, pred_mask);

        // Bilinear Interpolate. passed value(org_size, final_size)
        // mask size to trained size
        
        // Crop Mask (init FinalMask = 0) + Perform Binary Threshold
        float Threshold = 0.5;
        int left = fmax(0, floorf(Detection.Rect.left)), top = fmax(0, floorf(Detection.Rect.top));
        int right = fmin(TRAINED_SIZE_WIDTH, ceilf(Detection.Rect.right)), bottom = fmin(TRAINED_SIZE_HEIGHT, ceilf(Detection.Rect.right));

        for(int r = top ; r < bottom ; ++r){
            for(int c = left ; c < right ; ++c){
                if(pred_mask[r][c] > Threshold){
                    FinalMask[d][r][c] = 1;     // set to True
                }
            }
        }
    }
}

static inline void getMaskxyxy(int* xyxy){
    float ratio = fminf( TRAINED_SIZE_HEIGHT/ORG_SIZE_HEIGHT, TRAINED_SIZE_WIDTH/ORG_SIZE_WIDTH);
    int padding_h = (TRAINED_SIZE_HEIGHT - ORG_SIZE_HEIGHT * ratio) / 2, padding_w = (TRAINED_SIZE_WIDTH - ORG_SIZE_WIDTH * ratio) / 2;
    xyxy[0] = padding_w; // left
    xyxy[1] = padding_h; // top
    xyxy[2] = ORG_SIZE_HEIGHT - padding_w; // right
    xyxy[3] = ORG_SIZE_WIDTH - padding_h; // bottom
    return;
}

static void plot_box_draw_label(){

}

int main(int argc, char **argv)
{
    char* Bboxtype = "xyxy";

    // passed data from npu
    struct Pred_Input input;
    float Mask_Input[NUM_MASKS][MASK_SIZE_HEIGHT * MASK_SIZE_WIDTH];

    // Result of NMS
    struct Object ValidDetections[MAX_DETECTIONS]; 
    int NumDetections = 0;

    int Binary_Mask[MAX_DETECTIONS][TRAINED_SIZE_HEIGHT][TRAINED_SIZE_WIDTH] = {0}; // Binary mask
    int mask_xyxy[4] = {0}; // the real mask in the resized image. left top bottom right

    initPredInput(&input, argv);

    sigmoid(ROWSIZE, NUM_CLASSES, input.cls_pred);
    post_regpreds(input.reg_pred, Bboxtype);

    //non_max_suppression_seg(input, "None", ValidDetections, &NumDetections);

    handle_proto_test(ValidDetections, Mask_Input, Binary_Mask, NumDetections);  // [:NumDetections] is the output
    getMaskxyxy(mask_xyxy);
    for(int i = 0 ; i < NumDetections ; ++i){
        // resize mask by using openCV
        
    }
}
