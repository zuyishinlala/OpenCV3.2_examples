#include "Object.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
int cvRound(double value) {return(ceil(value));}

/*
float intersection_area(const struct Bbox a,const struct Bbox b) {
    float x_overlap = fmax(0, fmin(a.x + a.height / 2, b.x + b.height / 2) - fmax(a.x - a.height / 2, b.x - b.height / 2));
    float y_overlap = fmax(0, fmin(a.y + a.width / 2, b.y + b.width / 2) - fmax(a.y - a.width / 2, b.y - b.width / 2));

    return x_overlap * y_overlap;
}

static void nms_sorted_bboxes(const struct Object* faceobjects, int n, int* picked, float nms_threshold) {
    int i, j;
    int picked_count = 0;
    float* areas = (float*)malloc(n * sizeof(float));
    
    for (i = 0; i < n; i++)
        areas[i] = faceobjects[i].Rect.height * faceobjects[i].Rect.width;
    

    for (i = 0; i < n; i++) {
        int keep = 1;
        for (j = 0; j < picked_count; j++) {
            float inter_area = intersection_area(faceobjects[i].Rect, faceobjects[picked[j]].Rect);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep) {
            picked[picked_count++] = i;
        }
    }
    free(areas);
}
int main() {
    // Example usage
    struct Bbox rect1 = {2, 2, 4, 4}; // center_x=2, center_y=2, width=4, height=4
    struct Bbox rect2 = {4, 4, 4, 4}; // center_x=4, center_y=4, width=4, height=4
    
    float area = intersection_area(rect1, rect2);
    
    printf("Intersection area: %.2f\n", area);
    
    return 0;
}
*/