#include "Object.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
int cvRound(double value) {return(ceil(value));}

static float intersection_area(const struct Bbox a, const struct Bbox b) {
    float x_overlap = fmax(0, fmin(a.x + a.height / 2, b.x + b.height / 2) - fmax(a.x - a.height / 2, b.x - b.height / 2));
    float y_overlap = fmax(0, fmin(a.y + a.width / 2, b.y + b.width / 2) - fmax(a.y - a.width / 2, b.y - b.width / 2));
    return x_overlap * y_overlap;
}

static void nms_sorted_bboxes(const struct Object* faceobjects, int n, int* picked, float nms_threshold) {
    int i, j;
    int picked_count = 0;
    float* areas = (float*)malloc(n * sizeof(float));

    for (i = 0; i < n; i++) {
        areas[i] = (faceobjects[i].Rect.width) * (faceobjects[i].Rect.height);
    }
    /*
    for (i = 0 ; i < n ; i++) {
        int keep = 1;
        for (j = 0 ; j < picked_count; j++) {
            float inter_area = intersection_area(faceobjects[i].Rect, faceobjects[picked[j]].Rect);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep) {
            picked[picked_count++] = i;
        }
    }
    */
    for (i = 0 ; i < n ; i++) {
        int keep = 1;
        for (j = i + 1 ; j < picked_count; j++) {
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