#include "Bbox.h"

// Bbox.c Calculate Box area
float BoxArea(const struct Bbox* box){
    return (box->right - box->left) * (box->bottom - box->top);
}

// Bbox.c Make Sure the box is within [0: max_bound]
void clamp(struct Bbox* box, float max_w, float max_h){
    box->left = (box->left < 0) ? 0 : (box->left > max_w) ? max_w : box->left;
    box->right = (box->right < 0) ? 0 : (box->right > max_w) ? max_w : box->right;
    box->top = (box->top < 0) ? 0 : (box->top > max_h) ? max_h : box->top;
    box->bottom = (box->bottom < 0) ? 0 : (box->bottom > max_h) ? max_h : box->bottom;
}