#include "Bbox.h"

void xywh2xyxy(){
    
}

float BoxArea(const struct Bbox* box){
    return (box->right - box->left) * (box->bottom - box->top);
}