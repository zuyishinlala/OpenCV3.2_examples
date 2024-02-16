#include <stdio.h>
#include "Object.h"

int main(){
    struct Bbox boxes[5];
    for(int i = 0 ; i < 5 ; ++i){
        struct Bbox box = {i, i*2, i*3, i*4};
        boxes[i] = box;
    }
}