#include <stdio.h>

static char* GetClassName(int ClassIndex){
    char* names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee"};
    return names[ClassIndex];
}

int main(){
    char* Name = GetClassName(16);
    printf("%s\n", Name);
}