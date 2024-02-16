#ifndef OBJECTS_H
#define OBJECTS_H
#include <stdio.h>
#include "Input.h"
#include "Parameters.h"
#include "Bbox.h"

struct Object {
    struct Bbox Rect;               // Bounding Box 4
    int label;                      // Classification label 1
    float conf;                     // Classification max prob score 1
    float* maskcoeff;               // 32
};

#endif  // OBJECTS_H
