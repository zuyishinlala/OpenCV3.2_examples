#ifndef OBJECTS_H
#define OBJECTS_H
#include <stdio.h>
#include "Input.h"
#include "Parameters.h"
#include "Bbox.h"

struct Object {
    struct Bbox Rect;               // Bounding Box
    int label;                      // Classification label
    float conf;                     // Classification max prob score
    float* maskcoeff;               // 32
};

#endif  // OBJECTS_H