#ifndef OBJECTS_H
#define OBJECTS_H

struct Bbox {
    float x, y, width, height; // center_x, center_y, width, height
};

struct Object {
    struct Bbox Rect;               // Bounding Box
    int label;                      // Classification label
    float conf;                     // Classification max prob score
    float* maskcoeff;
};

#endif  // OBJECTS_H
