#ifndef OBJECTS_H
#define OBJECTS_H

struct Bbox {
    float x, y, width, height; // center_x, center_y, width, height
};

struct Object {
    struct Bbox Rect;
    int label;
    float prob;
};

#endif  // OBJECTS_H
