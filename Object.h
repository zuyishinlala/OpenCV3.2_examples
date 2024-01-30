#ifndef OBJECTS_H
#define OBJECTS_H

struct Bbox {
    float x, y, width, height; // center_x, center_y, width, height
};

struct Object {
    float confidence;   // obj confidence score
    struct Bbox Rect;   // Bounding Box
    int label;          // classification label
    float prob;         // classification max prob score
};

#endif  // OBJECTS_H
