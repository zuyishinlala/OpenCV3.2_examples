#ifndef BBOX_H
#define BBOX_H

struct Bbox {
    float x, y, width, height; // center_x, center_y, width, height
};

void xywh2xyxy();
#endif // BBOX_H