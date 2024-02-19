#ifndef BBOX_H
#define BBOX_H

struct Bbox {
    float left, top, right, bottom;
};

void xywh2xyxy();
float BoxArea(struct Bbox* box);
#endif // BBOX_H