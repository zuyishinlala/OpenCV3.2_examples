#ifndef BBOX_H
#define BBOX_H

struct Bbox {
    float left, top, right, bottom;
};

void xywh2xyxy();
float BoxArea(const struct Bbox*);
#endif // BBOX_H