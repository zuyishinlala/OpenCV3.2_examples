#ifndef BBOX_H
#define BBOX_H

struct Bbox {
    float left, top, right, bottom;
};

float BoxArea(const struct Bbox* box);
void clamp(struct Bbox* box, float max_w, float max_h);
#endif // BBOX_H