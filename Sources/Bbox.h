#ifndef BBOX_H
#define BBOX_H

struct Bbox {
    float left, top, right, bottom;
};

float BoxArea(const struct Bbox*);
void clamp(struct Bbox*, float , float );
#endif // BBOX_H