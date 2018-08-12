#pragma once

#include "obj-parser/src/types.hh"

using vector_t = point_t;

struct rgb_vector
{
    double r, g, b;
};

using color_t = rgb_vector;

struct Cam
{
    point_t pos;
    vector_t dir_x;
    vector_t dir_y;
    vector_t dir_z;
};

using cam_t = Cam;
