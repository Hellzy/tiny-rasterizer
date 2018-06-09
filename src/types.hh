#pragma once

struct rgb_vector
{
    double r, g, b;
};

struct point
{
    double x, y, z;
};

using color_t = rgb_vector;
using point_t = point;
using vector_t = point;
