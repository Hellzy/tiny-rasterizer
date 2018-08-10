#pragma once
#include "types.hh"

template <typename T, unsigned dim2>
void mat_mult(T m1[9], T m2[dim2], T out[dim2])
{
    int out_off = 0;

    for (unsigned i = 0; i < 3; ++i)
    {
        for (unsigned j = 0; j < dim2 / 3; j++)
        {
            for (unsigned k = 0; k < 3; ++k)
                out[out_off] += m1[i * 3 + k] * m2[k * 3 / dim2 + j];

            ++out_off;
        }
    }
}

/**
 * e1:  first edge point
 * e2:  second edge point
 */
inline
double edge_function(point_t e1, point_t e2, point_t p)
{
    return ((p.x - e1.x) * (e2.y - e1.y) - (p.y - e1.y) * (e2.x - e1.x));
}

void cam_project_point(const cam_t& cam, point_t* p);
