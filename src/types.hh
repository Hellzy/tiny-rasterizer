#pragma once

struct rgb_vector
{
    double r, g, b;
};

struct point
{
    double x, y, z;

    /*
     * Simple addition between two points
     */

    point operator+(const point& p);

    point& operator+=(const point& p);

    point operator-(const point& p);

    point& operator-=(const point& p);

    /*
     * Simple multiplication between two points
     */

    point operator*(const point& p);

    point& operator*=(const point& p);

    /*
     * Simple multiplication between a point and a scalar
     */

    template <typename T>
    point operator*(T scalar)
    {
        return point{x * scalar, y * scalar, z * scalar};
    }

    template <typename T>
    point& operator*=(T scalar)
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;

        return *this;
    }
};

using color_t = rgb_vector;
using point_t = point;
using vector_t = point;

struct Cam
{
    point_t pos;
    vector_t dir_x;
    vector_t dir_y;
    vector_t dir_z;
};

using cam_t = Cam;
