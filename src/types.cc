#include "types.hh"

point point::operator+(const point& p)
{
    return point{x + p.x, y + p.y, z + p.z};
}

point& point::operator+=(const point& p)
{
    x += p.x;
    y += p.y;
    z += p.z;

    return *this;
}

point point::operator-(const point& p)
{
    return point{x - p.x, y - p.y, z - p.z};
}

point& point::operator-=(const point& p)
{
    x -= p.x;
    y -= p.y;
    z -= p.z;

    return *this;
}

point point::operator*(const point& p)
{
    return point{x * p.x, y * p.y, z * p.z};
}

point& point::operator*=(const point& p)
{
    x *= p.x;
    y *= p.y;
    z *= p.z;

    return *this;
}
