#include "triangle.hh"
#include "utils.hh"

void Triangle::cam_project(const cam_t& cam)
{
    cam_project_point(cam, &p1);
    cam_project_point(cam, &p2);
    cam_project_point(cam, &p3);
}

void Triangle::screen_project(size_t width, size_t height)
{
    constexpr double ncp = 1;

    p1.x = ncp * (width / 2.0) * p1.x / -p1.z + width / 2.0;
    p1.y = ncp * (height / 2.0) * p1.y / -p1.z + height / 2.0;
    p1.z = -p1.z;

    p2.x = ncp * (width / 2.0) * p2.x / -p2.z + width / 2.0;
    p2.y = ncp * (height / 2.0) * p2.y / -p2.z + height / 2.0;
    p2.z = -p2.z;

    p3.x = ncp * (width / 2.0) * p3.x / -p3.z + width / 2.0;
    p3.y = ncp * (height / 2.0) * p3.y / -p3.z + height / 2.0;
    p3.z = -p3.z;

}

void Triangle::ndc_project(double l, double r, double b, double t)
{
    p1.x = 2 * p1.x / (r - l) - (r + l) / (r - l);
    p1.y = 2 * p1.y / (t - b) - (t + b) / (t - b);

    p2.x = 2 * p2.x / (r - l) - (r + l) / (r - l);
    p2.y = 2 * p2.y / (t - b) - (t + b) / (t - b);

    p3.x = 2 * p3.x / (r - l) - (r + l) / (r - l);
    p3.y = 2 * p3.y / (t - b) - (t + b) / (t - b);

    if (p1.x < -1)
        p1.x = -1;
    if (p1.y < -1)
        p1.y = -1;

    if (p2.x < -1)
        p2.x = -1;
    if (p2.y < -1)
        p2.y = -1;

    if (p3.x < -1)
        p3.x = -1;
    if (p3.y < -1)
        p3.y = -1;

    if (p1.x > 1)
        p1.x = 1;
    if (p1.y > 1)
        p1.y = 1;

    if (p2.x > 1)
        p2.x = 1;
    if (p2.y > 1)
        p2.y = 1;

    if (p3.x > 1)
        p3.x = 1;
    if (p3.y > 1)
        p3.y = 1;
}

void Triangle::raster_project(size_t width, size_t height)
{
    //do stuff
    p1.x = (p1.x + 1) / 2 * width;
    p1.y = (1 - p1.y) / 2 * height;

    p2.x = (p2.x + 1) / 2 * width;
    p2.y = (1 - p2.y) / 2 * height;

    p3.x = (p3.x + 1) / 2 * width;
    p3.y = (1 - p3.y) / 2 * height;
}

std::vector<point_t> Triangle::get_points() const
{
    return {p1, p2, p3};
}

bool Triangle::check_edges(const point_t& p)
{
    double res1 = edge_function(p1, p2, p);
    double res2 = edge_function(p2, p3, p);
    double res3 = edge_function(p3, p1, p);

    return (res1 >= 0 && res2 >= 0 && res3 >= 0) || (res1 < 0 && res2 < 0 && res3 < 0);
}
