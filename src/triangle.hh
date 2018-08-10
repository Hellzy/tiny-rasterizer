#pragma once
#include "object.hh"

struct Triangle : public Object
{
    Triangle(point_t p1, point_t p2, point_t p3)
        : p1(p1), p2(p2), p3(p3)
    {}

    void cam_project(const cam_t& cam) override;
    void screen_project(size_t width, size_t height) override;
    void ndc_project(double l, double r, double b, double t) override;
    void raster_project(size_t width, size_t height) override;
    bool check_edges(const point_t& p) override;

    std::vector<point_t> get_points() const override;

    point_t p1;
    point_t p2;
    point_t p3;
};
