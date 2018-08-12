#pragma once

#include <vector>

#include "types.hh"

class Rasterizer
{
public:
    /**
     * Opens a .obj file to load the scene into the rasterizer
     * - filename: name of the .obj file to open
     */
    void load_scene(const std::string& filename);

    /**
     * Writes the rasterized scene to filename
     * - filename: name of the file to write the output to
     */
    void write_scene(const std::string& filename) const;

    /**
     * Perorms the rasterization of the last loaded scene
     */
    void compute();

private:
    void project_scene();
    void cam_project();
    void screen_project();
    void ndc_project(double l, double r, double b, double t);
    void raster_project();

    void cam_project_point(point_t& p);

private:
    std::vector<mesh_t> meshes_;
    std::vector<double> z_buffer_;
    std::vector<color_t> screen_;

    cam_t cam_ = { {-0.3, 0.2, -2.5}, {1.2, 0, 0}, {0, 1.1, 0}, {0, 0, -1} };
    size_t screen_w_ = 800;
    size_t screen_h_ = 800;
};
