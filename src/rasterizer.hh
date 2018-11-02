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
     * Performs the rasterization of the loaded scene using cuda
     */
    void gpu_compute();

private:
    thrust::host_vector<mesh_t, pinned_alloc_t<mesh_t>> meshes_;
    thrust::host_vector<color_t, pinned_alloc_t<color_t>> screen_;

    cam_t cam_;
    size_t screen_w_;
    size_t screen_h_;
};
