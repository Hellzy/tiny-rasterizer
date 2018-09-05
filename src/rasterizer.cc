#include <cmath>
#include <fstream>
#include <limits>
#include <iostream>

#include "gpu_operations.hh"
#include "input_parser.hh"
#include "rasterizer.hh"
#include "utils.hh"

void Rasterizer::load_scene(const std::string& filename)
{
    InputParser parser(filename);
    meshes_ = parser.meshes_get();
    z_buffer_ = std::vector<double>(screen_w_ * screen_h_,
            std::numeric_limits<double>::max());
    screen_ = std::vector<color_t>(screen_w_ * screen_h_);
}

void Rasterizer::write_scene(const std::string& filename) const
{
  std::ofstream ofs(filename);

  ofs << "P3\n" << screen_w_ << ' ' << screen_h_ << "\n255\n";

  for (size_t i = 0; i < screen_h_; ++i)
  {
    for (size_t j = 0; j < screen_w_ ; ++j)
    {
        const auto& pix = screen_[i * screen_w_ + j];
        int r = pix.r;
        int g = pix.g;
        int b = pix.b;

        ofs << r << ' ' << g << ' ' << b;

        if (j < screen_w_ - 1)
            ofs << "  ";
    }
    ofs << '\n';
  }
}
void Rasterizer::gpu_compute()
{
    projection_kernel(meshes_.data(), meshes_.size(), cam_, screen_w_,
            screen_h_);
    auto bitsets_d = tiles_dispatch_kernel(meshes_.data(),  meshes_.size(),
            screen_w_, screen_h_);

    draw_mesh_kernel(screen_, screen_w_, screen_h_, meshes_, bitsets_d);
    cudaFree(bitsets_d);
    bitset_t::release_memory();
}
