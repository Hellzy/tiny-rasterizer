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
    cam_ = parser.cam_get();
    screen_w_ = parser.screen_width_get();
    screen_h_ = parser.screen_height_get();
    screen_ = std::vector<color_t>(screen_w_ * screen_h_);
    mats_ = parser.mats_get();
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
    mesh_t* meshes_d;

    cudaMalloc(&meshes_d, sizeof(mesh_t) * meshes_.size());
    cudaMemcpy(meshes_d, meshes_.data(), sizeof(mesh_t) * meshes_.size(),
            cudaMemcpyHostToDevice);

    projection_kernel(meshes_d, meshes_.size(), cam_, screen_w_,
            screen_h_);
    auto vecs_d = tiles_dispatch_kernel(meshes_d,  meshes_.size(),
            screen_w_, screen_h_);

    dev_mat_t* mats_d = nullptr;
    cudaMalloc(&mats_d, sizeof(dev_mat_t) * mats_.size());
    cudaMemcpy(mats_d, mats_.data(), sizeof(dev_mat_t) * mats_.size(), cudaMemcpyHostToDevice);

    draw_mesh_kernel(screen_, screen_w_, screen_h_, meshes_d, meshes_.size(),
            vecs_d, mats_d, mats_.size());

    cudaFree(meshes_d);
    cudaFree(vecs_d);
    cudaFree(mats_d);
}
