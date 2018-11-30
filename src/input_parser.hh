#pragma once

#include <string>
#include <utility>
#include <vector>

#include "types.hh"
#include "parser.hh"

class InputParser
{
public:
    InputParser();
    InputParser(const std::string& filename);

    void load(const std::string& filename);

    std::vector<vertex_t>&& vertices_get() { return std::move(vertices_); }
    std::vector<mesh_t>&& meshes_get() { return std::move(meshes_); }
    std::vector<dev_mat_t>&& mats_get() {return std::move(mats_);}
    cam_t&& cam_get() { return std::move(cam_); }
    size_t screen_width_get() const { return screen_w_; }
    size_t screen_height_get() const { return screen_h_; }

private:
    void parse_cfg();

private:
    std::vector<vertex_t> vertices_;
    std::vector<mesh_t> meshes_;
    std::vector<dev_mat_t> mats_;
    cam_t cam_;
    size_t screen_w_;
    size_t screen_h_;
};
