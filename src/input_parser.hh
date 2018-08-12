#pragma once

#include <string>
#include <utility>
#include <vector>

#include "types.hh"

class InputParser
{
public:
    InputParser() = default;
    InputParser(const std::string& filename);

    void load(const std::string& filename);

    std::vector<vertex_t>&& vertices_get() { return std::move(vertices_); }
    std::vector<mesh_t>&& meshes_get() { return std::move(meshes_); }

private:
    std::vector<vertex_t> vertices_;
    std::vector<mesh_t> meshes_;
};
