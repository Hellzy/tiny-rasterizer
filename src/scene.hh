#pragma once
#include <vector>

#include "object.hh"
#include "types.hh"

struct Scene
{
    Scene(size_t width = 0, size_t height = 0);

    size_t width;
    size_t height;
    std::vector<color_t> screen;
    std::vector<object_ptr_t> objects;
    cam_t eye;
};
