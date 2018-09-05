#pragma once

#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "obj-parser/src/types.hh"

using vector_t = point_t;

struct rgb_vector
{
    double r, g, b;
};

using color_t = rgb_vector;

struct Cam
{
    point_t pos;
    vector_t dir_x;
    vector_t dir_y;
    vector_t dir_z;
};

using cam_t = Cam;

struct BoundingBox
{
    point_t top_l;
    point_t top_r;
    point_t bot_l;
    point_t bot_r;
};

using bbox_t = BoundingBox;

template <typename T>
using pinned_alloc_t = thrust::system::cuda::experimental::pinned_allocator<T>;

template <typename T>
using host_vec_t = thrust::host_vector<T, pinned_alloc_t<T>>;
