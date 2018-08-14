#pragma once
#include <cuda_runtime.h>

#include "utils.hh"

void projection_kernel(point_t* points, size_t point_nb, const cam_t& cam, size_t screen_w, size_t screen_h);
