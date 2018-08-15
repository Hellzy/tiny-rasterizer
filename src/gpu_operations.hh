#pragma once
#include <cuda_runtime.h>

#include "utils.hh"

/*
 * Projects every point of the scene in raster space
 *
 * - points: array of all the points in the scene
 * - point_nb: length of the array of points
 * - cam: the camera looking at the scene
 * - screen_w: width of the screen
 * - screen_h: height of the screen
 */
void projection_kernel(point_t* points, size_t point_nb, const cam_t& cam, size_t screen_w, size_t screen_h);

/*
 * Draws the scene on the screen
 *
 * - screen: framebuffer to draw on
 * - screen_h: screen height
 * - screen_w: screen width
 * - meshes: meshes of the scene
 * - z_buffer: depth buffer
 */
void draw_mesh_kernel(std::vector<color_t>& screen, size_t screen_h, size_t screen_w,
        const std::vector<mesh_t>& meshes, const std::vector<double>& z_buffer);

