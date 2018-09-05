#pragma once
#include <cuda_runtime.h>

#include "device_bitset.hh"
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
void projection_kernel(mesh_t* meshes_d, size_t mesh_nb, const cam_t& cam,
        size_t screen_w, size_t screen_h);

/*
 * This kernel dispatches the meshes to every tile of the screen
 */
bitset_t* tiles_dispatch_kernel(mesh_t* meshes_d, size_t mesh_nb, size_t screen_w,
        size_t screen_h);

/*
 * Draws the scene on the screen
 *
 * - screen: framebuffer to draw on
 * - screen_h: screen height
 * - screen_w: screen width
 * - meshes: meshes of the scene
 * - z_buffer: depth buffer
 */
void draw_mesh_kernel(host_vec_t<color_t>& screen, size_t screen_w, size_t screen_h,
        mesh_t* meshes_d, size_t mesh_nb, bitset_t* bitsets);


__host__ dim3 compute_tiles(size_t screen_w, size_t screen_h);
