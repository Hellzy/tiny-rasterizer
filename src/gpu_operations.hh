#pragma once
#include <cuda_runtime.h>

#include "device_vector.hh"
#include "utils.hh"
#include "input_parser.hh"

/*
 * Projects every point of the scene in raster space
 *
 * - meshes_d: array of triangles allocated on GPU side
 * - mesh_nb: size of meshes_d
 * - cam: the camera looking at the scene
 * - screen_w: width of the screen
 * - screen_h: height of the screen
 */
void projection_kernel(mesh_t* meshes_d, size_t mesh_nb, const cam_t& cam,
        size_t screen_w, size_t screen_h);

/*
 * Dispatches each mech into a specific tile over the screen (or several if the mesh
 * overlaps several tiles. This accelerates further computations as it allows to
 * immediately know which mesh is in which during drawing kernel
 *
 * - meshes_d: array of triangles allocated on GPU side
 * - mesh_nb: size of meshes_d
 * - screen_w: width of the screen
 * - screen_h: height of the screen
 *
 *   RETURN VALUE: an array of device vectors, each holding mesh indexes
 *   indicating which meshes are in the tile the vector belongs to.
 */
device_vec_t* tiles_dispatch_kernel(mesh_t* meshes_d, size_t mesh_nb, size_t screen_w,
        size_t screen_h);

/*
 * Draws the scene on the screen
 *
 * - screen: framebuffer to draw onto
 * - screen_w: screen width
 * - screen_h: screen height
 * - meshes_d: array of triangles allocated on GPU side
 * - mesh_nb: size of meshes_d
 * - vecs: array of device vectors holding mesh indices (one vector per tile).
 */
void draw_mesh_kernel(host_vec_t<color_t>& screen, size_t screen_w, size_t screen_h,
        mesh_t* meshes_d, size_t mesh_nb, device_vec_t* vecs, dev_mat_t* mats,
        size_t mats_nb);


/**
 * Computes how many tiles the screen is made of by detecting how many thread
 * we can fit in a GPU block (one GPU block is basically one tile)
 *
 * - screen_w: screen width
 * - screen_h: screen height
 */
__host__ dim3 compute_tiles(size_t screen_w, size_t screen_h);
