#ifdef BENCH
# include <chrono>

# define BEGIN_BENCH() auto gpu_start = std::chrono::system_clock::now();
# define END_BENCH(bench_name)\
    auto gpu_end = std::chrono::system_clock::now();\
    std::chrono::duration<double> elapsed_gpu = gpu_end - gpu_start;\
    std::cout << "Elapsed " << bench_name << " : " << elapsed_gpu.count() << "s\n";
#endif

#include <cmath>
#include <iostream>

#include "gpu_operations.hh"

template <typename T, unsigned dim2>
__device__ void device_mat_mult(T m1[9], T m2[dim2], T out[dim2])
{
    int out_off = 0;

    for (unsigned i = 0; i < 3; ++i)
    {
        for (unsigned j = 0; j < dim2 / 3; j++)
        {
            for (unsigned k = 0; k < 3; ++k)
                out[out_off] += m1[i * 3 + k] * m2[k * 3 / dim2 + j];

            ++out_off;
        }
    }
}

__device__ void cam_project_point(const cam_t& cam, point_t& p)
{
    double rot_mat[] =
    {
        cam.dir_x.x, cam.dir_x.y, cam.dir_x.z,
        cam.dir_y.x, cam.dir_y.y, cam.dir_y.z,
        cam.dir_z.x, cam.dir_z.y, cam.dir_z.z
    };

    p.x -= cam.pos.x;
    p.y -= cam.pos.y;
    p.z -= cam.pos.z;

    double trans_mat[] = { p.x, p.y, p.z };
    double out_mat[3] = { 0 };

    device_mat_mult<double, 3>(rot_mat, trans_mat, out_mat);

    p.x = out_mat[0];
    p.y = out_mat[1];
    p.z = out_mat[2];
}

__global__ void cuda_project_points(mesh_t* meshes, size_t mesh_nb, cam_t cam, size_t screen_w, size_t screen_h)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < mesh_nb)
    {
        auto mesh = meshes[idx];
        point_t points[3] =  { mesh.v1.pos, mesh.v2.pos, mesh.v3.pos };

        for (int i = 0; i < 3; ++i)
        {
            auto& p = points[i];
            constexpr double ncp = 1;

            cam_project_point(cam, p);

            /* Screen projection */
            p.x = ncp * (screen_w / 2.0) * p.x / -p.z + screen_w / 2.0;
            p.y = ncp * (screen_h / 2.0) * p.y / -p.z + screen_h / 2.0;
            p.z = -p.z;

            /* NDC projection */
            double l = 0;
            double r = screen_w;
            double b = 0;
            double t = screen_h;

            p.x = 2 * p.x / (r - l) - (r + l) / (r - l);
            p.y = 2 * p.y / (t - b) - (t + b) / (t - b);

            /* Raster projection */
            p.x = (p.x + 1) / 2 * screen_w;
            p.y = (1 - p.y) / 2 * screen_h;
        }

        mesh.v1.pos = points[0];
        mesh.v2.pos = points[1];
        mesh.v3.pos = points[2];

        meshes[idx] = mesh;
    }
}

void projection_kernel(mesh_t* meshes_d, size_t mesh_nb, const cam_t& cam, size_t screen_w, size_t screen_h)
{
#ifdef BENCH
    BEGIN_BENCH();
#endif

    cuda_project_points<<<mesh_nb / 1024 + 1, 1024>>>(meshes_d, mesh_nb, cam, screen_w, screen_h);

#ifdef BENCH
    END_BENCH("projection kernel");
#endif
}

__global__ void cuda_bounding_boxes(mesh_t* meshes, size_t mesh_nb, bbox_t* bboxes)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < mesh_nb)
    {
        auto mesh = meshes[idx];

        point_t top_l{-1u,  -1u};
        point_t top_r{-1,   -1u};
        point_t bot_l{-1u,  -1};
        point_t bot_r{-1,   -1};
        point_t points[] = { mesh.v1.pos, mesh.v2.pos, mesh.v3.pos };

        for (const auto& p : points)
        {
            top_l.x = p.x < top_l.x ? p.x : top_l.x;
            top_l.y = p.y < top_l.y ? p.y : top_l.y;

            top_r.x = p.x > top_r.x ? p.x : top_r.x;
            top_r.y = p.y < top_r.y ? p.y : top_r.y;

            bot_l.x = p.x < bot_l.x ? p.x : bot_l.x;
            bot_l.y = p.y > bot_l.y ? p.y : bot_l.y;

            bot_r.x = p.x > bot_r.x ? p.x : bot_r.x;
            bot_r.y = p.y > bot_r.y ? p.y : bot_r.y;
        }

        bboxes[idx] = {top_l, top_r, bot_l, bot_r};
    }
}

__global__ void cuda_tiles_dispatch(size_t mesh_nb, bbox_t* bboxes,
        dim3 tiles_dim, device_vec_t* vecs)
{
    size_t idx = blockIdx.x;

    if (idx < mesh_nb)
    {
        /* Compute distance in terms of tiles (i.e 1 tile, 2 tiles, 3 tiles...) */
        bbox_t bbox = bboxes[idx];
        size_t h_dst = ceil(bbox.top_r.x / 32) - bbox.top_l.x / 32;
        size_t v_dst = ceil(bbox.bot_l.y / 32) - bbox.top_l.y / 32;

        /* Look for top left bounding box */
        size_t start_w = bbox.top_l.x / 32;
        size_t start_h = bbox.top_l.y / 32;

        /* For each tile in the bounding box, notify the bitset that the
           current mesh is inside. */
        for (size_t i = 0; i <= v_dst; ++i)
        {
            for (size_t j = 0; j <= h_dst; ++j)
            {
                size_t off = (start_h + i) * tiles_dim.x + start_w + j;
                if (off < tiles_dim.x * tiles_dim.y)
                    vecs[off].push(idx);
            }
        }
    }
}

device_vec_t* tiles_dispatch_kernel(mesh_t* meshes_d, size_t mesh_nb,
        size_t screen_w, size_t screen_h)
{
    size_t tpb = 1024;
    bbox_t* bboxes_d;

#ifdef BENCH
    BEGIN_BENCH()
#endif

    cudaMalloc(&bboxes_d, sizeof(bbox_t) * mesh_nb);

    cuda_bounding_boxes<<<mesh_nb / tpb + 1, tpb>>>(meshes_d, mesh_nb,
            bboxes_d);

    auto tiles_dim = compute_tiles(screen_w, screen_h);
    auto tiles_size = tiles_dim.x * tiles_dim.y;

    device_vec_t* vecs = new device_vec_t[tiles_size];
    device_vec_t* vecs_d;

    cudaMalloc(&vecs_d, sizeof(device_vec_t) * tiles_size);
    cudaMemcpy(vecs_d, vecs, sizeof(device_vec_t) * tiles_size, cudaMemcpyHostToDevice);

    cuda_tiles_dispatch<<<mesh_nb, 1>>>(mesh_nb, bboxes_d,
            tiles_dim, vecs_d);

    cudaFree(bboxes_d);

#ifdef BENCH
    END_BENCH("tile dispatch kernel");
#endif

    return vecs_d;
}

static inline
__device__ color_t interpolate_tex(const mesh_t& mesh, double weights[3],
        dev_mat_t* mats, size_t mats_nb)
{
    point_t p0{ mesh.v1.tex.x * weights[0], mesh.v1.tex.y * weights[0]};
    point_t p1{ mesh.v2.tex.x * weights[1], mesh.v2.tex.y * weights[1]};
    point_t p2{ mesh.v3.tex.x * weights[2], mesh.v3.tex.y * weights[2]};

    dev_mat_t& mat = mats[mesh.mat_idx];

    double u = p0.x + p1.x + p2.x;
    double v = p0.y + p1.y + p2.y;

    /* Interpolation, here simply getting nearest neighbourg */
    while (u < 0.)
        u += 1.;

    while (u > 1.)
        u -= 1.;

    while (v < 0.)
        v += 1.;

    while (v > 1.)
        v -= 1.;

    size_t x = u * (double)mat.tex_w;
    size_t y = v * (double)mat.tex_h;

    auto vec = mat.kd_map[mat.tex_w * y + x];

    //printf("vec.x = %f | vec.y = %f | vec.z = %f\n", vec.x, vec.y, vec.z);
    return color_t{vec.x, vec.y, vec.z};

}

__global__ void cuda_draw_mesh(mesh_t* meshes, size_t mesh_nb, color_t* screen,
        size_t screen_w, size_t screen_h, device_vec_t* vecs, dev_mat_t* mats,
        size_t mats_nb)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t idx = y * screen_w + x;
    size_t screen_size = screen_w * screen_h;
    size_t block_idx = blockIdx.y * gridDim.x + blockIdx.x;

    if (idx >= screen_size)
        return;

    point_t p = {x, y, 0};
    double best_z = -1u; //set to max int

    auto* vec = vecs + block_idx;

    for (size_t i = 0; i < vec->size(); ++i)
    {
        mesh_t mesh = meshes[(*vec)[i]];
        point_t p1 = mesh.v1.pos;
        point_t p2 = mesh.v2.pos;
        point_t p3 = mesh.v3.pos;

        if (check_edges(p1, p2, p3, p))
        {
            volatile double area = edge_function(p1, p2, p3);
            volatile double w0 = edge_function(p2, p3, p) / area;
            volatile double w1 = edge_function(p3, p1, p) / area;
            volatile double w2 = edge_function(p1, p2, p) / area;

            p1.z = 1.0 / p1.z;
            p2.z = 1.0 / p2.z;
            p3.z = 1.0 / p3.z;

            volatile double z_invert = p1.z * w0 + p2.z * w1 + p3.z * w2;
            volatile double z = 1.0 / z_invert;

            if (z < best_z)
            {
                best_z = z;

                double w[] = { w0, w1, w2 };
                //interpolate_tex(mesh, w, mats, mats_nb);
                //screen[idx] = {w0 * 255.0, w1 * 255.0, w2 * 255.0};
                screen[idx] = interpolate_tex(mesh, w, mats, mats_nb);
            }
        }
    }
}

void draw_mesh_kernel(host_vec_t<color_t>& screen, size_t screen_w, size_t screen_h,
        mesh_t* meshes_d, size_t mesh_nb, device_vec_t* vecs, dev_mat_t* mats,
        size_t mats_nb)
{
    color_t* screen_d = nullptr;

#ifdef BENCH
    BEGIN_BENCH();
#endif

    cudaMalloc(&screen_d, sizeof(color_t) * screen.size());
    cudaMemset(screen_d, 0, sizeof(color_t) * screen.size());

    auto blockDim = dim3(32, 32);
    auto gridDim = dim3(std::ceil(screen_w / blockDim.x), std::ceil(screen_h / blockDim.y));

    cuda_draw_mesh<<<gridDim, blockDim>>>(meshes_d, mesh_nb, screen_d, screen_w,
            screen_h, vecs, mats, mats_nb);

    cudaMemcpy(screen.data(), screen_d, sizeof(color_t) * screen.size(), cudaMemcpyDeviceToHost);
    cudaFree(screen_d);

#ifdef BENCH
    END_BENCH("draw mesh kernel");
#endif
}

__host__ dim3 compute_tiles(size_t screen_w, size_t screen_h)
{
    cudaDeviceProp dev_prop;

    cudaGetDeviceProperties(&dev_prop, 0);

    auto max_tpb = dev_prop.maxThreadsPerBlock;

    unsigned tiles_w = screen_w / sqrt(max_tpb);
    unsigned tiles_h = screen_h / sqrt(max_tpb);

    return { tiles_w, tiles_h };
}
