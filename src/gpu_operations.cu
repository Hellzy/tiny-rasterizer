#ifdef BENCH
#include <iostream>
#include <chrono>
#endif

#include <cmath>

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

void projection_kernel(mesh_t* meshes, size_t mesh_nb, const cam_t& cam, size_t screen_w, size_t screen_h)
{
    mesh_t* meshes_d;

    cudaMalloc(&meshes_d, sizeof(mesh_t) * mesh_nb);
    cudaMemcpy(meshes_d, meshes, sizeof(mesh_t) * mesh_nb,
            cudaMemcpyHostToDevice);

#ifdef BENCH
    auto gpu_start = std::chrono::system_clock::now();
#endif

    cuda_project_points<<<mesh_nb / 1024 + 1, 1024>>>(meshes_d, mesh_nb, cam, screen_w, screen_h);

#ifdef BENCH
    auto gpu_end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_gpu = gpu_end - gpu_start;
    std::cout << "Elapsed projection_kernel: " << elapsed_gpu.count() << "s\n";
#endif

    cudaMemcpy(meshes, meshes_d, sizeof(mesh_t) * mesh_nb,
            cudaMemcpyDeviceToHost);
    cudaFree(meshes_d);
}

__global__ void cuda_draw_mesh(point_t p1, point_t p2, point_t p3,
        double* z_buffer, color_t* screen, size_t screen_w, size_t screen_h)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t idx = y * screen_w + x;
    size_t screen_size = screen_w * screen_h;

    if (idx >= screen_size)
        return;

    point_t p = {x, y, 0};

    bool edge_check = check_edges(p1, p2, p3, p);

    if (edge_check)
    {
        double area = edge_function(p1, p2, p3);
        double w0 = edge_function(p2, p3, p) / area;
        double w1 = edge_function(p3, p1, p) / area;
        double w2 = edge_function(p1, p2, p) / area;

        p1.z = 1.0 / p1.z;
        p2.z = 1.0 / p2.z;
        p3.z = 1.0 / p3.z;

        double z_invert = p1.z * w0 + p2.z * w1 + p3.z * w2;
        double z = 1.0 / z_invert;

        if (z < z_buffer[idx])
        {
            z_buffer[idx] = z;

            color_t color = { 1.0, 0.0, 0.0 };
            screen[idx] = {color.r * 255.0, color.g * 255.0, color.b * 255.0};
        }
    }
}

void draw_mesh_kernel(std::vector<color_t>& screen, size_t screen_h, size_t screen_w,
        const std::vector<mesh_t>& meshes, const std::vector<double>& z_buffer)
{
    double* z_buffer_d = nullptr;
    color_t* screen_d = nullptr;

    cudaMalloc(&z_buffer_d, sizeof(double) * z_buffer.size());
    cudaMemcpy(z_buffer_d, z_buffer.data(), sizeof(double) * z_buffer.size(),
            cudaMemcpyHostToDevice);
    cudaMalloc(&screen_d, sizeof(color_t) * screen.size());
    cudaMemset(screen_d, 0, sizeof(color_t) * screen.size());

#ifdef BENCH
    auto gpu_start = std::chrono::system_clock::now();
#endif

    for (const auto& mesh : meshes)
    {
        auto v1 = mesh.v1;
        auto v2 = mesh.v2;
        auto v3 = mesh.v3;

        auto blockDim = dim3(32, 32);
        auto gridDim = dim3(std::ceil(screen_w / blockDim.x),
                std::ceil(screen_h / blockDim.y));

        cuda_draw_mesh<<<gridDim, blockDim>>>(v1.pos, v2.pos, v3.pos,
                z_buffer_d, screen_d, screen_h, screen_w);
    }

#ifdef BENCH
    auto gpu_end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_gpu = gpu_end - gpu_start;
    std::cout << "Elapsed draw_mesh_kernel: " << elapsed_gpu.count() << "s\n";
#endif

    cudaMemcpy(screen.data(), screen_d, sizeof(color_t) * screen.size(), cudaMemcpyDeviceToHost);
    cudaFree(screen_d);
    cudaFree(z_buffer_d);
}
