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

__global__ void cuda_project_points(point_t* points, size_t point_nb, cam_t cam, size_t screen_w, size_t screen_h)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < point_nb)
    {
        constexpr double ncp = 1;
        point_t p = points[idx];

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

        points[idx] = p;
    }
}

void projection_kernel(point_t* points, size_t point_nb, const cam_t& cam, size_t screen_w, size_t screen_h)
{
    point_t* points_d;

    cudaMalloc(&points_d, sizeof(point_t) * point_nb);
    cudaMemcpy(points_d, points, sizeof(point_t) * point_nb,
            cudaMemcpyHostToDevice);
    cuda_project_points<<<point_nb / 1024 + 1, 1024>>>(points_d, point_nb, cam, screen_w, screen_h);
    cudaMemcpy(points, points_d, sizeof(point_t) * point_nb,
            cudaMemcpyDeviceToHost);
    cudaFree(points_d);
}
