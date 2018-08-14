#include <cmath>
#include <fstream>
#include <limits>

#ifdef GPU
 #include "gpu_operations.hh"
#endif

#include "input_parser.hh"
#include "rasterizer.hh"
#include "utils.hh"

void Rasterizer::load_scene(const std::string& filename)
{
    InputParser parser(filename);
    meshes_ = parser.meshes_get();
    z_buffer_ = std::vector<double>(screen_w_ * screen_h_,
            std::numeric_limits<double>::max());
    screen_ = std::vector<color_t>(screen_w_ * screen_h_);
}

void Rasterizer::write_scene(const std::string& filename) const
{
  std::ofstream ofs(filename);

  ofs << "P3\n" << screen_w_ << ' ' << screen_h_ << "\n255\n";

  for (size_t i = 0; i < screen_h_; ++i)
  {
    for (size_t j = 0; j < screen_w_ ; ++j)
    {
        const auto& pix = screen_[i * screen_w_ + j];
        int r = pix.r;
        int g = pix.g;
        int b = pix.b;

        ofs << r << ' ' << g << ' ' << b;

        if (j < screen_w_ - 1)
            ofs << "  ";
    }
    ofs << '\n';
  }
}

void Rasterizer::compute()
{
#ifndef GPU
    project_scene();
#else
    gpu_project_scene();
#endif

    for (auto& mesh : meshes_)
    {
        auto color = random_color();
        for (size_t i = 0; i < screen_h_; ++i)
        {
            for (size_t j = 0; j < screen_w_; ++j)
            {
                //TODO: bounding boxes

                point_t p = {j, i, 0};

                auto vertices = mesh.vertices;
                bool edge_check = check_edges(vertices[0].pos, vertices[1].pos,
                        vertices[2].pos, p);

                /*
                for (size_t i = 0; i < vertices.size(); ++i)
                {
                    edge_check &= edge_function(vertices[i].pos,
                            vertices[(i + 1) % vertices.size()].pos, p) >= 0;
                }
                */

                if (edge_check)
                {
                    std::vector<double> weights;
                    double area = edge_function(vertices[0].pos,
                            vertices[1].pos, vertices[2].pos);

                    weights.push_back(edge_function(vertices[1].pos,
                                vertices[2].pos, p) / area);
                    weights.push_back(edge_function(vertices[2].pos,
                                vertices[0].pos, p) / area);
                    weights.push_back(edge_function(vertices[0].pos,
                                vertices[1].pos, p) / area);

                    for (auto& v : vertices)
                        v.pos.z = 1.0 / v.pos.z;

                    double z_invert = 0;
                    for (size_t i = 0; i < vertices.size(); ++i)
                        z_invert += vertices[i].pos.z * weights[i];

                    double z = 1.0 / z_invert;

                    if (z < z_buffer_[i * screen_w_ + j])
                    {
                        auto& pix = screen_[i * screen_w_ + j];

                        z_buffer_[i * screen_w_ + j] = z;

                        pix.r = color.r * 255.0;
                        pix.g = color.g * 255.0;
                        pix.b = color.b * 255.0;
                    }
                }
            }
        }
    }
}

void Rasterizer::project_scene()
{
    /* World to camera */
    cam_project();

    /* Camera to screen */
    screen_project();

    /* Screen to NDC */
    ndc_project(0, screen_w_, 0, screen_h_);

    /* NDC to raster */
    raster_project();
}

void Rasterizer::cam_project()
{
    for (auto& mesh : meshes_)
    {
        for (auto& v : mesh.vertices)
            cam_project_point(v.pos);
    }
}

void Rasterizer::screen_project()
{
    for (auto& mesh : meshes_)
    {
        for (auto& v : mesh.vertices)
        {
            constexpr double ncp = 1;

            v.pos.x = ncp * (screen_w_ / 2.0) * v.pos.x / -v.pos.z + screen_w_ / 2.0;
            v.pos.y = ncp * (screen_h_ / 2.0) * v.pos.y / -v.pos.z + screen_h_ / 2.0;
            v.pos.z = -v.pos.z;
        }
    }
}

void Rasterizer::ndc_project(double l, double r, double b, double t)
{
    for (auto& mesh : meshes_)
    {
        for (auto& v : mesh.vertices)
        {
            v.pos.x = 2 * v.pos.x / (r - l) - (r + l) / (r - l);
            v.pos.y = 2 * v.pos.y / (t - b) - (t + b) / (t - b);

        }
    }
}

void Rasterizer::raster_project()
{
    for (auto& mesh : meshes_)
    {
        for (auto& v : mesh.vertices)
        {
            v.pos.x = (v.pos.x + 1) / 2 * screen_w_;
            v.pos.y = (1 - v.pos.y) / 2 * screen_h_;
        }
    }
}

void Rasterizer::cam_project_point(point_t& p)
{
    double rot_mat[] =
    {
        cam_.dir_x.x, cam_.dir_x.y, cam_.dir_x.z,
        cam_.dir_y.x, cam_.dir_y.y, cam_.dir_y.z,
        cam_.dir_z.x, cam_.dir_z.y, cam_.dir_z.z
    };

    p -= cam_.pos;

    double trans_mat[] = { p.x, p.y, p.z };
    double out_mat[3] = { 0 };

    mat_mult<double, 3>(rot_mat, trans_mat, out_mat);

    p.x = out_mat[0];
    p.y = out_mat[1];
    p.z = out_mat[2];
}

#ifdef GPU
void Rasterizer::gpu_project_scene()
{
    /* Accumulate points considering we only deal with triangles */
    size_t points_nb = meshes_.size() * 3;
    point_t* points = new point_t[points_nb];
    size_t i = 0;

    for (const auto& mesh : meshes_)
    {
        for (const auto& vertex : mesh.vertices)
            points[i++] = vertex.pos;
    }

    /* Call to cuda projection */
    projection_kernel(points, points_nb, cam_, screen_w_, screen_h_);

    i = 0;
    for (auto& mesh : meshes_)
    {
        for (auto& vertex : mesh.vertices)
            vertex.pos = points[i++];
    }

    delete[] points;
}
#endif
