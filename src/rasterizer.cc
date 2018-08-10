#include <cmath>
#include <fstream>
#include <limits>

#include "input_parser.hh"
#include "rasterizer.hh"
#include "triangle.hh"
#include "utils.hh"

void Rasterizer::load_scene(const std::string& filename)
{
    InputParser parser(filename);
    scene_ = parser.scene_get();
    z_buffer = std::vector<double>(scene_.width * scene_.height,
            std::numeric_limits<double>::max());
}

void Rasterizer::write_scene(const std::string& filename) const
{
  std::ofstream ofs(filename);

  ofs << "P3\n" << scene_.width << ' ' << scene_.height << "\n255\n";

  for (size_t i = 0; i < scene_.height; ++i)
  {
    for (size_t j = 0; j < scene_.width; ++j)
    {
        const auto& pix = scene_.screen[i * scene_.width + j];
        int r = pix.r;
        int g = pix.g;
        int b = pix.b;

        ofs << r << ' ' << g << ' ' << b;

        if (j < scene_.width - 1)
            ofs << "  ";
    }
    ofs << '\n';
  }
}

void Rasterizer::compute()
{
    project_scene();

    for (auto obj : scene_.objects)
    {
        for (size_t i = 0; i < scene_.height; ++i)
        {
            for (size_t j = 0; j < scene_.width; ++j)
            {
                //FIXME: remove object abstraction, only work with triangle
                //to trim a lot of useless code

                point_t p = {j, i, 0};

                if (obj->check_edges(p))
                {
                    std::vector<double> weights;
                    auto points = obj->get_points();
                    double area = edge_function(points[0], points[1], points[2]);

                    weights.push_back(edge_function(points[1], points[2], p) / area);
                    weights.push_back(edge_function(points[2], points[0], p) / area);
                    weights.push_back(edge_function(points[0], points[1], p) / area);

                    for (auto& p : points)
                        p.z = 1.0 / p.z;

                    double z_invert = 0;
                    for (size_t i = 0; i < points.size(); ++i)
                        z_invert += points[i].z * weights[i];

                    double z = 1.0 / z_invert;

                    if (z < z_buffer[i * scene_.width + j])
                    {
                        auto& pix = scene_.screen[i * scene_.width + j];

                        z_buffer[i * scene_.width + j] = z;

                        pix.r = obj->col.r * 255.0;
                        pix.g = obj->col.g * 255.0;
                        pix.b = obj->col.b * 255.0;
                    }
                }
            }
        }
    }
}

void Rasterizer::project_scene()
{
    for (auto obj : scene_.objects)
    {
        /* World to camera */
        obj->cam_project(scene_.eye);

        /* Camera to screen */
        obj->screen_project(scene_.width, scene_.height);

        /* Screen to NDC */
        obj->ndc_project(0, scene_.width, 0, scene_.height);

        /* NDC to raster */
        obj->raster_project(scene_.width, scene_.height);
    }
}
