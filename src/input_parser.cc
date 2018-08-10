#include <fstream>
#include <iostream>

#include "input_parser.hh"
#include "triangle.hh"

/* This funciotn only handles triangles for now */
InputParser::InputParser(const std::string& filename)
{
    scene_load(filename);
}

void InputParser::scene_load(const std::string& filename)
{
    std::ifstream ifs(filename);

    ifs >> scene_.width >> scene_.height;
    ifs >> scene_.eye.pos.x >> scene_.eye.pos.y >> scene_.eye.pos.z;

    /* Reading camera direction */
    ifs >> scene_.eye.dir_x.x >> scene_.eye.dir_x.y >> scene_.eye.dir_x.z;
    ifs >> scene_.eye.dir_y.x >> scene_.eye.dir_y.y >> scene_.eye.dir_y.z;
    ifs >> scene_.eye.dir_z.x >> scene_.eye.dir_z.y >> scene_.eye.dir_z.z;

    while (ifs.good() && !ifs.eof())
    {
        std::string str;
        double x, y, z;
        point_t points[3];

        ifs >> str;

        if (ifs.eof())
            return;

        for (int i = 0; i < 3; ++i)
        {
            ifs >> x >> y >> z;
            points[i] = { x, y ,z };
        }

        scene_.objects.push_back(std::make_shared<Triangle>(points[0],
                    points[1], points[2]));
        scene_.screen = std::vector<color_t>(scene_.width * scene_.height);
    }
}
