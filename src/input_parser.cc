#include <fstream>
#include <ios>

#include "input_parser.hh"

InputParser::InputParser()
{
    parse_cfg();
}

/* This funciotn only handles triangles for now */
InputParser::InputParser(const std::string& filename)
{
    parse_cfg();
    load(filename);
}

void InputParser::load(const std::string& filename)
{
    OBJParser parser(filename);

    vertices_ = parser.vertices_get();
    meshes_ = parser.meshes_get();
    mats_ = parser.mats_get();
}

void InputParser::parse_cfg()
{
    std::ifstream ifs(".config");

    if (!ifs.good())
        throw std::ios_base::failure("Could not open .config");

    ifs >> screen_w_ >> screen_h_;
    ifs >> cam_.pos.x >> cam_.pos.y >> cam_.pos.z;
    ifs >> cam_.dir_x.x >> cam_.dir_x.y >> cam_.dir_x.z;
    ifs >> cam_.dir_y.x >> cam_.dir_y.y >> cam_.dir_y.z;
    ifs >> cam_.dir_z.x >> cam_.dir_z.y >> cam_.dir_z.z;
}
