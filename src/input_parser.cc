#include "input_parser.hh"
#include "parser.hh"

/* This funciotn only handles triangles for now */
InputParser::InputParser(const std::string& filename)
{
    load(filename);
}

void InputParser::load(const std::string& filename)
{
    OBJParser parser(filename);

    vertices_ = parser.vertices_get();
    meshes_ = parser.meshes_get();
}
