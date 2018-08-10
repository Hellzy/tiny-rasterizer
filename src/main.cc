#include "object.hh"
#include "triangle.hh"
#include "rasterizer.hh"
#include "input_parser.hh"

int main()
{
    Rasterizer r;
    r.load_scene("pyramid.in");
    r.compute();
    r.write_scene("pyramid.ppm");
    return 0;
}
