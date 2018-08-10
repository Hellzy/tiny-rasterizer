#include "object.hh"
#include "triangle.hh"
#include "rasterizer.hh"
#include "input_parser.hh"

int main()
{
    Rasterizer r;
    r.load_scene("input/cube.in");
    r.compute();
    r.write_scene("output/cube.ppm");
    return 0;
}
