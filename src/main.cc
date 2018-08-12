#include <iostream>

#include "rasterizer.hh"
#include "input_parser.hh"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " 'path/to/obj'" << std::endl;
        return 1;
    }
    Rasterizer r;
    r.load_scene(argv[1]);
    r.compute();
    r.write_scene("output/out.ppm");
    return 0;
}
