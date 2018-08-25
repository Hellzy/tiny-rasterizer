#include <chrono>
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

    auto start = std::chrono::system_clock::now();
    r.gpu_compute();
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed total: " << elapsed.count() << "s\n";

    r.write_scene("output/out.ppm");
    return 0;
}
