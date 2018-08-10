#include <random>

#include "object.hh"

Object::Object()
{
    static int count = 0;
    static color_t cur_col;

    if (!(count++ % 2))
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        cur_col = {dis(gen), dis(gen), dis(gen)};
    }

    col = cur_col;
}
