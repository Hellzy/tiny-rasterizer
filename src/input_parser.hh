#pragma once
#include <utility>

#include "scene.hh"

class InputParser
{
public:
    InputParser() = default;
    InputParser(const std::string& filename);

    void scene_load(const std::string& filename);
    Scene&& scene_get() { return std::move(scene_); }

private:
    Scene scene_;
};
