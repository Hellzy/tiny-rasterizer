#pragma once
#include "scene.hh"

class Rasterizer
{
public:
    /**
     * Opens a .obj file to load the scene into the rasterizer
     * - filename: name of the .obj file to open
     */
    void load_scene(const std::string& filename);

    /**
     * Writes the rasterized scene to filename
     * - filename: name of the file to write the output to
     */
    void write_scene(const std::string& filename) const;

    /**
     * Perorms the rasterization of the last loaded scene
     */
    void compute();

private:
    Scene scene_;
};
