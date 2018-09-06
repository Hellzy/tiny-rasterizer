# Tiny rasterizer

This is a custom project I made to teach myself the therory lying behind rasterization.

# Work Done

As of now, the rasterizer is really simple. It handles triangle meshes and doesn't bother
using vertex normals or textures yet, simply rendering a flat version of the model.
Most of the algorithms run on GPU. An object composed of about 6000 meshes takes around
0,3 seconds to be rendered, which is not good enough.

# Work to do

- Improve the algorithms and GPU usage to make the program go faster.
- Handle vertex normals and textures to allow accurate rendering of the objects.
