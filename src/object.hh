#pragma once

#include <memory>

#include "types.hh"

/**
 * Represents an object in the scene. Most primary object is a triangle
 */
struct Object
{
  Object();


  /** Color for the light, absorbtion for the primitives **/
  color_t col;
};
