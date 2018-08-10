#pragma once

#include <memory>
#include <vector>

#include "types.hh"

/**
 * Represents an object in the scene. Most primary object is a triangle
 */
struct Object
{
  Object();

  virtual void cam_project(const cam_t& cam) = 0;
  virtual void screen_project(size_t width, size_t height) = 0;
  virtual void raster_project(size_t width, size_t height) = 0;
  virtual void ndc_project(double l, double r, double b, double t) = 0;
  virtual bool check_edges(const point_t& p) = 0;

  virtual std::vector<point_t> get_points() const = 0;

  /** Color for the light, absorbtion for the primitives **/
  color_t col;
};

using object_ptr_t = std::shared_ptr<Object>;
