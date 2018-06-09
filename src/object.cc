#include <random>
#include "object.hh"

Object::Object()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  col = {dis(gen), dis(gen), dis(gen)};
}
