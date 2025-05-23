#include <vector>

template <typename T>
struct Tensor : public std::vector<T> {
  int dimensions[10];

  T& at(int x) {
    return (*std::vector<T>)[x];
  }
  T& at2D(int x, int y) {
    return data[dimensions[0] * x + y];
  }
  T& at3D(int x, int y, int z) {
    return data[dimensions[1] * (dimensions[0] * x + y) + z];
  }
};
