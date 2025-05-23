#include <vector>

template <typename T>
struct Tensor : public std::vector<T> {
  int dimensions[10];

  T& at(int x) {
    std::vector<T>& a = *this;
    return a[x];
  }

  const T& at(int x) const {
    std::vector<T>& a = *this;
    return a[x];
  }

  T& at2D(int x, int y) {
    std::vector<T>& a = *this;
    return a[dimensions[0] * x + y];
  }

  const T& at2D(int x, int y) const {
    std::vector<T>& a = *this;
    return a[dimensions[0] * x + y];
  }

  T& at3D(int x, int y, int z) {
    std::vector<T>& a = *this;
    return a[dimensions[1] * (dimensions[0] * x + y) + z];
  }

  const T& at3D(int x, int y, int z) const {
    std::vector<T>& a = *this;
    return a[dimensions[1] * (dimensions[0] * x + y) * z];
  }

  int width() { return dimensions[0]; }
  int height() { return dimensions[1]; }
};
