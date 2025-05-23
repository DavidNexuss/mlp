#pragma once
#include <containers/tensor.hpp>

struct ivec {
  int x, y, z;
};

struct CNNLayerCreateInfo {
  ivec size;
  ivec kernelSize;
  int  filters;
  int  stride;
  int  offset;
};

struct CNNCreateInfo {
  ivec inputSize;
  ivec outputSize;
};

struct CNN {
  virtual void AddLayer(CNNLayerCreateInfo ci)                        = 0;
  virtual void Propagate(Tensor<float>& input, Tensor<float>& output) = 0;
};
