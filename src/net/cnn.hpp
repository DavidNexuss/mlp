#pragma once
#include "net.h"

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

struct CNN : public NetWork {
  virtual void AddLayer(CNNLayerCreateInfo ci) = 0;
};
