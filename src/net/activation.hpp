#pragma once
#include <math.h>
#include <algorithm>
#include "net.h"
#include <cuda.h>

__host__ __device__ inline float activate(ActivationFunction f, float x) {
  switch (f) {
    case MLP_ACTIVATION_AFFINE: return x;
    case MLP_ACTIVATION_COSINE: return cosf(x);
    case MLP_ACTIVATION_SIGMOID: return 1.0f / (1.0f + expf(-x));
    case MLP_ACTIVATION_TANH: return tanhf(x);
    case MLP_ACTIVATION_RELU: return x > 0.0f ? x : 0.0f;
    case MLP_ACTIVATION_LEAKY_RELU: return x > 0.0f ? x : 0.01f * x;
    default: return x;
  }
}

__host__ __device__ inline float activation_derivative(ActivationFunction f, float x) {
  switch (f) {
    case MLP_ACTIVATION_AFFINE: return 1.0f;
    case MLP_ACTIVATION_COSINE: return -sinf(x);
    case MLP_ACTIVATION_SIGMOID: {
      float sig = 1.0f / (1.0f + expf(-x));
      return sig * (1.0f - sig);
    }
    case MLP_ACTIVATION_TANH: {
      float t = tanhf(x);
      return 1.0f - t * t;
    }
    case MLP_ACTIVATION_RELU: return x > 0.0f ? 1.0f : 0.0f;
    case MLP_ACTIVATION_LEAKY_RELU: return x > 0.0f ? 1.0f : 0.01f;
    default: return 1.0f;
  }
}

inline void softmax(std::vector<float>& x) {
  float maxVal = *std::max_element(x.begin(), x.end());
  float sum    = 0.0f;

#pragma omp simd reduction(+ : sum)
  for (int i = 0; i < x.size(); i++) {
    x[i] = expf(x[i] - maxVal);
    sum += x[i];
  }
#pragma omp simd
  for (int i = 0; i < x.size(); i++) {
    x[i] /= sum;
  }
}
