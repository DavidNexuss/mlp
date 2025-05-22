#pragma once
#include <vector>
#include <math.h>
#include "net.h"

inline float computeLoss(const std::vector<float>& predicted, const std::vector<float>& target, LossFunction lossFn) {
  switch (lossFn) {
    case MLP_LOSS_MSE: {
      float sum = 0.0f;
      for (int i = 0; i < predicted.size(); i++) {
        float diff = predicted[i] - target[i];
        sum += diff * diff;
      }
      return sum / predicted.size();
    }
    case MLP_LOSS_CROSS_ENTROPY: {
      float loss = 0.0f;
      for (int i = 0; i < predicted.size(); i++) {
        loss -= target[i] * logf(predicted[i] + 1e-7f);
      }
      return loss / predicted.size();
    }
    default:
      return 0.0f;
  }
}

inline std::vector<float> computeLossDerivative(const std::vector<float>& output, const std::vector<float>& target, LossFunction loss) {
  std::vector<float> delta(output.size());

  switch (loss) {
    case MLP_LOSS_MSE: // Mean Squared Error: dL/dy = (y - t)
      for (size_t i = 0; i < output.size(); ++i) {
        delta[i] = output[i] - target[i];
      }
      break;

    case MLP_LOSS_CROSS_ENTROPY:
      // Assumes softmax + cross-entropy; derivative simplifies to: dL/dy = y - t
      for (size_t i = 0; i < output.size(); ++i) {
        delta[i] = output[i] - target[i];
      }
      break;

    default:
      // Fallback: zero gradient
      for (float& d : delta) d = 0.0f;
      break;
  }

  return delta;
}
