#pragma once
#include <vector>
#include "net.h"

struct OptimizerParameters {
  std::vector<std::vector<float>>& weights;
  std::vector<std::vector<float>>& gradWeights;
  std::vector<float>&              bias;
  std::vector<float>&              gradBias;
};

struct MLPOptimizer {
  virtual void initialize(OptimizerCreateInfo ci)     = 0;
  virtual void update(OptimizerParameters parameters) = 0;
};


struct SGDOptimizer : public MLPOptimizer {
  float learningRate;

  virtual void initialize(OptimizerCreateInfo ci) override { learningRate = ci.learningRate; }
  virtual void update(OptimizerParameters layer) override {
    for (int i = 0; i < layer.weights.size(); ++i) {
      for (int j = 0; j < layer.weights[i].size(); ++j) {
        layer.weights[i][j] -= learningRate * layer.gradWeights[i][j];
      }
      layer.bias[i] -= learningRate * layer.gradBias[i];
    }
  }
  virtual ~SGDOptimizer() {}
};

struct SGDMomentum : public MLPOptimizer {
  float                           learningRate;
  float                           momentum;
  std::vector<std::vector<float>> velocityWeights;
  std::vector<float>              velocityBias;

  virtual void initialize(OptimizerCreateInfo ci) override {
    learningRate = ci.learningRate;
    momentum     = ci.momentum;

    //TODO: Fix this
  }

  virtual void update(OptimizerParameters layers) override {
    for (int i = 0; i < layers.weights.size(); ++i) {
      for (int j = 0; j < layers.weights[i].size(); ++j) {
        velocityWeights[i][j] = momentum * velocityWeights[i][j] - learningRate * layers.gradWeights[i][j];
        layers.weights[i][j] += velocityWeights[i][j];
      }
      velocityBias[i] = momentum * velocityBias[i] - learningRate * layers.gradBias[i];
      layers.bias[i] += velocityBias[i];
    }
  }

  virtual ~SGDMomentum() {}
};

MLPOptimizer* mlpOptimzerCreate(OptimizerCreateInfo ci) {
  switch (ci.function) {
    default:
    case MLP_OPTIMIZER_SGD: return new SGDOptimizer;
    case MLP_OPTIMIZER_SGD_MOMENTUM: return new SGDMomentum;
  }
}
