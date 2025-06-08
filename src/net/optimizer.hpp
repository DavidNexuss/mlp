#pragma once
#include <vector>
#include "net.h"

//Parameters of ONE layer of the Network
struct OptimizerUpdateParameters {
  std::vector<std::vector<float>>& weights;
  std::vector<std::vector<float>>& gradWeights;
  std::vector<float>&              bias;
  std::vector<float>&              gradBias;
};

struct OptimizerInputParameters {
  int inputNeurons;
  int outputNeurons;
};

struct MLPOptimizer {
  virtual void initialize(OptimizerInputParameters input) = 0;
  virtual void update(OptimizerUpdateParameters update)   = 0;
};


struct SGDOptimizer : public MLPOptimizer {
  float learningRate;

  SGDOptimizer(OptimizerCreateInfo ci) { learningRate = ci.learningRate; }

  virtual void initialize(OptimizerInputParameters ci) override {}
  virtual void update(OptimizerUpdateParameters layer) override {
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
  float                           lambda;
  std::vector<std::vector<float>> velocityWeights;
  std::vector<float>              velocityBias;

  SGDMomentum(OptimizerCreateInfo ci) {
    learningRate = ci.learningRate;
    momentum     = ci.momentum;
    lambda       = ci.l2;
  }

  virtual void initialize(OptimizerInputParameters ci) override {
    velocityWeights.resize(ci.outputNeurons, std::vector<float>(ci.inputNeurons));
    velocityBias.resize(ci.outputNeurons);
  }

  virtual void update(OptimizerUpdateParameters layer) override {
    for (int i = 0; i < layer.weights.size(); ++i) {
      for (int j = 0; j < layer.weights[i].size(); ++j) {

        float grad = layer.gradWeights[i][j] + lambda * layer.weights[i][j];

        velocityWeights[i][j] = momentum * velocityWeights[i][j] - learningRate * grad;
        layer.weights[i][j] += velocityWeights[i][j];
      }
      velocityBias[i] = momentum * velocityBias[i] - learningRate * layer.gradBias[i];
      layer.bias[i] += velocityBias[i];
    }
  }

  virtual ~SGDMomentum() {}
};

struct Adam : public MLPOptimizer {
  Adam(OptimizerCreateInfo ci) {}

  virtual void initialize(OptimizerInputParameters ci) override {
  }

  virtual void update(OptimizerUpdateParameters layer) override {
  }

  virtual ~Adam() {}
};


inline MLPOptimizer* mlpOptimzerCreate(OptimizerCreateInfo ci) {
  switch (ci.function) {
    default:
    case MLP_OPTIMIZER_SGD: return new SGDOptimizer(ci);
    case MLP_OPTIMIZER_SGD_MOMENTUM: return new SGDMomentum(ci);
    case MLP_OPTIMIZER_ADAM: return new Adam(ci);
  }
}
