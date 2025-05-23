#pragma once
#include <vector>
#include <iostream>
#include "net.h"

const char* optimizerFunctionToString(OptimizerFunction f) {
  switch (f) {
    case MLP_OPTIMIZER_SGD: return "SGD";
    case MLP_OPTIMIZER_SGD_MOMENTUM: return "SGD with Momentum";
    case MLP_OPTIMIZER_ADAM: return "Adam";
    case MLP_OPTIMIZER_RMSPROP: return "RMSProp";
    default: return "Unknown";
  }
}

void OptimizerCreateInfo::print() {
  std::cout << "=== Optimizer Configuration ===\n";
  std::cout << "Function     : " << optimizerFunctionToString(function) << "\n";
  std::cout << "LearningRate : " << learningRate << "\n";
  std::cout << "Momentum     : " << momentum << "\n";
  std::cout << "L2 Lambda    : " << l2lambda << "\n";
  std::cout << "================================\n";
}

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

struct MLPOptimizer : public OptimizerCreateInfo {
  MLPOptimizer(OptimizerCreateInfo ci) :
    OptimizerCreateInfo(ci) {}

  virtual void initialize(OptimizerInputParameters input) = 0;
  virtual void update(OptimizerUpdateParameters update)   = 0;
};

struct SGDOptimizer : public MLPOptimizer {
  SGDOptimizer(OptimizerCreateInfo ci) :
    MLPOptimizer(ci) {
  }

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
  std::vector<std::vector<float>> velocityWeights;
  std::vector<float>              velocityBias;

  SGDMomentum(OptimizerCreateInfo ci) :
    MLPOptimizer(ci) {
  }

  virtual void initialize(OptimizerInputParameters ci) override {
    velocityWeights.resize(ci.outputNeurons, std::vector<float>(ci.inputNeurons));
    velocityBias.resize(ci.outputNeurons);
  }

  virtual void update(OptimizerUpdateParameters layer) override {
    for (int i = 0; i < layer.weights.size(); ++i) {
      for (int j = 0; j < layer.weights[i].size(); ++j) {
        velocityWeights[i][j] = momentum * velocityWeights[i][j] - learningRate * (layer.gradWeights[i][j] + l2lambda * layer.weights[i][j]);
        layer.weights[i][j] += velocityWeights[i][j];
      }
      velocityBias[i] = momentum * velocityBias[i] - learningRate * layer.gradBias[i];
      layer.bias[i] += velocityBias[i];
    }
  }

  virtual ~SGDMomentum() {}
};

struct Adam : public MLPOptimizer {
  Adam(OptimizerCreateInfo ci) :
    MLPOptimizer(ci) {}

  virtual void initialize(OptimizerInputParameters ci) override {
  }

  virtual void update(OptimizerUpdateParameters layer) override {
  }

  virtual ~Adam() {}
};


MLPOptimizer* mlpOptimzerCreate(OptimizerCreateInfo ci) {

  if (ci.momentum == 0.0f && ci.function == MLP_OPTIMIZER_SGD_MOMENTUM) ci.function = MLP_OPTIMIZER_SGD;

  switch (ci.function) {
    default:
    case MLP_OPTIMIZER_SGD: return new SGDOptimizer(ci);
    case MLP_OPTIMIZER_SGD_MOMENTUM: return new SGDMomentum(ci);
    case MLP_OPTIMIZER_ADAM: return new Adam(ci);
  }
}
