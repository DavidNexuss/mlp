#pragma once
#include <vector>
#include "net.h"

inline static float epsilon = 1e-8;
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
        float grad            = layer.gradWeights[i][j] + lambda * layer.weights[i][j];
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
  float learningRate;
  float beta1;
  float beta2;
  float lambda;
  float epsilon;

  std::vector<std::vector<float>> mWeights;
  std::vector<std::vector<float>> vWeights;
  std::vector<float>              mBias;
  std::vector<float>              vBias;

  int timestep = 0;

  Adam(OptimizerCreateInfo ci) {
    learningRate = ci.learningRate;
    beta1        = ci.adam_beta1;
    beta2        = ci.adam_beta2;
    lambda       = ci.l2;
    epsilon      = 1e-8f;
  }

  virtual void initialize(OptimizerInputParameters ci) override {
    mWeights.assign(ci.outputNeurons, std::vector<float>(ci.inputNeurons, 0.0f));
    vWeights.assign(ci.outputNeurons, std::vector<float>(ci.inputNeurons, 0.0f));
    mBias.assign(ci.outputNeurons, 0.0f);
    vBias.assign(ci.outputNeurons, 0.0f);
    timestep = 0;
  }

  virtual void update(OptimizerUpdateParameters layer) override {
    timestep++;

    float beta1t = std::pow(beta1, timestep);
    float beta2t = std::pow(beta2, timestep);

    for (int i = 0; i < layer.weights.size(); ++i) {
      for (int j = 0; j < layer.weights[i].size(); ++j) {
        float grad = layer.gradWeights[i][j] + lambda * layer.weights[i][j];

        mWeights[i][j] = beta1 * mWeights[i][j] + (1 - beta1) * grad;
        vWeights[i][j] = beta2 * vWeights[i][j] + (1 - beta2) * grad * grad;

        float mHat = mWeights[i][j] / (1 - beta1t);
        float vHat = vWeights[i][j] / (1 - beta2t);

        layer.weights[i][j] -= learningRate * mHat / (std::sqrt(vHat) + epsilon);
      }

      float gradBias = layer.gradBias[i];

      mBias[i] = beta1 * mBias[i] + (1 - beta1) * gradBias;
      vBias[i] = beta2 * vBias[i] + (1 - beta2) * gradBias * gradBias;

      float mHatBias = mBias[i] / (1 - beta1t);
      float vHatBias = vBias[i] / (1 - beta2t);

      layer.bias[i] -= learningRate * mHatBias / (std::sqrt(vHatBias) + epsilon);
    }
  }
};


MLPOptimizer* mlpOptimzerCreate(OptimizerCreateInfo ci) {
  switch (ci.function) {
    default:
    case MLP_OPTIMIZER_SGD: return new SGDOptimizer(ci);
    case MLP_OPTIMIZER_SGD_MOMENTUM: return new SGDMomentum(ci);
    case MLP_OPTIMIZER_ADAM: return new Adam(ci);
  }
}
