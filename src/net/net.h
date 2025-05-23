#pragma once
#include <vector>
#include <containers/tensor.hpp>

using vector = Tensor<float>;

//Activations functions
enum ActivationFunction {
  MLP_ACTIVATION_AFFINE = 0, // f(x) = ax + b
  MLP_ACTIVATION_COSINE,     // f(x) = cos(x)
  MLP_ACTIVATION_SIGMOID,    // f(x) = 1 / (1 + exp(-x))
  MLP_ACTIVATION_TANH,       // f(x) = tanh(x)
  MLP_ACTIVATION_RELU,       // f(x) = max(0, x)
  MLP_ACTIVATION_LEAKY_RELU,
  MLP_ACTIVATION_SOFTMAX
};

enum InitializationStrategy {
  MLP_INITIALIZE_NONE = 0,
  MLP_INITIALIZE_RANDOM,
  MLP_INITIALIZE_xAVIER,
  MLP_INITIALIZE_HE,
  MLP_INITIALIZE_AUTO
};


// Loss functions
enum LossFunction {
  MLP_LOSS_MSE = 0,       // Mean Squared Error
  MLP_LOSS_CROSS_ENTROPY, // Cross Entropy Loss
};

// Optimizer functions
enum OptimizerFunction {
  MLP_OPTIMIZER_SGD = 0,      // Stochastic Gradient Descent
  MLP_OPTIMIZER_SGD_MOMENTUM, // SGD with Momentum
  MLP_OPTIMIZER_ADAM,         // Adaptive Moment Estimation
  MLP_OPTIMIZER_RMSPROP,      // RMSProp optimizer
};

struct OptimizerCreateInfo {
  OptimizerFunction function;

  //Hyper parameters
  float learningRate = 0.1f;
  float momentum     = 0.9f;
  float l2lambda     = 0.0f;

  void print();
};

struct DataSet {
  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<float>> targets;
};

struct NetWork {
  virtual void  Propagate(const vector& input, vector& output)                                = 0;
  virtual float ComputeLoss(const vector& predicted, const vector& target, LossFunction loss) = 0;
  virtual void  Backpropagate(const vector& input, const vector& target, LossFunction loss)   = 0;
  virtual void  TrainStep(const vector& input, const vector& target, LossFunction loss)       = 0;
  virtual void  SetOptimizer(const OptimizerCreateInfo ci)                                    = 0;
};
