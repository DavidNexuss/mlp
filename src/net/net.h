#pragma once
#include <vector>
#include <memory>

//Basics

using vector = std::vector<float>;

//Activations functions
enum ActivationFunction {
  MLP_ACTIVATION_AFFINE,  // f(x) = ax + b
  MLP_ACTIVATION_COSINE,  // f(x) = cos(x)
  MLP_ACTIVATION_SIGMOID, // f(x) = 1 / (1 + exp(-x))
  MLP_ACTIVATION_TANH,    // f(x) = tanh(x)
  MLP_ACTIVATION_RELU,    // f(x) = max(0, x)
  MLP_ACTIVATION_LEAKY_RELU,
  MLP_ACTIVATION_SOFTMAX
};

enum InitializationStrategy {
  MLP_INITIALIZE_NONE,
  MLP_INITIALIZE_RANDOM,
  MLP_INITIALIZE_xAVIER,
  MLP_INITIALIZE_HE,
  MLP_INITIALIZE_AUTO
};


// Loss functions
enum LossFunction {
  MLP_LOSS_MSE,           // Mean Squared Error
  MLP_LOSS_CROSS_ENTROPY, // Cross Entropy Loss
};

// Optimizer functions
enum OptimizerFunction {
  MLP_OPTIMIZER_SGD,          // Stochastic Gradient Descent
  MLP_OPTIMIZER_SGD_MOMENTUM, // SGD with Momentum
  MLP_OPTIMIZER_ADAM,         // Adaptive Moment Estimation
  MLP_OPTIMIZER_RMSPROP,      // RMSProp optimizer
};

struct OptimizerCreateInfo {
  float             learningRate;
  OptimizerFunction function;

  //Hyper parameters
  float momentum = 0.9f;
};


struct MLPCreateInfo {
  int                             inputSize;
  int                             outputSize;
  std::vector<int>                hiddenSizes;
  std::vector<ActivationFunction> activations;
};

// MLP
typedef struct MLP MLP;

struct MLP {
  virtual void  AddLayer(int neurons, ActivationFunction function, InitializationStrategy strategy = MLP_INITIALIZE_NONE) = 0;
  virtual void  Propagate(const vector& input, vector& output)                                                            = 0;
  virtual float ComputeLoss(const vector& predicted, const vector& target, LossFunction loss)                             = 0;
  virtual void  Backpropagate(const vector& input, const vector& target, LossFunction loss)                               = 0;
  virtual void  TrainStep(const vector& input, const vector& target, LossFunction loss)                                   = 0;
  virtual void  SetOptimizer(const OptimizerCreateInfo ci)                                                                = 0;
  virtual void  InitializeLayer(InitializationStrategy strategy, int layerIndex)                                          = 0;
  virtual void  Initialize(InitializationStrategy strategy)                                                               = 0;
  virtual ~MLP() {}
};

MLP* mlpCreate(int inputLayerSize);
MLP* mlpCreate(MLPCreateInfo ci);

struct DataSet {
  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<float>> targets;
};

///MLP Trainer
struct MLPTrainer {
  virtual void SetLossFunction(LossFunction func)      = 0;
  virtual void SetNetwork(std::shared_ptr<MLP> mlp)    = 0;
  virtual void SetDataset(std::shared_ptr<DataSet> ds) = 0;
  virtual void Train()                                 = 0;
  virtual ~MLPTrainer() {};
};

MLPTrainer* mlpTrainerCreate();
//Optimizers
