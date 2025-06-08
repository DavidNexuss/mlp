#pragma once
#include <vector>
#include <memory>
#include <random>
#include "compute.hpp"

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

struct DenseLayerCreateInfo {
  int                    neurons;
  ActivationFunction     function;
  InitializationStrategy initialization;
};

struct ConvolutionalLayerCreateInfo {
  int                    inputChannels;
  int                    inputWidth;
  int                    inputHeight;
  int                    outputChannels;
  int                    kernelSize;
  int                    stride;
  int                    padding;
  ActivationFunction     activation;
  InitializationStrategy initialization;
};

// MLP
typedef struct MLP MLP;

struct MLP {
  virtual void  AddLayer(int neurons, ActivationFunction function, InitializationStrategy strategy = MLP_INITIALIZE_NONE)                                                                                                                  = 0;
  virtual void  AddMaxPoolLayer(int inChannels, int inWidth, int inHeight, int kSize, int stride, InitializationStrategy init = MLP_INITIALIZE_NONE)                                                                                       = 0;
  virtual void  AddConvolutionalLayer(int inputChannels, int inputWidth, int inputHeight, int outputChannels, int kernelSize, int stride, int padding, ActivationFunction function, InitializationStrategy strategy = MLP_INITIALIZE_NONE) = 0;
  virtual void  Propagate(const vector& input, vector& output)                                                                                                                                                                             = 0;
  virtual float ComputeLoss(const vector& predicted, const vector& target, LossFunction loss)                                                                                                                                              = 0;
  virtual void  Backpropagate(const vector& input, const vector& target, LossFunction loss)                                                                                                                                                = 0;
  virtual void  TrainStep(const vector& input, const vector& target, LossFunction loss)                                                                                                                                                    = 0;
  virtual void  SetOptimizer(const OptimizerCreateInfo ci)                                                                                                                                                                                 = 0;
  virtual void  InitializeLayer(InitializationStrategy strategy, int layerIndex)                                                                                                                                                           = 0;
  virtual void  Initialize(InitializationStrategy strategy)                                                                                                                                                                                = 0;
  virtual ~MLP() {}
};

MLP* mlpCreate(int inputLayerSize);
MLP* mlpCreate(MLPCreateInfo ci);

MLP* mlpCreateGPU(int inputLayerSize);
MLP* mlpCreateGPU(MLPCreateInfo ci);

struct DataSet {
  virtual const std::vector<float>& getInput(int index)  = 0;
  virtual const std::vector<float>& getOutput(int index) = 0;
  virtual int                       getInputCount()      = 0;
  virtual int                       getOutputCount()     = 0;
};

//Maps all the output to the input data without making copies
struct AutoencoderDataSet : public DataSet {
  std::shared_ptr<DataSet> parent;

  AutoencoderDataSet(std::shared_ptr<DataSet> _parent) :
    parent(_parent) {}

  const std::vector<float>& getInput(int index) override { return parent->getInput(index); }
  int                       getInputCount() override { return parent->getInputCount(); }

  const std::vector<float>& getOutput(int index) override { return getInput(index); }
  int                       getOutputCount() override { return getInputCount(); }
};

inline std::shared_ptr<DataSet> makeAutoencodingDataset(std::shared_ptr<DataSet> parent) { return std::shared_ptr<DataSet>(new AutoencoderDataSet(parent)); }


template <typename T, typename V>
struct DataSetStorage : public DataSet {
  std::vector<T> inputs;
  std::vector<V> targets;

  inline int getInputCount() override { return inputs.size(); }
  inline int getOutputCount() override { return targets.size(); }

  inline void trim(int size) {
    inputs.resize(size);
    targets.resize(size);
  }

  inline void shuffle() {
    static std::random_device rd;
    static std::mt19937       g(rd());

    for (int i = (int)inputs.size() - 1; i > 0; i--) {
      std::uniform_int_distribution<int> dist(0, i);

      int j = dist(g);

      std::swap(inputs[i], inputs[j]);
      std::swap(targets[i], targets[j]);
    }
  }
};

struct ManualDataSet : public DataSetStorage<std::vector<float>, std::vector<float>> {
  inline virtual const std::vector<float>& getInput(int index) override { return inputs[index]; }
  inline virtual const std::vector<float>& getOutput(int index) override { return targets[index]; }
};


std::shared_ptr<DataSet> createStorageDataSet(const std::string& filepath);

///MLP Trainer
struct MLPTrainer {
  virtual void SetLossFunction(LossFunction func)          = 0;
  virtual void SetNetwork(std::shared_ptr<MLP> mlp)        = 0;
  virtual void SetDataset(std::shared_ptr<DataSet> ds)     = 0;
  virtual void SetTestDataset(std::shared_ptr<DataSet> ds) = 0;
  virtual void Train()                                     = 0;
  virtual ~MLPTrainer() {};
};

MLPTrainer* mlpTrainerCreate();
//Optimizers
