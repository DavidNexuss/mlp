#pragma once
#include "net.h"

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
  virtual void  Print()                                                                                                   = 0;
  virtual ~MLP() {}
};

MLP* mlpCreate(int inputLayerSize);
MLP* mlpCreate(MLPCreateInfo ci);
