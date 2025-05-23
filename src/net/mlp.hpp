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

struct MLP : public NetWork {
  virtual void AddLayer(int neurons, ActivationFunction function, InitializationStrategy strategy = MLP_INITIALIZE_NONE) = 0;
  virtual void SetOptimizer(const OptimizerCreateInfo ci)                                                                = 0;
  virtual void InitializeLayer(InitializationStrategy strategy, int layerIndex)                                          = 0;
  virtual void Initialize(InitializationStrategy strategy)                                                               = 0;
  virtual void Print()                                                                                                   = 0;
  virtual ~MLP() {}
};

MLP* mlpCreate(int inputLayerSize);
MLP* mlpCreate(MLPCreateInfo ci);
