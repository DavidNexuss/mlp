#include "net.h"

struct MLPGPU : public MLP {
  void  AddLayer(int neurons, ActivationFunction function, InitializationStrategy strategy = MLP_INITIALIZE_NONE) override {}
  void  AddMaxPoolLayer(int inChannels, int inWidth, int inHeight, int kSize, int stride, InitializationStrategy init = MLP_INITIALIZE_NONE) override {}
  void  AddConvolutionalLayer(int inputChannels, int inputWidth, int inputHeight, int outputChannels, int kernelSize, int stride, int padding, ActivationFunction function, InitializationStrategy strategy = MLP_INITIALIZE_NONE) override {}
  void  Propagate(const vector& input, vector& output) override {}
  float ComputeLoss(const vector& predicted, const vector& target, LossFunction loss) override {}
  void  Backpropagate(const vector& input, const vector& target, LossFunction loss) override {}
  void  TrainStep(const vector& input, const vector& target, LossFunction loss) override {}
  void  SetOptimizer(const OptimizerCreateInfo ci) override {}
  void  InitializeLayer(InitializationStrategy strategy, int layerIndex) override {}
  void  Initialize(InitializationStrategy strategy) override {}
  ~MLPGPU() {}
};
