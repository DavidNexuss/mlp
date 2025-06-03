#include <math.h>
#include <memory>
#include <algorithm>
#include <stdlib.h>
#include "net.h"
#include "activation.hpp"
#include "optimizer.hpp"
#include "loss.hpp"
#include <memory>

struct Layer {
  //Layer information
  std::vector<std::vector<float>> weights;
  std::vector<float>              bias;
  ActivationFunction              activationFunction;

  //Layer meta information
  std::vector<std::vector<float>> gradWeigths;
  std::vector<float>              gradBias;

  virtual void propagate(const std::vector<float>& input, std::vector<float>& output) = 0;

  virtual int inputSize() { return weights[0].size(); }
  virtual int outputSize() { return weights.size(); }
};


struct ConvolutionalLayer : public Layer {
  int inputChannels;
  int outputChannels;
  int kernelSize;
  int stride;
  int padding;
  int inputHeight;
  int inputWidth;

  ConvolutionalLayer(int inChannels, int inWidth, int inHeight, int outChannels, int kSize, int stride, int pad, ActivationFunction function) :
    inputChannels(inChannels), outputChannels(outChannels), kernelSize(kSize), stride(stride), padding(pad), inputWidth(inWidth), inputHeight(inHeight) {

    int kernelArea = inChannels * kSize * kSize;

    weights.resize(outChannels, std::vector<float>(kernelArea));
    gradWeigths.resize(outChannels, std::vector<float>(kernelArea));

    bias.resize(outChannels, 0.0f);
    gradBias.resize(outChannels, 0.0f);

    this->activationFunction = function;
  }

  inline int outputWidth() {
    return (inputWidth + 2 * padding - kernelSize) / stride + 1;
  }

  inline int outputHeight() {
    return (inputHeight + 2 * padding - kernelSize) / stride + 1;
  }

  virtual int inputSize() override { return inputChannels * inputWidth * inputHeight; }
  virtual int outputSize() override { return outputChannels * outputHeight() * outputWidth(); }

  // Input shape: [inputChannels][H][W] flattened into 1D vector (channel-major)
  // Output shape is computed internally and returned as a flattened 1D vector.
  void propagate(const std::vector<float>& input, std::vector<float>& output) override {
    int ow = outputWidth();
    int oh = outputHeight();
    output.resize(outputChannels * oh * ow, 0.0f);

#pragma omp parallel for collapse(3)
    for (int oc = 0; oc < outputChannels; ++oc) {
      for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
          float sum    = bias[oc];
          int   wIndex = 0;

          for (int ic = 0; ic < inputChannels; ++ic) {
            for (int ky = 0; ky < kernelSize; ++ky) {
              for (int kx = 0; kx < kernelSize; ++kx) {
                int ix = ox * stride + kx - padding;
                int iy = oy * stride + ky - padding;

                float val = 0.0f;
                if (ix >= 0 && ix < inputWidth && iy >= 0 && iy < inputHeight) {
                  val = input[ic * inputWidth * inputHeight + iy * inputWidth + ix];
                }

                sum += val * weights[oc][wIndex++];
              }
            }
          }

          int outIndex     = oc * ow * oh + oy * ow + ox;
          output[outIndex] = activate(activationFunction, sum);
        }
      }
    }

    if (activationFunction == MLP_ACTIVATION_SOFTMAX) {
      softmax(output); // Applied across spatially flattened output
    }
  }
};

struct DenseLayer : public Layer {

  DenseLayer(int input, int output, ActivationFunction function) {
    weights.resize(output, std::vector<float>(input));
    bias.resize(output, 0.0f);
    this->activationFunction = function;
  }

  void propagate(const std::vector<float>& input, std::vector<float>& output) override {
    output.resize(outputSize());

#pragma omp parallel for
    for (int i = 0; i < output.size(); i++) {
      float accum = bias[i];

#pragma omp simd reduction(+ : accum)
      for (int j = 0; j < input.size(); j++) {
        accum += input[j] * weights[i][j];
      }

      output[i] = accum;
    }

    if (activationFunction == MLP_ACTIVATION_SOFTMAX) {
      softmax(output);
    } else {

#pragma omp simd
      for (int i = 0; i < output.size(); i++) {
        output[i] = activate(activationFunction, output[i]);
      }
    }
  }
};
struct MLPImpl : public MLP {
  std::vector<std::shared_ptr<Layer>> layers;
  int                                 inputLayerSize;
  LossFunction                        loss = MLP_LOSS_MSE;

  //optimizer
  OptimizerCreateInfo                        optimizerCreateInfo;
  std::vector<std::unique_ptr<MLPOptimizer>> optimizer;
  std::vector<std::vector<float>>            layerOutputs;
  std::vector<std::vector<float>>            layerDeltas;

  MLPImpl(int inputLayerSize) {
    this->inputLayerSize = inputLayerSize;
  }

  virtual void AddLayer(int neurons, ActivationFunction function, InitializationStrategy init) override {
    int lastLayerSize = layers.size() ? layers.back()->outputSize() : inputLayerSize;
    layers.push_back(std::shared_ptr<Layer>(new DenseLayer(lastLayerSize, neurons, function)));
    if (init != MLP_INITIALIZE_NONE)
      InitializeLayer(init, layers.size() - 1);
  }

  virtual void Propagate(const vector& input, vector& output) override {
    std::vector<float> swapbuffers[2];
    output.clear();
    layerOutputs.clear();
    layerOutputs.push_back(input);

    for (int i = 0; i < layers.size(); i++) {
      const std::vector<float>& pinput  = i == 0 ? input : swapbuffers[0];
      std::vector<float>&       poutput = swapbuffers[1];

      layers[i]->propagate(pinput, poutput);
      std::swap(swapbuffers[1], swapbuffers[0]);
      layerOutputs.push_back(swapbuffers[0]);
    }

    std::swap(output, swapbuffers[0]);
  }

  virtual float ComputeLoss(const vector& predicted, const vector& target, LossFunction loss) override {
    return ::computeLoss(predicted, target, loss);
  }

  void Backpropagate(const std::vector<float>& input, const std::vector<float>& target, LossFunction loss) override {
    layerOutputs.resize(layers.size());
    layerDeltas.resize(layers.size());

    std::vector<float> inputBuffer = input;
    for (int i = 0; i < layers.size(); ++i) {
      layers[i]->propagate(inputBuffer, layerOutputs[i]);
      inputBuffer = layerOutputs[i];
    }

    // Compute loss gradient (dL/dy)
    std::vector<float> delta = ::computeLossDerivative(layerOutputs.back(), target, loss);
    layerDeltas.back()       = delta;

    // Backpropagate errors
    for (int l = layers.size() - 1; l >= 0; --l) {
      auto& layer  = layers[l];
      auto& output = layerOutputs[l];
      auto& gradW  = layer->gradWeigths;
      auto& gradB  = layer->gradBias;

      int                       inputSize  = l == 0 ? input.size() : layerOutputs[l - 1].size();
      const std::vector<float>& prevOutput = (l == 0) ? input : layerOutputs[l - 1];
      const std::vector<float>& deltaCurr  = layerDeltas[l];

      gradW.resize(layer->outputSize(), std::vector<float>(inputSize, 0.0f));
      gradB.resize(layer->outputSize(), 0.0f);

      // Compute gradients
      for (int i = 0; i < layer->outputSize(); ++i) {
        float dOut     = deltaCurr[i];
        float actDeriv = activation_derivative(layer->activationFunction, output[i]);

        gradB[i] = dOut * actDeriv;

        for (int j = 0; j < inputSize; ++j) {
          gradW[i][j] = prevOutput[j] * gradB[i];
        }
      }

      if (l > 0) {
        auto& nextDelta = layerDeltas[l - 1];
        nextDelta.resize(inputSize, 0.0f);
        for (int j = 0; j < inputSize; ++j) {
          float sum = 0.0f;
          for (int i = 0; i < layer->outputSize(); ++i) {
            sum += layer->weights[i][j] * gradB[i];
          }
          nextDelta[j] = sum;
        }
      }
    }
  }

  void TrainStep(const std::vector<float>& input, const std::vector<float>& target, LossFunction loss) override {
    std::vector<float> output;
    Propagate(input, output);
    Backpropagate(input, target, loss);

    if (optimizer.size() != layers.size()) {
      optimizer.clear();
      for (int i = 0; i < layers.size(); i++) {
        optimizer.push_back(std::unique_ptr<MLPOptimizer>(mlpOptimzerCreate(optimizerCreateInfo)));

        OptimizerInputParameters input;
        input.inputNeurons  = layers[i]->inputSize();
        input.outputNeurons = layers[i]->outputSize();

        optimizer.back()->initialize(input);
      }
    }
    for (int i = 0; i < layers.size(); ++i) {
      OptimizerUpdateParameters params = {
        layers[i]->weights,
        layers[i]->gradWeigths,
        layers[i]->bias,
        layers[i]->gradBias};
      optimizer[i]->update(params);
    }
  }
  virtual void SetOptimizer(const OptimizerCreateInfo ci) override {
    this->optimizerCreateInfo = ci;
  }

  void InitializeRandomize(int index) {
    auto& layer = layers[index];
    for (auto& row : layer->weights)
      for (float& w : row)
        w = ((float)rand() / RAND_MAX - 0.5f) * 2.0f; // Range: [-1, 1]
  }

  void InitializeXavier(int index) {
    auto&  layer   = layers[index];
    size_t fan_in  = layer->weights[0].size(); // Inputs to each neuron
    size_t fan_out = layer->weights.size();    // Number of neurons in this layer

    float limit = sqrt(6.0f / (fan_in + fan_out));

    for (auto& row : layer->weights)
      for (float& w : row)
        w = ((float)rand() / RAND_MAX) * 2.0f * limit - limit; // Range: [-limit, limit]
  }

  void InitializeHe(int index) {
    auto&  layer  = layers[index];
    size_t fan_in = layer->weights[0].size(); // Inputs to each neuron
    float  stddev = sqrt(2.0f / fan_in);
    float  limit  = sqrt(6.0f / fan_in);

    for (auto& row : layer->weights) {
      for (float& w : row) {
        w = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
      }
    }
  }

  void InitializeLayer(InitializationStrategy strategy, int index) override {
    switch (strategy) {
      case MLP_INITIALIZE_xAVIER: InitializeXavier(index); break;
      case MLP_INITIALIZE_RANDOM: InitializeRandomize(index); break;
      case MLP_INITIALIZE_HE: InitializeHe(index); break;
      default:
      case MLP_INITIALIZE_NONE: break;
    }
  }

  void Initialize(InitializationStrategy strategy) override {
    for (int i = 0; i < layers.size(); i++) {
      InitializeLayer(strategy, i);
    }
  }

  virtual ~MLPImpl() {}
};

MLP* mlpCreate(int inputLayerSize) {
  return new MLPImpl(inputLayerSize);
}

MLP* mlpCreate(MLPCreateInfo ci) {
  MLP* mlp = mlpCreate(ci.inputSize);
  for (int i = 0; i < ci.activations.size(); i++) {
    mlp->AddLayer(ci.hiddenSizes[i], ci.activations[i]);
  }
  mlp->AddLayer(ci.outputSize, MLP_ACTIVATION_SIGMOID);
  return mlp;
}
