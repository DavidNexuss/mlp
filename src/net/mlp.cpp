#include <math.h>
#include <memory>
#include <algorithm>
#include <stdlib.h>
#include "net.h"
#include "activation.hpp"
#include "optimizer.hpp"
#include "loss.hpp"

struct Layer {
  //Layer information
  std::vector<std::vector<float>> weights;
  std::vector<float>              bias;
  ActivationFunction              activationFunction;

  //Layer meta information
  std::vector<std::vector<float>> gradWeigths;
  std::vector<float>              gradBias;

  inline int inputSize() { return weights[0].size(); }
  inline int outputSize() { return weights.size(); }

  Layer(int input, int output, ActivationFunction function) {
    weights.resize(output, std::vector<float>(input));
    bias.resize(output, 0.0f);
    this->activationFunction = function;
  }

  void propagate(const std::vector<float>& input, std::vector<float>& output) {
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
  std::vector<Layer> layers;
  int                inputLayerSize;
  LossFunction       loss = MLP_LOSS_MSE;

  //optimizer
  std::unique_ptr<MLPOptimizer>   optimizer;
  std::vector<std::vector<float>> layerOutputs;
  std::vector<std::vector<float>> layerDeltas;

  MLPImpl(int inputLayerSize) {
    this->inputLayerSize = inputLayerSize;
  }

  virtual void AddLayer(int neurons, ActivationFunction function) override {
    int lastLayerSize = layers.size() ? layers.back().outputSize() : inputLayerSize;
    layers.emplace_back(lastLayerSize, neurons, function);
  }

  virtual void Propagate(const vector& input, vector& output) override {
    std::vector<float> swapbuffers[2];
    output.clear();
    layerOutputs.clear();
    layerOutputs.push_back(input);

    for (int i = 0; i < layers.size(); i++) {
      const std::vector<float>& pinput  = i == 0 ? input : swapbuffers[0];
      std::vector<float>&       poutput = swapbuffers[1];

      layers[i].propagate(pinput, poutput);
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
      layers[i].propagate(inputBuffer, layerOutputs[i]);
      inputBuffer = layerOutputs[i];
    }

    // Compute loss gradient (dL/dy)
    std::vector<float> delta = ::computeLossDerivative(layerOutputs.back(), target, loss);
    layerDeltas.back()       = delta;

    // Backpropagate errors
    for (int l = layers.size() - 1; l >= 0; --l) {
      auto& layer  = layers[l];
      auto& output = layerOutputs[l];
      auto& gradW  = layer.gradWeigths;
      auto& gradB  = layer.gradBias;

      int                       inputSize  = l == 0 ? input.size() : layerOutputs[l - 1].size();
      const std::vector<float>& prevOutput = (l == 0) ? input : layerOutputs[l - 1];
      const std::vector<float>& deltaCurr  = layerDeltas[l];

      gradW.resize(layer.outputSize(), std::vector<float>(inputSize, 0.0f));
      gradB.resize(layer.outputSize(), 0.0f);

      // Compute gradients
      for (int i = 0; i < layer.outputSize(); ++i) {
        float dOut     = deltaCurr[i];
        float actDeriv = activation_derivative(layer.activationFunction, output[i]);

        gradB[i] = dOut * actDeriv;

        for (int j = 0; j < inputSize; ++j) {
          gradW[i][j] = prevOutput[j] * gradB[i];
        }
      }

      // Compute delta for next layer
      if (l > 0) {
        auto& nextDelta = layerDeltas[l - 1];
        nextDelta.resize(inputSize, 0.0f);
        for (int j = 0; j < inputSize; ++j) {
          float sum = 0.0f;
          for (int i = 0; i < layer.outputSize(); ++i) {
            sum += layer.weights[i][j] * gradB[i];
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

    for (int i = 0; i < layers.size(); ++i) {
      OptimizerParameters params = {
        layers[i].weights,
        layers[i].gradWeigths,
        layers[i].bias,
        layers[i].gradBias};
      optimizer->update(params);
    }
  }
  virtual void SetOptimizer(const OptimizerCreateInfo ci) override {
    this->optimizer.reset(mlpOptimzerCreate(ci));
    this->optimizer->initialize(ci);
  }
  virtual void Randomize() override {
    for (auto& layer : layers) {
      for (auto& row : layer.weights)
        for (float& w : row)
          w = ((float)rand() / RAND_MAX - 0.5f) * 2.0f; // Range: [-1, 1]
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
