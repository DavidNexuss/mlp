#include <math.h>
#include <memory>
#include <algorithm>
#include <stdlib.h>
#include "net.h"
#include "activation.hpp"
#include "optimizer.hpp"
#include "loss.hpp"
#include <memory>
#include <core/debug.hpp>
#include "tracy/Tracy.hpp"

#define USE_OPENMP 0

#if __linux__ && USE_OPENMP
#  include <omp.h>
#else

int omp_get_thread_num() { return 0; }
int omp_get_max_threads() { return 1; }

#endif


struct Layer {
  //Layer information
  std::vector<std::vector<float>> weights;
  std::vector<float>              bias;
  ActivationFunction              activationFunction;

  //Layer meta information
  std::vector<std::vector<float>> gradWeigths;
  std::vector<float>              gradBias;

  virtual void propagate(const std::vector<float>& input, std::vector<float>& output)                                                                           = 0;
  virtual void backpropagate(const std::vector<float>& input, const std::vector<float>& output, const std::vector<float>& delta, std::vector<float>& prevDelta) = 0;

  virtual int inputSize() { return weights[0].size(); }
  virtual int outputSize() { return weights.size(); }

  virtual ~Layer() = default;
};

struct MaxPoolingLayer : public Layer {
  int inputChannels;
  int inputWidth;
  int inputHeight;

  int kernelSize;
  int stride;

  virtual int inputSize() override {
    return inputChannels * inputWidth * inputHeight;
  }

  virtual int outputSize() override {
    return inputChannels * outputWidth() * outputHeight();
  }

  MaxPoolingLayer(int inChannels, int inWidth, int inHeight, int kSize, int stride) :
    inputChannels(inChannels), inputWidth(inWidth), inputHeight(inHeight), kernelSize(kSize), stride(stride) {}

  inline int outputWidth() const {
    return (inputWidth - kernelSize) / stride + 1;
  }

  inline int outputHeight() const {
    return (inputHeight - kernelSize) / stride + 1;
  }

  void propagate(const std::vector<float>& input, std::vector<float>& output) override {

    ZoneScopedN("Max Pooling Propagate");
    int ow = outputWidth();
    int oh = outputHeight();
    output.resize(inputChannels * ow * oh);

#pragma omp parallel for collapse(3)
    for (int c = 0; c < inputChannels; ++c) {
      for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
          float maxVal = -std::numeric_limits<float>::infinity();

          for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
              int ix = ox * stride + kx;
              int iy = oy * stride + ky;

              int inputIndex = c * inputWidth * inputHeight + iy * inputWidth + ix;
              maxVal         = std::max(maxVal, input[inputIndex]);
            }
          }

          int outIndex     = c * ow * oh + oy * ow + ox;
          output[outIndex] = maxVal;
        }
      }
    }
  }
  void backpropagate(const std::vector<float>& input,
                     const std::vector<float>& output,
                     const std::vector<float>& delta,
                     std::vector<float>&       prevDelta) override {

    ZoneScopedN("Max Pooling BackPropagate");
    int ow = outputWidth();
    int oh = outputHeight();

    prevDelta.resize(inputChannels * inputWidth * inputHeight, 0.0f);

#pragma omp parallel for schedule(static)
    for (int c = 0; c < inputChannels; ++c) {
      for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
          float maxVal   = -std::numeric_limits<float>::infinity();
          int   maxIndex = -1;

          // Find max element index in input window corresponding to this output position
          for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
              int ix = ox * stride + kx;
              int iy = oy * stride + ky;

              int inputIndex = c * inputWidth * inputHeight + iy * inputWidth + ix;

              float val = input[inputIndex];
              if (val > maxVal) {
                maxVal   = val;
                maxIndex = inputIndex;
              }
            }
          }

          int outIndex = c * ow * oh + oy * ow + ox;

          // Propagate delta only through the max input element
          prevDelta[maxIndex] += delta[outIndex];
        }
      }
    }
  }
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
    ZoneScopedN("Convolution Propagate");
    int ow = outputWidth();
    int oh = outputHeight();
    output.resize(outputChannels * oh * ow, 0.0f);

#pragma omp parallel for schedule(static)
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

  void backpropagate(const std::vector<float>& input, const std::vector<float>& output, const std::vector<float>& delta, std::vector<float>& prevDelta) override {
    ZoneScopedN("Convolution BackPropagate");
    int iw = inputWidth;
    int ih = inputHeight;
    int ow = outputWidth();
    int oh = outputHeight();

    int kernelArea = inputChannels * kernelSize * kernelSize;

    for (auto& gw : gradWeigths) std::fill(gw.begin(), gw.end(), 0.0f);
    std::fill(gradBias.begin(), gradBias.end(), 0.0f);

    prevDelta.resize(input.size(), 0.0f); // Gradient w.r.t. input

#pragma omp parallel for schedule(static) collapse(3)
    for (int oc = 0; oc < outputChannels; ++oc) {
      for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
          int   outIdx   = oc * ow * oh + oy * ow + ox;
          float act      = output[outIdx];
          float dAct     = activation_derivative(activationFunction, act);
          float deltaVal = delta[outIdx] * dAct;

          gradBias[oc] += deltaVal;

          int wIndex = 0;

          for (int ic = 0; ic < inputChannels; ++ic) {
            for (int ky = 0; ky < kernelSize; ++ky) {
              for (int kx = 0; kx < kernelSize; ++kx) {
                int ix = ox * stride + kx - padding;
                int iy = oy * stride + ky - padding;

                if (ix >= 0 && ix < iw && iy >= 0 && iy < ih) {
                  int   inIdx = ic * iw * ih + iy * iw + ix;
                  float inVal = input[inIdx];

                  gradWeigths[oc][wIndex] += deltaVal * inVal;

                  prevDelta[inIdx] += weights[oc][wIndex] * deltaVal;
                }

                ++wIndex;
              }
            }
          }
        }
      }
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
    ZoneScopedN("Dense Propagate");
    output.resize(outputSize());

    if (weights.size() > 0 && weights[0].size() != input.size()) {
      LOG("[DenseLayer] Size mismatch. Expected input: %d, Received input: %d\n", inputSize(), (int)input.size());
      exit(1);
    }

#pragma omp parallel for
    for (int i = 0; i < (int)output.size(); i++) {
      float accum = bias[i];
      for (int j = 0; j < (int)input.size(); j++) {
        accum += input[j] * weights[i][j];
      }
      output[i] = accum;
    }

    if (activationFunction == MLP_ACTIVATION_SOFTMAX) {
      softmax(output);
    } else {
#pragma omp parallel for
      for (int i = 0; i < (int)output.size(); i++) {
        output[i] = activate(activationFunction, output[i]);
      }
    }
  }
  void backpropagate(const std::vector<float>& input, const std::vector<float>& output, const std::vector<float>& delta, std::vector<float>& prevDelta) override {
    ZoneScopedN("Dense BackPropagate");
    int inSize  = input.size();
    int outSize = output.size();

    // Initialize gradients
    if (gradWeigths.size() != outSize || gradWeigths[0].size() != inSize) {
      gradWeigths.resize(outSize, std::vector<float>(inSize, 0.0f));
    } else {
      for (auto& row : gradWeigths) std::fill(row.begin(), row.end(), 0.0f);
    }

    if (gradBias.size() != outSize) {
      gradBias.resize(outSize, 0.0f);
    } else {
      std::fill(gradBias.begin(), gradBias.end(), 0.0f);
    }

    prevDelta.assign(inSize, 0.0f);

    // Create thread-local buffer for prevDelta to avoid race condition
    std::vector<std::vector<float>> threadPrevDelta(omp_get_max_threads(), std::vector<float>(inSize, 0.0f));

#pragma omp parallel
    {
      int                 tid            = omp_get_thread_num();
      std::vector<float>& localPrevDelta = threadPrevDelta[tid];

#pragma omp for
      for (int i = 0; i < outSize; ++i) {
        float dAct = activation_derivative(activationFunction, output[i]);
        float dVal = delta[i] * dAct;

        gradBias[i] += dVal;

        for (int j = 0; j < inSize; ++j) {
          float inVal = input[j];

          gradWeigths[i][j] += dVal * inVal;

          localPrevDelta[j] += weights[i][j] * dVal;
        }
      }
    }

    // Reduce thread-local prevDelta into global prevDelta
    for (int tid = 0; tid < (int)threadPrevDelta.size(); ++tid) {
      for (int j = 0; j < inSize; ++j) {
        prevDelta[j] += threadPrevDelta[tid][j];
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

  MLPImpl(int inputLayerSize) {
    this->inputLayerSize = inputLayerSize;
  }

  void AddLayer(int neurons, ActivationFunction function, InitializationStrategy init) override {
    int lastLayerSize = layers.size() ? layers.back()->outputSize() : inputLayerSize;
    layers.push_back(std::shared_ptr<Layer>(new DenseLayer(lastLayerSize, neurons, function)));
    if (init != MLP_INITIALIZE_NONE)
      InitializeLayer(init, layers.size() - 1);
  }

  void AddMaxPoolLayer(int inChannels, int inWidth, int inHeight, int kSize, int stride, InitializationStrategy init) override {
    layers.push_back(std::make_shared<MaxPoolingLayer>(inChannels, inWidth, inHeight, kSize, stride));

    if (init != MLP_INITIALIZE_NONE) {
      InitializeLayer(init, layers.size() - 1);
    }
  }

  void AddConvolutionalLayer(int inputChannels, int inputWidth, int inputHeight, int outputChannels, int kernelSize, int stride, int padding, ActivationFunction function, InitializationStrategy strategy) override {
    layers.push_back(std::make_shared<ConvolutionalLayer>(inputChannels, inputWidth, inputHeight, outputChannels, kernelSize, stride, padding, function));

    if (strategy != MLP_INITIALIZE_NONE) {
      InitializeLayer(strategy, layers.size() - 1);
    }
  }

  void Propagate(const vector& input, vector& output) override {
    std::vector<float> swapbuffers[2];
    output.clear();

    for (int i = 0; i < layers.size(); i++) {
      const std::vector<float>& pinput  = i == 0 ? input : swapbuffers[0];
      std::vector<float>&       poutput = swapbuffers[1];

      layers[i]->propagate(pinput, poutput);
      std::swap(swapbuffers[1], swapbuffers[0]);
    }

    std::swap(output, swapbuffers[0]);
  }

  float ComputeLoss(const vector& predicted, const vector& target, LossFunction loss) override {
    return ::computeLoss(predicted, target, loss);
  }

  void Backpropagate(const std::vector<float>& input, const std::vector<float>& target, LossFunction loss) override {
    ZoneScopedN("BackPropagate");

    static std::vector<float> dummyDeltaBuffer;

    std::vector<std::vector<float>> layerOutputs(layers.size());
    std::vector<std::vector<float>> layerDeltas(layers.size());

    // Forward pass
    std::vector<float> buffer = input;
    for (size_t i = 0; i < layers.size(); ++i) {
      layers[i]->propagate(buffer, layerOutputs[i]);
      buffer = layerOutputs[i];
    }

    // Compute loss gradient (dL/dy)
    layerDeltas.back() = ::computeLossDerivative(layerOutputs.back(), target, loss);

    // Backward pass
    for (int l = static_cast<int>(layers.size()) - 1; l >= 0; --l) {
      const std::vector<float>& prevOutput = (l == 0) ? input : layerOutputs[l - 1];
      const std::vector<float>& output     = layerOutputs[l];
      const std::vector<float>& delta      = layerDeltas[l];

      // Prepare previous delta container
      if (l > 0) {
        layerDeltas[l - 1].resize(prevOutput.size(), 0.0f);
      }

      layers[l]->backpropagate(prevOutput, output, delta, (l > 0 ? layerDeltas[l - 1] : dummyDeltaBuffer));
    }
  }

  void TrainStep(const std::vector<float>& input, const std::vector<float>& target, LossFunction loss) override {
    ZoneScopedN("TrainStep");
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

  void SetOptimizer(OptimizerCreateInfo ci) override {
    if (ci.function == MLP_OPTIMIZER_ADAM)
      ci.learningRate = ci.learningRate * 0.1f;
    this->optimizerCreateInfo = ci;
  }

  void InitializeRandomize(int index) {
    auto& layer = layers[index];
    for (auto& row : layer->weights)
      for (float& w : row)
        w = ((float)rand() / RAND_MAX - 0.5f) * 2.0f; // Range: [-1, 1]
  }

  void InitializeXavier(int index) {
    auto& layer = layers[index];
    if (layer->weights.size() == 0) return;
    size_t fan_in  = layer->weights[0].size(); // Inputs to each neuron
    size_t fan_out = layer->weights.size();    // Number of neurons in this layer

    float limit = sqrt(6.0f / (fan_in + fan_out));

    for (auto& row : layer->weights)
      for (float& w : row)
        w = ((float)rand() / RAND_MAX) * 2.0f * limit - limit; // Range: [-limit, limit]
  }

  void InitializeHe(int index) {
    auto& layer = layers[index];
    if (layer->weights.size() == 0) return;
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
    ZoneScopedN("Initialize");
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
