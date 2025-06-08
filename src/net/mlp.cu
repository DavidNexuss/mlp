#include "net.h"
#include "optimizer.hpp"
#include "compute.hpp"
#include "activation.hpp"
#include <core/debug.hpp>
#include <curand_kernel.h>

__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void reduce(const float* __restrict__ x, int count, float* __restrict__ value) {
  extern __shared__ float shared[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float sum = 0.0f;

  if (idx < count)
    sum = x[idx] + ((idx + blockDim.x < count) ? x[idx + blockDim.x] : 0.0f);

  shared[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s)
      shared[tid] += shared[tid + s];
    __syncthreads();
  }

  if (tid < 32) {
    float v = shared[tid];
    v       = warp_reduce_sum(v);
    if (tid == 0)
      atomicAdd(value, v);
  }
}

__device__ void softmax(cuda_vector<float>& x) {
  extern __shared__ float buffer[];

  float* max_buf = buffer;
  float* sum_buf = buffer + blockDim.x;

  int   tid = threadIdx.x;
  float val = (tid < x.size()) ? x[tid] : -INFINITY;

  max_buf[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1)
    if (tid < s) max_buf[tid] = fmaxf(max_buf[tid], max_buf[tid + s]);
  __syncthreads();
  float max_val = max_buf[0];

  if (tid < x.size()) {
    val    = expf(val - max_val);
    x[tid] = val;
  }
  sum_buf[tid] = val;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1)
    if (tid < s) sum_buf[tid] += sum_buf[tid + s];
  __syncthreads();
  float sum = sum_buf[0];

  if (tid < x.size())
    x[tid] = x[tid] / sum;
}

__device__ void project(cuda_vector<float>& M, const cuda_vector<float>& X, cuda_vector<float>& Y, int R) {
  extern __shared__ float partial_sums[];

  float* A = M.data();

  int row = blockIdx.x;
  int tid = threadIdx.x;

  float sum = 0.0f;

  for (int col = tid; col < R; col += blockDim.x) {
    sum += A[row * R + col] * X[col];
  }

  partial_sums[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      partial_sums[tid] += partial_sums[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    Y[row] = partial_sums[0];
  }
}
__device__ void compute_activate(cuda_vector<float>& X, ActivationFunction function) {
  if (function == MLP_ACTIVATION_SOFTMAX) softmax(X);
  else {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < X.size()) {
      X[tid] = activate(function, X[tid]);
    }
  }
}
__global__ void compute_dVal_and_gradBias(
  const float*       delta,
  const float*       output,
  float*             gradBias,
  int                outSize,
  ActivationFunction activationFunction,
  float*             dValArray) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= outSize) return;

  float dAct = activation_derivative(activationFunction, output[i]);
  float dVal = delta[i] * dAct;

  gradBias[i] += dVal;
  dValArray[i] = dVal;
}

__global__ void compute_gradWeights_and_prevDelta(
  const float* input,
  const float* weights,
  const float* dValArray,
  float*       gradWeights,
  float*       prevDelta,
  int          outSize,
  int          inSize) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= outSize || j >= inSize) return;

  float dVal   = dValArray[i];
  float inVal  = input[j];
  float weight = weights[i * inSize + j];

  gradWeights[i * inSize + j] += dVal * inVal;

  if (j == 0)
    prevDelta[j] += weight * dVal;
}

struct Layer {

  virtual bool isValid(const cuda_vector<float>& input)                                                                                                         = 0;
  virtual void propagate(const cuda_vector<float>& input, cuda_vector<float>& output)                                                                           = 0;
  virtual void backpropagate(const cuda_vector<float>& input, const cuda_vector<float>& output, const cuda_vector<float>& delta, cuda_vector<float>& prevDelta) = 0;
};

struct DenseLayer : public Layer {

  ActivationFunction activationFunction;
  cuda_vector<float> weights;
  cuda_vector<float> bias;
  cuda_vector<float> gradWeights;
  cuda_vector<float> gradBias;
  cuda_vector<float> dval;
  int                inputSize;
  int                outputSize;

  DenseLayer(int input, int output, ActivationFunction function) {
    weights     = cuda_vector<float>(input * output);
    gradWeights = cuda_vector<float>(input * output);
    bias        = cuda_vector<float>(output);
    gradBias    = cuda_vector<float>(output);

    this->activationFunction = function;

    this->inputSize  = input;
    this->outputSize = output;
  }

  bool isValid(const cuda_vector<float>& input) override {
    if (weights.size() > 0 && inputSize != input.size()) {
      LOG("[DenseLayer] Size mismatch. Expected input: %d, Received input: %d\n", inputSize, (int)input.size());
      return false;
    }
    return true;
  }

  void propagate(const cuda_vector<float>& input, cuda_vector<float>& output) override {
    output.resize(outputSize);
    project(weights, input, output, inputSize);
    compute_activate(output, activationFunction);
  }

  void backpropagate(const cuda_vector<float>& input, const cuda_vector<float>& output, const cuda_vector<float>& delta, cuda_vector<float>& prevDelta) override {
    int inSize  = input.size();
    int outSize = output.size();

    fill(gradWeights, 0.f);
    fill(gradBias, 0.f);
    fill(prevDelta, 0.0f);

    compute_dVal_and_gradBias(delta.data(), output.data(), gradBias.data(), outputSize, activationFunction, dval.data());
    compute_gradWeights_and_prevDelta(input.data(), output.data(), dval.data(), gradWeights.data(), prevDelta.data(), outputSize, inputSize);
  }
};

struct MLPGPU : public MLP {
  int          inputLayerSize;
  LossFunction loss = MLP_LOSS_MSE;

  //optimizer
  OptimizerCreateInfo                        optimizerCreateInfo;
  std::vector<std::unique_ptr<MLPOptimizer>> optimizer;
  std::vector<std::shared_ptr<Layer>>        layers;

  void AddLayer(int neurons, ActivationFunction function, InitializationStrategy strategy = MLP_INITIALIZE_NONE) override {}

  void Propagate(const vector& input, vector& output) override {
    cuda_vector<float> swapbuffers[2];

    for (int i = 0; i < layers.size(); i++) {
      const cuda_vector<float>& pinput  = i == 0 ? (const cuda_vector<float>&)input : swapbuffers[0];
      cuda_vector<float>&       poutput = swapbuffers[1];

      layers[i]->propagate(pinput, poutput);
      std::swap(swapbuffers[1], swapbuffers[0]);
    }
    output = std::move(swapbuffers[0]);
  }

  float ComputeLoss(const vector& predicted, const vector& target, LossFunction loss) override {
  }
  void Backpropagate(const vector& input, const vector& target, LossFunction loss) override {
    static std::vector<float> dummyDeltaBuffer;

    std::vector<cuda_vector<float>> layerOutputs(layers.size());
    std::vector<cuda_vector<float>> layerDeltas(layers.size());

    for (size_t i = 0; i < layers.size(); ++i) {
      const cuda_vector<float>& layerInput = i > 0 ? layerOutputs[i - 1] : (const cuda_vector<float>&)input;
      layers[i]->propagate(layerInput, layerOutputs[i]);
    }

    // Compute loss gradient (dL/dy)
    layerDeltas.back() = ::computeLossDerivative(layerOutputs.back(), target, loss);

    // Backward pass
    for (int l = static_cast<int>(layers.size()) - 1; l >= 0; --l) {
      const cuda_vector<float>& prevOutput = (l == 0) ? (const cuda_vector<float>&)input : layerOutputs[l - 1];
      const cuda_vector<float>& output     = layerOutputs[l];
      const cuda_vector<float>& delta      = layerDeltas[l];

      if (l > 0) {
        layerDeltas[l - 1].resize(prevOutput.size(), 0.0f);
      }

      layers[l]->backpropagate(prevOutput, output, delta, (l > 0 ? layerDeltas[l - 1] : dummyDeltaBuffer));
    }
  }
  void TrainStep(const vector& input, const vector& target, LossFunction loss) override {
  }

  void SetOptimizer(const OptimizerCreateInfo ci) override {
    this->optimizerCreateInfo = ci;
  }

  __device__ void InitializeRandomize(int index) {
  }

  __device__ void InitializeXavier(int index) {
  }

  __device__ void InitializeHe(int index) {
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
  }


  ~MLPGPU() {}

  //TODO to be implemented:
  void AddMaxPoolLayer(int inChannels, int inWidth, int inHeight, int kSize, int stride, InitializationStrategy init = MLP_INITIALIZE_NONE) override {}
  void AddConvolutionalLayer(int inputChannels, int inputWidth, int inputHeight, int outputChannels, int kernelSize, int stride, int padding, ActivationFunction function, InitializationStrategy strategy = MLP_INITIALIZE_NONE) override {}
};

MLP* mlpCreateGPU(int inputLayerSize) {}
MLP* mlpCreateGPU(MLPCreateInfo ci) {}
