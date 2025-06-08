#include "net.h"
#include "optimizer.hpp"
#include "compute.hpp"
#include "activation.hpp"
#include <core/debug.hpp>
#include <curand_kernel.h>

__device__ void fill(cuda_vector<float>& x, float value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < x.size())
    x[tid] = value;
}
// Assumes: blockDim.x == warp size (e.g., 256), gridDim.x == n (one block per row)

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
__device__ void activate(cuda_vector<float>& X, ActivationFunction function) {}

struct DenseLayer {

  ActivationFunction activationFunction;
  cuda_vector<float> weights;
  cuda_vector<float> bias;
  cuda_vector<float> gradWeights;
  cuda_vector<float> gradBias;
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

  __host__ bool isValid(const cuda_vector<float>& input) {
    if (weights.size() > 0 && input != input.size()) {
      LOG("[DenseLayer] Size mismatch. Expected input: %d, Received input: %d\n", input, (int)input.size());
      exit(1);
    }
  }

  __device__ void propagate(const cuda_vector<float>& input, cuda_vector<float>& output) {
    project(weights, input, output, inputSize);
    activate(output, activationFunction);
  }

  __device__ void backpropagate(const cuda_vector<float>& input, const cuda_vector<float>& output, const cuda_vector<float>& delta, cuda_vector<float>& prevDelta) {
    int inSize  = input.size();
    int outSize = output.size();

    fill(gradWeights, 0.f);
    fill(gradBias, 0.f);
    fill(prevDelta, 0.0f);

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

struct MLPGPU : public MLP {


  int          inputLayerSize;
  LossFunction loss = MLP_LOSS_MSE;

  //optimizer
  OptimizerCreateInfo                        optimizerCreateInfo;
  std::vector<std::unique_ptr<MLPOptimizer>> optimizer;

  void  AddLayer(int neurons, ActivationFunction function, InitializationStrategy strategy = MLP_INITIALIZE_NONE) override {}
  void  Propagate(const vector& input, vector& output) override {}
  float ComputeLoss(const vector& predicted, const vector& target, LossFunction loss) override {}
  void  Backpropagate(const vector& input, const vector& target, LossFunction loss) override {}
  void  TrainStep(const vector& input, const vector& target, LossFunction loss) override {}

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
