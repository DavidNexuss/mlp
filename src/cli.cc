#include "net/net.h"
#include "util/stdout.hpp"

//XOR test suite backpropagator
void xortest() {
  printf("================[XOR TEST]========================\n");
  OptimizerCreateInfo optInfo;
  optInfo.learningRate = 0.1f;
  optInfo.momentum     = 0.9f;
  optInfo.function     = MLP_OPTIMIZER_SGD_MOMENTUM;

  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(2));
  net->AddLayer(4, MLP_ACTIVATION_RELU);
  net->AddLayer(1, MLP_ACTIVATION_SIGMOID);
  net->SetOptimizer(optInfo);
  net->Initialize(MLP_INITIALIZE_RANDOM);

  std::shared_ptr<ManualDataSet> ds = std::make_shared<ManualDataSet>();

  ds->inputs = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f}};

  ds->targets = {
    {0.0f},
    {1.0f},
    {1.0f},
    {0.0f}};

  std::unique_ptr<MLPTrainer> trainer = std::unique_ptr<MLPTrainer>(mlpTrainerCreate());

  trainer->SetLossFunction(MLP_LOSS_MSE);
  trainer->SetDataset(ds);
  trainer->SetNetwork(net);
  trainer->Train();


  for (int i = 0; i < ds->inputs.size(); i++) {
    std::vector<float> output;
    net->Propagate(ds->inputs[i], output);
    std::cout << "Input: " << ds->inputs[i] << " " << output << std::endl;
  }

  printf("\n");
}

void autoencoder() {
  printf("================[AUTOENCODER]========================\n");
  OptimizerCreateInfo optInfo;
  optInfo.learningRate = 0.1f;
  optInfo.momentum     = 0.9f;
  optInfo.function     = MLP_OPTIMIZER_SGD_MOMENTUM;

  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(2));

  net->AddLayer(10, MLP_ACTIVATION_RELU);
  net->AddLayer(2, MLP_ACTIVATION_RELU);
  net->AddLayer(10, MLP_ACTIVATION_RELU);
  net->AddLayer(3, MLP_ACTIVATION_SIGMOID);

  net->SetOptimizer(optInfo);
  net->Initialize(MLP_INITIALIZE_RANDOM);

  std::shared_ptr<ManualDataSet> ds = std::make_shared<ManualDataSet>();

  ds->inputs = {
    {0.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 1.0f},
    {1.0f, 0.0f, 1.0f},
    {1.0f, 1.0f, 0.0f}};

  ds->targets = ds->inputs;

  std::unique_ptr<MLPTrainer> trainer = std::unique_ptr<MLPTrainer>(mlpTrainerCreate());

  trainer->SetLossFunction(MLP_LOSS_MSE);
  trainer->SetDataset(ds);
  trainer->SetNetwork(net);
  trainer->Train();


  for (int i = 0; i < ds->inputs.size(); i++) {
    std::vector<float> output;
    net->Propagate(ds->inputs[i], output);
    std::cout << "Input: " << ds->inputs[i] << " " << output << std::endl;
  }
  printf("\n");
}

std::shared_ptr<MLP> createMNISTCNN() {
  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(28 * 28));
  net->AddConvolutionalLayer(1, 28, 28, 16, 3, 1, 1, MLP_ACTIVATION_RELU);
  net->AddConvolutionalLayer(16, 28, 28, 32, 3, 1, 1, MLP_ACTIVATION_RELU);
  net->AddLayer(128, MLP_ACTIVATION_RELU);
  net->AddLayer(10, MLP_ACTIVATION_SIGMOID);

  OptimizerCreateInfo optInfo;
  optInfo.learningRate = 0.005f;
  optInfo.momentum     = 0.9f;
  optInfo.function     = MLP_OPTIMIZER_SGD_MOMENTUM;
  net->SetOptimizer(optInfo);

  net->Initialize(MLP_INITIALIZE_HE);

  return net;
}

std::shared_ptr<MLP> createMNISTDeepMLP() {
  // Input layer size is 28*28 = 784 pixels flattened
  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(28 * 28));

  // Deep fully connected layers with ReLU activations
  net->AddLayer(512, MLP_ACTIVATION_RELU);
  net->AddLayer(256, MLP_ACTIVATION_RELU);
  net->AddLayer(128, MLP_ACTIVATION_RELU);

  // Output layer with 10 neurons for 10 classes, using sigmoid or softmax
  net->AddLayer(10, MLP_ACTIVATION_SIGMOID);

  OptimizerCreateInfo optInfo;
  optInfo.learningRate = 0.005f;
  optInfo.momentum     = 0.9f;
  optInfo.function     = MLP_OPTIMIZER_SGD_MOMENTUM;
  net->SetOptimizer(optInfo);

  net->Initialize(MLP_INITIALIZE_HE);

  return net;
}

void train(std::shared_ptr<DataSet> ds, std::shared_ptr<DataSet> test, std::shared_ptr<MLP> mlp) {

  std::unique_ptr<MLPTrainer> trainer = std::unique_ptr<MLPTrainer>(mlpTrainerCreate());
  trainer->SetLossFunction(MLP_LOSS_MSE);
  trainer->SetDataset(ds);
  trainer->SetTestDataset(test);
  trainer->SetNetwork(mlp);
  trainer->Train();

  printf("\n");
}

void mnistclassifier() {

  printf("================[MNIST TEST]========================\n");

  std::shared_ptr<DataSet> ds   = createStorageDataSet("assets/MNIST Dataset JPG format/MNIST - JPG - training/");
  std::shared_ptr<DataSet> test = createStorageDataSet("assets/MNIST Dataset JPG format/MNIST - JPG - testing/");

  printf("====(DNN)===\n");
  train(ds, test, createMNISTDeepMLP());
  printf("====(CNN)===\n");
  train(ds, test, createMNISTCNN());
}

int main() {
  xortest();
  autoencoder();
  mnistclassifier();
}
