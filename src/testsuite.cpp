#include "net/net.h"
#include "net/tunning.hpp"

using Test = std::shared_ptr<MLPTrainer>;

Test xortest() {
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

  printf("\n");

  return trainer;
}

Test autoencoder() {
  printf("================[AUTOENCODER]========================\n");
  OptimizerCreateInfo optInfo;
  optInfo.learningRate = 0.1f;
  optInfo.momentum     = 0.9f;
  optInfo.function     = MLP_OPTIMIZER_SGD_MOMENTUM;
  optInfo.l2           = 1e-4;

  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(3));

  net->AddLayer(10, MLP_ACTIVATION_RELU);
  net->AddLayer(4, MLP_ACTIVATION_RELU);
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

  return trainer;
}

std::shared_ptr<MLP> createMNISTCNN() {
  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(28 * 28));
  net->AddConvolutionalLayer(1, 28, 28, 16, 3, 1, 1, MLP_ACTIVATION_RELU);
  net->AddConvolutionalLayer(16, 28, 28, 32, 3, 1, 1, MLP_ACTIVATION_RELU);
  net->AddLayer(128, MLP_ACTIVATION_RELU);
  net->AddLayer(10, MLP_ACTIVATION_SIGMOID);

  OptimizerCreateInfo optInfo;
  optInfo.learningRate = 0.05f;
  optInfo.momentum     = 0.9f;
  optInfo.function     = MLP_OPTIMIZER_SGD_MOMENTUM;
  net->SetOptimizer(optInfo);

  net->Initialize(MLP_INITIALIZE_HE);

  return net;
}

std::shared_ptr<MLP> createMNISTCNNPooling() {
  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(28 * 28));

  net->AddConvolutionalLayer(1, 28, 28, 16, 3, 1, 1, MLP_ACTIVATION_RELU);
  net->AddMaxPoolLayer(16, 28, 28, 2, 2);
  net->AddConvolutionalLayer(16, 14, 14, 32, 3, 1, 1, MLP_ACTIVATION_RELU);
  net->AddMaxPoolLayer(32, 14, 14, 2, 2);

  net->AddLayer(128, MLP_ACTIVATION_RELU);
  net->AddLayer(10, MLP_ACTIVATION_SOFTMAX);

  OptimizerCreateInfo optInfo;
  optInfo.learningRate = 0.001f;
  optInfo.momentum     = 0.9f;
  optInfo.function     = MLP_OPTIMIZER_SGD_MOMENTUM;
  net->SetOptimizer(optInfo);

  net->Initialize(MLP_INITIALIZE_HE);

  return net;
}

std::shared_ptr<MLP> createMNISTDeepMLP() {
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
std::shared_ptr<MLP> createMNISTDeepAutoencoder() {
  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(28 * 28));

  net->AddLayer(512, MLP_ACTIVATION_RELU);
  net->AddLayer(256, MLP_ACTIVATION_RELU);
  net->AddLayer(64, MLP_ACTIVATION_RELU); // Latent space (bottleneck)

  net->AddLayer(256, MLP_ACTIVATION_RELU);
  net->AddLayer(512, MLP_ACTIVATION_RELU);

  net->AddLayer(28 * 28, MLP_ACTIVATION_SIGMOID);

  OptimizerCreateInfo optInfo;
  optInfo.learningRate = 0.001f;
  optInfo.momentum     = 0.9f;
  optInfo.function     = MLP_OPTIMIZER_SGD_MOMENTUM;
  net->SetOptimizer(optInfo);

  net->Initialize(MLP_INITIALIZE_HE);

  return net;
}

Test train(std::shared_ptr<DataSet> ds, std::shared_ptr<DataSet> test, std::shared_ptr<MLP> mlp) {

  std::unique_ptr<MLPTrainer> trainer = std::unique_ptr<MLPTrainer>(mlpTrainerCreate());
  trainer->SetLossFunction(MLP_LOSS_MSE);
  trainer->SetDataset(ds);
  trainer->SetTestDataset(test);
  trainer->SetNetwork(mlp);

  printf("\n");
  return trainer;
}

std::vector<Test> mnistclassifier() {
  printf("================[MNIST TEST]========================\n");

  std::shared_ptr<DataSet> ds   = createStorageDataSet("assets/MNIST Dataset JPG format/MNIST - JPG - training/");
  std::shared_ptr<DataSet> test = createStorageDataSet("assets/MNIST Dataset JPG format/MNIST - JPG - testing/");

  return {
    train(ds, test, createMNISTCNNPooling()),
    train(ds, test, createMNISTDeepMLP()),
    train(ds, test, createMNISTCNN())};
}

Test mnistautoencoder() {
  printf("================[MNIST AUTOENCODER]========================\n");

  std::shared_ptr<DataSet> ds   = makeAutoencodingDataset(createStorageDataSet("assets/MNIST Dataset JPG format/MNIST - JPG - training/"));
  std::shared_ptr<DataSet> test = makeAutoencodingDataset(createStorageDataSet("assets/MNIST Dataset JPG format/MNIST - JPG - testing/"));

  return {train(ds, test, createMNISTDeepAutoencoder())};
}

std::vector<Test> getTests() {
  std::vector<Test> tests;

  tests.push_back(xortest());
  tests.push_back(autoencoder());
  for (auto& t : mnistclassifier()) tests.push_back(t);
  tests.push_back(mnistautoencoder());
  return tests;
}
