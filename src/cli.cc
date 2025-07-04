#include "net/net.h"
#include "util/stdout.hpp"
#include "net/tunning.hpp"

bool useAdam         = false;
bool useCrossEntropy = false;
//XOR test suite backpropagator

std::string mangleOutput(const std::string& filename) {
  return std::string(useAdam ? "_adam_" : "_mom_") + std::string(useCrossEntropy ? "_cross_" : "_mse_") + filename;
}
void xortest() {
  printf("================[XOR TEST]========================\n");
  OptimizerCreateInfo optInfo;
  optInfo.learningRate = 0.1f;
  optInfo.momentum     = 0.9f;
  optInfo.function     = useAdam ? MLP_OPTIMIZER_ADAM : MLP_OPTIMIZER_SGD_MOMENTUM;

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
  trainer->SetResultFile(mangleOutput("xor.csv"));
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
  trainer->SetResultFile(mangleOutput("autoencoder.csv"));
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



void configureOptimizer(std::shared_ptr<MLP> mlp) {
  OptimizerCreateInfo optInfo;
  if (useAdam) {
    optInfo.function   = MLP_OPTIMIZER_ADAM;
    optInfo.adam_beta1 = 0.9f;
    optInfo.adam_beta2 = 0.999f;
  } else {
    optInfo.function     = MLP_OPTIMIZER_SGD_MOMENTUM;
    optInfo.learningRate = 0.002f;
    optInfo.momentum     = 0.9f;
    optInfo.l2           = 0.0001f;
  }

  mlp->SetOptimizer(optInfo);
}

std::shared_ptr<MLP> createMNISTCNN() {
  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(28 * 28));
  net->AddConvolutionalLayer(1, 28, 28, 16, 3, 1, 1, MLP_ACTIVATION_RELU);
  net->AddConvolutionalLayer(16, 28, 28, 32, 3, 1, 1, MLP_ACTIVATION_RELU);
  net->AddLayer(128, MLP_ACTIVATION_RELU);
  net->AddLayer(10, MLP_ACTIVATION_SOFTMAX);

  configureOptimizer(net);

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

  configureOptimizer(net);

  net->Initialize(MLP_INITIALIZE_HE);

  return net;
}

std::shared_ptr<MLP> createMNISTDeepMLP() {
  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(28 * 28));

  // Deep fully connected layers with ReLU activations
  net->AddLayer(512, MLP_ACTIVATION_RELU);
  net->AddLayer(256, MLP_ACTIVATION_RELU);
  net->AddLayer(128, MLP_ACTIVATION_RELU);

  net->AddLayer(10, MLP_ACTIVATION_SIGMOID);

  configureOptimizer(net);

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

  configureOptimizer(net);

  net->Initialize(MLP_INITIALIZE_HE);

  return net;
}

void train(std::shared_ptr<DataSet> ds, std::shared_ptr<DataSet> test, std::shared_ptr<MLP> mlp, const std::string& suffix) {

  std::unique_ptr<MLPTrainer> trainer = std::unique_ptr<MLPTrainer>(mlpTrainerCreate());
  if (useCrossEntropy)
    trainer->SetLossFunction(MLP_LOSS_CROSS_ENTROPY);
  else
    trainer->SetLossFunction(MLP_LOSS_MSE);

  trainer->SetResultFile(mangleOutput("mnist" + suffix + ".csv"));
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

  auto executeTest = [&]() {
    printf("====(CNN + Pooling)===\n");
    train(ds, test, createMNISTCNNPooling(), "_cnn_pooling_");
    printf("====(DNN)===\n");
    train(ds, test, createMNISTDeepMLP(), "_deep_");
    printf("====(CNN)===\n");
    train(ds, test, createMNISTCNN(), "_cnn_");
  };

  executeTest();
}

void mnistautoencoder() {
  printf("================[MNIST AUTOENCODER]========================\n");

  std::shared_ptr<DataSet> ds   = makeAutoencodingDataset(createStorageDataSet("assets/MNIST Dataset JPG format/MNIST - JPG - training/"));
  std::shared_ptr<DataSet> test = makeAutoencodingDataset(createStorageDataSet("assets/MNIST Dataset JPG format/MNIST - JPG - testing/"));

  train(ds, test, createMNISTDeepAutoencoder(), "_auto_deep_");
}

void experiments() {
  xortest();
  autoencoder();
  mnistclassifier();
}
int main() {
  //tunning_unit_test();
  xortest();
  autoencoder();
  mnistclassifier();
  //mnistautoencoder();
}
