#include "net/net.h"
#include "util/stdout.hpp"

//XOR test suite backpropagator
void xortest() {
  OptimizerCreateInfo optInfo;
  optInfo.learningRate = 0.1f;
  optInfo.momentum     = 0.9f;
  optInfo.function     = MLP_OPTIMIZER_SGD_MOMENTUM;

  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(2));
  net->AddLayer(4, MLP_ACTIVATION_RELU);
  net->AddLayer(1, MLP_ACTIVATION_SIGMOID);
  net->SetOptimizer(optInfo);
  net->Initialize(MLP_INITIALIZE_RANDOM);

  std::shared_ptr<DataSet> ds = std::make_shared<DataSet>();

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
}

void autoencoder() {
  OptimizerCreateInfo optInfo;
  optInfo.learningRate = 0.1f;
  optInfo.momentum     = 0.5f;
  optInfo.function     = MLP_OPTIMIZER_SGD_MOMENTUM;

  std::shared_ptr<MLP> net = std::shared_ptr<MLP>(mlpCreate(2));

  net->AddLayer(8, MLP_ACTIVATION_RELU, MLP_INITIALIZE_HE);
  net->AddLayer(2, MLP_ACTIVATION_RELU, MLP_INITIALIZE_HE);
  net->AddLayer(8, MLP_ACTIVATION_RELU, MLP_INITIALIZE_HE);
  net->AddLayer(2, MLP_ACTIVATION_SIGMOID, MLP_INITIALIZE_xAVIER);

  net->SetOptimizer(optInfo);

  std::shared_ptr<DataSet>
    ds = std::make_shared<DataSet>();

  ds->inputs = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f}};

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
}

int main() {
  xortest();
  autoencoder();
}
