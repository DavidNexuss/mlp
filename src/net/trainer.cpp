#include "net.h"
#include <iostream>

const char* lossFunctionToString(LossFunction f) {
  switch (f) {
    case MLP_LOSS_MSE: return "Mean Squared Error (MSE)";
    default: return "Unknown";
  }
}

void MLPTrainerCreateInfo::print() {
  std::cout << "=== Trainer Configuration ===\n";
  std::cout << "Loss Function  : " << lossFunctionToString(lossFunction) << "\n";
  std::cout << "Loss Threshold : " << lossThreshold << "\n";
  std::cout << "Max Epochs     : " << maxEpochs << "\n";
  std::cout << "===============================\n";
}

struct MLPTrainerImpl : public MLPTrainer {
  std::shared_ptr<MLP>     net;
  std::shared_ptr<DataSet> ds;

  MLPTrainerImpl(MLPTrainerCreateInfo ci) :
    MLPTrainer(ci) {}

  virtual void SetLossFunction(LossFunction func) override { this->lossFunction = func; }

  virtual void SetDataset(std::shared_ptr<DataSet> ds) override { this->ds = ds; }

  virtual void SetNetwork(std::shared_ptr<MLP> net) override { this->net = net; }

  virtual void Train() override {

    float loss = 0.0f;

    for (int epoch = 0; epoch < maxEpochs; ++epoch) {
      loss = 0.0f;
      for (size_t i = 0; i < ds->inputs.size(); ++i) {
        net->TrainStep(ds->inputs[i], ds->targets[i], lossFunction);

        std::vector<float> output;
        net->Propagate(ds->inputs[i], output);
        for (size_t j = 0; j < output.size(); ++j) {
          float diff = output[j] - ds->targets[i][j];
          loss += 0.5f * diff * diff;
        }
      }

      loss /= ds->inputs.size();

      if (epoch % 500 == 0 || loss < lossThreshold) {
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
      }

      if (loss < lossThreshold) {
        std::cout << "Early stopping at epoch " << epoch << std::endl;
        break;
      }
    }
  }

  virtual ~MLPTrainerImpl() {}
};

MLPTrainer* mlpTrainerCreate(MLPTrainerCreateInfo ci) { return new MLPTrainerImpl(ci); }
