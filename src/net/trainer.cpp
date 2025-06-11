#include "net.h"
#include <util/stdout.hpp>
#include <iostream>

struct MLPTrainerImpl : public MLPTrainer {
  std::shared_ptr<MLP>     net;
  std::shared_ptr<DataSet> ds;
  std::shared_ptr<DataSet> test;
  LossFunction             lossFunction = MLP_LOSS_MSE;


  std::shared_ptr<DataSet> getDataset() override { return ds; }
  std::shared_ptr<MLP>     getNetwork() override { return net; }
  std::shared_ptr<DataSet> getTestSet() override { return test; }

  void SetLossFunction(LossFunction func) override { this->lossFunction = func; }

  void SetDataset(std::shared_ptr<DataSet> ds) override { this->ds = ds; }

  void SetNetwork(std::shared_ptr<MLP> net) override { this->net = net; }

  void SetTestDataset(std::shared_ptr<DataSet> _ds) override { this->test = _ds; }

  virtual void Train() override {

    const float lossThreshold = 0.001f;
    const int   maxEpochs     = 10'000;

    auto lossDataset = [this](std::shared_ptr<DataSet> ds) {
      float loss = 0.0f;
      for (size_t i = 0; i < ds->getInputCount(); ++i) {
        net->TrainStep(ds->getInput(i), ds->getOutput(i), lossFunction);

        std::vector<float> output;
        net->Propagate(ds->getInput(i), output);
        for (size_t j = 0; j < output.size(); ++j) {
          float diff = output[j] - ds->getOutput(i)[j];
          loss += 0.5f * diff * diff;
        }
      }

      loss /= ds->getInputCount();
      return loss;
    };

    for (int epoch = 0; epoch < maxEpochs; ++epoch) {

      float loss = lossDataset(ds);

      std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;

      if (loss < lossThreshold) {
        std::cout << "Early stopping at epoch " << epoch << std::endl;
        break;
      }
    }

    if (test) {
      std::cout << "Performing testing... ";
      float loss = lossDataset(test);
      std::cout << " loss = " << loss << std::endl;
    }
  }

  virtual ~MLPTrainerImpl() {}
};

MLPTrainer* mlpTrainerCreate() {
  return new MLPTrainerImpl;
}
