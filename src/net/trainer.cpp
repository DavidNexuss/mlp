#include "net.h"
#include <iostream>

struct MLPTrainerImpl : public MLPTrainer {
  std::shared_ptr<MLP>     net;
  std::shared_ptr<DataSet> ds;
  LossFunction             lossFunction = MLP_LOSS_MSE;

  virtual void SetDataset(std::shared_ptr<DataSet> ds) override { this->ds = ds; }

  virtual void SetNetwork(std::shared_ptr<MLP> net) override { this->net = net; }

  virtual void Train() override {

    const float lossThreshold = 0.01f;
    const int   maxEpochs     = 10'000;
    float       loss          = 0.0f;

    for (int epoch = 0; epoch < maxEpochs; ++epoch) {
      loss = 0.0f;
      for (size_t i = 0; i < ds->inputs.size(); ++i) {
        net->TrainStep(ds->inputs[i], ds->targets[i], lossFunction);

        std::vector<float> output;
        net->Propagate(ds->inputs[i], output);
        float diff = output[0] - ds->targets[i][0];
        loss += 0.5f * diff * diff;
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

MLPTrainer* mlpTrainerCreate() {
  return new MLPTrainerImpl;
}
