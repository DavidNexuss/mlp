#include "net.h"
#include <util/stdout.hpp>
#include <iostream>

struct MLPTrainerImpl : public MLPTrainer {
  std::shared_ptr<MLP>     net;
  std::shared_ptr<DataSet> ds;
  std::shared_ptr<DataSet> test;
  LossFunction             lossFunction = MLP_LOSS_MSE;

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

      if (epoch % 10 == 0 || loss < lossThreshold) {
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
      }

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

/*
struct MLPTrainerImpl : public MLPTrainer {
  std::shared_ptr<MLP>     net;
  std::shared_ptr<DataSet> ds;
  LossFunction             lossFunction = MLP_LOSS_MSE;

  float learningRate = 0.1f;
  float decayFactor  = 0.95f;
  int   batchSize    = 2; // mini-batches

  virtual void SetLossFunction(LossFunction func) override { this->lossFunction = func; }
  virtual void SetDataset(std::shared_ptr<DataSet> ds) override { this->ds = ds; }
  virtual void SetNetwork(std::shared_ptr<MLP> net) override { this->net = net; }

  virtual void Train() override {
    const float lossThreshold       = 0.001f;
    const int   maxEpochs           = 10'000;
    float       loss                = 0.0f;
    int         noImprovementEpochs = 0;
    float       bestLoss            = 1e10;

    size_t numSamples = ds->inputs.size();

    for (int epoch = 0; epoch < maxEpochs; ++epoch) {
      loss = 0.0f;

      for (size_t batchStart = 0; batchStart < numSamples; batchStart += batchSize) {
        size_t batchEnd = std::min(batchStart + batchSize, numSamples);

        // Train step over batch
        for (size_t i = batchStart; i < batchEnd; ++i) {
          net->TrainStep(ds->inputs[i], ds->targets[i], lossFunction);
        }

        // Optionally update optimizer params here if needed (momentum, learning rate)

        // Accumulate loss on batch
        for (size_t i = batchStart; i < batchEnd; ++i) {
          std::vector<float> output;
          net->Propagate(ds->inputs[i], output);
          for (size_t j = 0; j < output.size(); ++j) {
            float diff = output[j] - ds->targets[i][j];
            loss += 0.5f * diff * diff;
          }
        }
      }

      loss /= numSamples;

      if (loss < bestLoss - 1e-6f) {
        bestLoss            = loss;
        noImprovementEpochs = 0;
      } else {
        noImprovementEpochs++;
      }

      // Decay learning rate if no improvement for 100 epochs
      if (noImprovementEpochs > 100) {
        learningRate *= decayFactor;
        noImprovementEpochs = 0;
        std::cout << "[Trainer] Decayed learning rate to " << learningRate << std::endl;
        // Update optimizer learning rate here if your MLP interface supports it
      }

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

*/
MLPTrainer* mlpTrainerCreate() {
  return new MLPTrainerImpl;
}
