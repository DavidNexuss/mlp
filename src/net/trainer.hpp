#include "mlp.hpp"
#include <memory>

struct MLPTrainerCreateInfo {
  LossFunction lossFunction  = MLP_LOSS_MSE;
  float        lossThreshold = 0.001f;
  int          maxEpochs     = 10'000;

  void print();
};

struct MLPTrainer : public MLPTrainerCreateInfo {
  MLPTrainer(MLPTrainerCreateInfo ci) :
    MLPTrainerCreateInfo(ci) {}

  virtual void SetLossFunction(LossFunction func)      = 0;
  virtual void SetNetwork(std::shared_ptr<MLP> mlp)    = 0;
  virtual void SetDataset(std::shared_ptr<DataSet> ds) = 0;
  virtual void Train()                                 = 0;
  virtual ~MLPTrainer() {};
};

MLPTrainer* mlpTrainerCreate(MLPTrainerCreateInfo ci = {});
