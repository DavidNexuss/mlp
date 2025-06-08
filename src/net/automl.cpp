#include <stdlib.h>
#include "automl.hpp"
#include "tunning.hpp"
#include <iostream>

struct AutoTunningImpl : public AutoTunning, TunningInterface, std::enable_shared_from_this<TunningInterface> {
  std::shared_ptr<MLP>     base;
  std::shared_ptr<DataSet> data;
  std::shared_ptr<DataSet> test;

  int                 maxEpochs;
  OptimizerCreateInfo baseCI;

  void SetBase(OptimizerCreateInfo ci) override {
    baseCI = ci;
  }

  AutoTunningImpl() {
    this->parameters.push_back({0.0001f, 0.15f, 0.001f});
    this->parameters.push_back({0.0001f, 0.8f, 0.001f});
    this->parameters.push_back({0.000000000001f, 0.000000002f, 0.001f});
  }

  OptimizerCreateInfo fromVariables(const TunningFunctionVariables& values) {
    OptimizerCreateInfo ci = baseCI;
    ci.learningRate        = values[0];
    ci.momentum            = values[1];
    ci.l2                  = values[2];
    return ci;
  }

  TunningFunctionVariables fromInfo(OptimizerCreateInfo ci) {
    return {ci.learningRate, ci.momentum, ci.l2};
  }

  void SetNetwork(std::shared_ptr<MLP> net) override {
    base = net;
  }

  void SetDataSet(std::shared_ptr<DataSet> ds) override {
    data = ds;
  }

  void SetTestSet(std::shared_ptr<DataSet> tes) override {
    test = tes;
  }

  TunningFunctionResult evaluate(const TunningFunctionVariables& values) override {
    std::shared_ptr<MLPTrainer> trainer = std::shared_ptr<MLPTrainer>(mlpTrainerCreate());
    auto                        net     = base->Clone();
    net->SetOptimizer(fromVariables(values));
    trainer->SetDataset(data);
    trainer->SetTestDataset(test);
    trainer->SetNetwork(net);
    return {trainer->Train(10)};
  }

  OptimizerCreateInfo Tune(int maxIterations) override {
    TunningConfigurtion config;
    config.interface          = shared_from_this();
    config.mode               = TUNNING_MODE_RANDOM;
    config.start              = TUNNING_START_NONE;
    config.rand_maxIterations = maxIterations;
    config.start              = {};

    TunningResult result = optimize(config);

    std::cout << "Optimized Variables:\n";
    for (float v : result.variables) std::cout << v << " ";
    std::cout << "\nFunction Value: " << result.expectedResult[0] << std::endl;

    return fromVariables(result.variables);
  }
};

std::shared_ptr<AutoTunning> autoTunningCreate() { return std::shared_ptr<AutoTunning>(new AutoTunningImpl()); }
