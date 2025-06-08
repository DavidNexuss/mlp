#include "tunning.hpp"
#include "net.h"

struct AutoTunning {
  virtual void                SetBase(OptimizerCreateInfo ci)          = 0;
  virtual void                SetNetwork(std::shared_ptr<MLP> net)     = 0;
  virtual void                SetDataSet(std::shared_ptr<DataSet> ds)  = 0;
  virtual void                SetTestSet(std::shared_ptr<DataSet> tes) = 0;
  virtual OptimizerCreateInfo Tune(int maxIterations)                  = 0;
};
std::shared_ptr<AutoTunning> autoTunningCreate();
