#pragma once
#include <memory>
#include <vector>

/**
 * Describes the parameters of each variable x,y,z... of the function to optimize
 */
struct TunningParameters {
  float minValue;
  float maxValue;
  float stepValue;
};

/**
 * Each member represents a value for a variable xi of the function
 */
using TunningFunctionResult = std::vector<float>;

/**
 * Each member represents the i-th derivative of the y value of the function, being 0 the value, 1 the first derivative and so on...
 */
using TunningFunctionVariables = std::vector<float>;


struct TunningInterface {
  std::vector<TunningParameters> parameters;

  virtual TunningFunctionResult evaluate(const TunningFunctionVariables& values) = 0;
};

enum TunningMode {
  TUNNING_MODE_GRID,
  TUNNING_MODE_RANDOM,
  TUNNING_MODE_NEWTON,
  TUNNING_MODE_BINARY,
};

enum TunningStart {
  TUNNING_START_MIN,
  TUNNING_START_MAX,
  TUNNING_START_NONE,
  TUNNING_START_RANDOM,
};

struct TunningConfigurtion {
  std::shared_ptr<TunningInterface> interface;

  TunningFunctionVariables startValue;

  TunningMode  mode;
  TunningStart start;

  size_t rand_maxIterations = 1000;
  size_t sgd_maxIterations  = 100;
  float  sgd_learningRate   = 1.0f;
  float  sgd_momentum       = 0.9f;
};

struct TunningResult {
  TunningFunctionVariables variables;
  TunningFunctionResult    expectedResult;
  size_t                   iterations;
};


TunningResult optimize(const TunningConfigurtion& config);
void          tunning_unit_test();
