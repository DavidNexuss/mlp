#include <stdlib.h>
#include "tunning.hpp"
#include <limits>
#include <random>
#include <cassert>
#include <iostream>
#include <core/debug.hpp>

float randf() {
  static thread_local std::mt19937 generator(std::random_device{}());

  static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  return dist(generator);
}

const float epsilon = 1e-6f;

//===========================[INITIALIZATION TECHNIQUES]=================================

TunningFunctionVariables initialize_max(const TunningConfigurtion& config) {
  const size_t dim = config.interface->parameters.size();

  TunningFunctionVariables current(dim, 0.0f);

  for (int i = 0; i < dim; i++) {
    float max = config.interface->parameters[i].maxValue;

    current[i] = max;
  }
  return current;
}

TunningFunctionVariables initialize_min(const TunningConfigurtion& config) {
  const size_t dim = config.interface->parameters.size();

  TunningFunctionVariables current(dim, 0.0f);

  for (int i = 0; i < dim; i++) {
    float min = config.interface->parameters[i].minValue;

    current[i] = min;
  }
  return current;
}

TunningFunctionVariables initialize_random(const TunningConfigurtion& config) {
  const size_t dim = config.interface->parameters.size();

  TunningFunctionVariables current(dim, 0.0f);

  for (int i = 0; i < dim; i++) {
    float min = config.interface->parameters[i].minValue;
    float max = config.interface->parameters[i].maxValue;

    float range = max - min;

    current[i] = randf() * range + min;
  }
  return current;
}

TunningFunctionVariables initialize(const TunningConfigurtion& config) {
  if (config.start == TUNNING_START_MIN) {
    return initialize_min(config);
  } else if (config.start == TUNNING_START_MAX) {
    return initialize_max(config);
  } else if (config.start == TUNNING_START_RANDOM) {
    return initialize_random(config);
  } else {
    return config.startValue;
  }
}

//===============================[OPTIMIZATION]============================================================

TunningResult optimize_grid(const TunningConfigurtion& config, std::vector<float>& variables) {
  const auto& interface = config.interface;
  const auto& params    = interface->parameters;

  TunningResult best;
  float         bestScore = std::numeric_limits<float>::max();

  std::vector<size_t> steps;
  for (const auto& param : params) {
    size_t stepCount = static_cast<size_t>(
                         std::floor((param.maxValue - param.minValue) / param.stepValue)) +
      1;
    steps.push_back(stepCount);
  }

  size_t paramCount = params.size();

  std::vector<size_t> indices(paramCount, 0);

  size_t totalCombinations = 1;
  for (size_t s : steps) totalCombinations *= s;

  for (size_t i = 0; i < totalCombinations; ++i) {
    size_t carry = i;
    for (size_t j = 0; j < paramCount; ++j) {
      indices[j] = carry % steps[j];
      carry /= steps[j];
      variables[j] = params[j].minValue + indices[j] * params[j].stepValue;
    }

    TunningFunctionResult result = interface->evaluate(variables);
    if (!result.empty()) {
      float score = result[0];
      if (score < bestScore) {
        LOG("[OPT] Epoch %lu loss %f\n", i, score);
        bestScore           = score;
        best.variables      = variables;
        best.expectedResult = result;
        best.iterations     = totalCombinations;
      }
    }
  }

  return best;
}

TunningResult optimize_random(const TunningConfigurtion& config, std::vector<float>& variables) {
  const auto& interface = config.interface;
  const auto& params    = interface->parameters;

  TunningResult best;
  float         bestScore = std::numeric_limits<float>::max();

  const size_t trials = config.rand_maxIterations;

  for (size_t i = 0; i < trials; ++i) {
    TunningFunctionVariables candidate = initialize_random(config);
    TunningFunctionResult    result    = interface->evaluate(candidate);

    if (!result.empty()) {
      float score = result[0];
      if (score < bestScore) {
        LOG("[OPT] Epoch %lu loss %f\n", i, score);
        bestScore           = score;
        best.variables      = candidate;
        best.expectedResult = result;
        best.iterations     = trials;
      }
    }
  }

  return best;
}

TunningResult optimize_newton(const TunningConfigurtion& config, std::vector<float>& current) {
  const auto& interface = config.interface;
  const auto& params    = interface->parameters;

  const size_t maxIterations = config.sgd_maxIterations;
  const float  alpha         = config.sgd_learningRate;
  const float  momentum      = config.sgd_momentum;

  const size_t dim = params.size();

  std::vector<float>       velocity(dim, 0.0f);
  float                    bestLoss = std::numeric_limits<float>::max();
  TunningFunctionVariables bestVars;
  TunningFunctionResult    bestResult;

  size_t iter;
  for (iter = 0; iter < maxIterations; ++iter) {
    TunningFunctionResult eval = interface->evaluate(current);
    if (eval.size() < dim + 1) break;

    float loss = eval[0];

    if (loss < bestLoss) {
      LOG("[OPT] Epoch %lu loss %f\n", iter, loss);
      bestLoss   = loss;
      bestVars   = current;
      bestResult = eval;
    }

    std::vector<float> grad(dim);
    for (size_t i = 0; i < dim; ++i)
      grad[i] = eval[i + 1];

    std::vector<float> hessianDiag(dim, 1.0f);
    const float        h = 1e-3f;
    for (size_t i = 0; i < dim; ++i) {
      TunningFunctionVariables forward  = current;
      TunningFunctionVariables backward = current;
      forward[i] += h;
      backward[i] -= h;

      TunningFunctionResult evalF = interface->evaluate(forward);
      TunningFunctionResult evalB = interface->evaluate(backward);

      if (evalF.size() < 1 || evalB.size() < 1) continue;

      float second   = (evalF[0] - 2 * eval[0] + evalB[0]) / (h * h);
      hessianDiag[i] = std::abs(second) > epsilon ? second : 1.0f;
    }

    bool converged = true;
    for (size_t i = 0; i < dim; ++i) {
      float delta = -alpha * grad[i] / hessianDiag[i];
      velocity[i] = momentum * velocity[i] + delta;

      float newVal = current[i] + velocity[i];
      newVal       = std::max(params[i].minValue, std::min(params[i].maxValue, newVal));

      if (std::abs(newVal - current[i]) > epsilon)
        converged = false;

      current[i] = newVal;
    }

    if (converged)
      break;
  }

  return TunningResult{
    .variables      = bestVars,
    .expectedResult = bestResult,
    .iterations     = iter};
}

TunningResult optimize_binary(const TunningConfigurtion& config, std::vector<float>& current) {
  return {};
}

TunningResult optimize(const TunningConfigurtion& config) {
  TunningFunctionVariables variables = initialize(config);

  switch (config.mode) {
    case TUNNING_MODE_GRID: return optimize_grid(config, variables);
    case TUNNING_MODE_RANDOM: return optimize_random(config, variables);
    case TUNNING_MODE_NEWTON: return optimize_newton(config, variables);
    case TUNNING_MODE_BINARY: return optimize_binary(config, variables);
    default: return {};
  }
}

//=========================[UNIT]===================================

class TestFunction5D : public TunningInterface {
  public:
  std::vector<float> target = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};

  TestFunction5D() {
    parameters.resize(5);
    for (int i = 0; i < 5; ++i) {
      parameters[i].minValue  = -10.0f;
      parameters[i].maxValue  = 10.0f;
      parameters[i].stepValue = 2.0f;
    }
  }

  TunningFunctionResult evaluate(const TunningFunctionVariables& x) override {
    assert(x.size() == 5);
    TunningFunctionResult result;
    float                 value = 0.0f;

    for (int i = 0; i < 5; ++i) {
      float diff = x[i] - target[i];
      value += diff * diff;
    }
    result.push_back(value);

    for (int i = 0; i < 5; ++i) {
      result.push_back(2.0f * (x[i] - target[i]));
    }

    return result;
  }
};

void tunning_unit_test_experiment(TunningMode mode, TunningStart start) {

  auto interface = std::make_shared<TestFunction5D>();

  TunningConfigurtion config;
  config.interface = interface;
  config.mode      = mode;
  config.start     = start;

  TunningResult result = optimize(config); // Calls your optimizer

  std::cout << "Optimized Variables:\n";
  for (float v : result.variables) std::cout << v << " ";
  std::cout << "\nFunction Value: " << result.expectedResult[0] << std::endl;
}

void tunning_unit_test() {
  printf("=========[TESTING NEWTON TUNNING OPTIMIZER]===========\n");
  tunning_unit_test_experiment(TUNNING_MODE_NEWTON, TUNNING_START_RANDOM);
  printf("=========[TESTING RANDOM TUNNING OPTIMIZER]===========\n");
  tunning_unit_test_experiment(TUNNING_MODE_RANDOM, TUNNING_START_RANDOM);
  printf("=========[TESTING GRID TUNNING OPTIMIZER]===========\n");
  tunning_unit_test_experiment(TUNNING_MODE_GRID, TUNNING_START_RANDOM);
}
