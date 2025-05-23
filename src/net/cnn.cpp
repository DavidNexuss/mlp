#include "cnn.hpp"

struct CNNLayer {

  struct Filter {
    Tensor<float> kernel;
    float         bias;
  };
  std::vector<Filter> filters;

  ivec size;
  int  padding;

  void Propagate(Tensor<float>& input, Tensor<float>& output) {
    for (int t = 0; t < filters.size(); t++) {
      auto& filter = filters[t];

      int rx = filter.kernel.width() / 2;
      int ry = filter.kernel.height() / 2;

      for (int i = rx; i < size.x - rx; i++) {
        for (int j = ry; j < size.y - ry; j++) {

          float accum = 0.0f;

          for (int ii = 0; ii < filter.kernel.width(); ii++) {
            for (int jj = 0; jj < filter.kernel.height(); jj++) {
              accum += filter.kernel.at2D(ii, jj) * input.at2D(i + ii - rx, j + jj - ry);
            }
          }
          output.at3D(i, j, t) = accum + filter.bias;
        }
      }
    }
  }
};


struct CNNImpl : public CNN {
  std::vector<CNNLayer> layers;

  virtual void AddLayer(CNNLayerCreateInfo ci) override {
    layers.emplace_back(ci);
  }

  virtual Vector Propagate(Vector input) override {
    Tensor<float> ping[2];

    for (int i = 0; i < layers.size(); i++) {
      Tensor<float>& input  = i == 0 ? *input : ping[0];
      Tensor<float>& output = ping[1];

      layers[i].Propagate(input, output);

      std::swap(input, output);
    }

    return std::make_shared<Tensor<float>>(std::move(ping[0]));
  }
  virtual float ComputeLoss(Vector predicted, Vector target, LossFunction loss) override {}
  virtual void  Backpropagate(Vector input, Vector target, LossFunction loss) override {}
  virtual void  TrainStep(Vector input, Vector target, LossFunction loss) override {}
  virtual void  SetOptimizer(const OptimizerCreateInfo ci) override {}
};
