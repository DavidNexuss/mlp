#include <filesystem>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include "net.h"

namespace fs = std::filesystem;

struct StorageDataset : public DataSet {
  int maxElements;

  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<float>> outputs;

  virtual std::vector<float>& getInput(int index) override {
    return inputs[index];
  }
  virtual std::vector<float>& getOutput(int index) override {
    return outputs[index];
  }
  virtual int getInputCount() override {
    if (maxElements != 0)
      return std::min(maxElements, (int)inputs.size());
    return inputs.size();
  }
  virtual int getOutputCount() override {
    if (maxElements != 0)
      return std::min(maxElements, (int)outputs.size());
    return outputs.size();
  }

  void shuffle() {
    static std::random_device rd;
    static std::mt19937       g(rd());

    for (int i = (int)inputs.size() - 1; i > 0; i--) {
      std::uniform_int_distribution<int> dist(0, i);

      int j = dist(g);

      std::swap(inputs[i], inputs[j]);
      std::swap(outputs[i], outputs[j]);
    }
  }
};

static std::vector<float> loadimage(const std::string& filepath, int& width, int& height, int& channels) {
  unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, 1);
  if (!data) {
    std::cerr << "Failed to load image: " << filepath << "\n";
    return {};
  }

  std::vector<float> imgFloats;
  imgFloats.reserve(width * height);
  for (int i = 0; i < width * height; ++i) {
    imgFloats.push_back(data[i] / 255.0f);
  }

  stbi_image_free(data);
  return imgFloats;
}

std::shared_ptr<DataSet> createStorageDataSet(const std::string& filepath) {
  std::cout << "Reading large dataset..." << std::endl;
  auto dataset = std::make_shared<StorageDataset>();

  auto makevector = [](int index, int size) {
    std::vector<float> v(size, 0);
    v[index] = 1;
    return v;
  };

  int tagcount = 0;
  for (auto& tagEntry : fs::directory_iterator(filepath)) {
    tagcount++;
  }

  std::cout << "tag count = " << tagcount << std::endl;

  int i = 0;
  for (auto& tagEntry : fs::directory_iterator(filepath)) {
    if (!tagEntry.is_directory()) continue;
    const auto& tagPath = tagEntry.path();

    auto tag       = makevector(i++, tagcount);
    int  filecount = 0;
    int  width, height, channels;

    for (auto& fileEntry : fs::directory_iterator(tagPath)) {
      if (!fileEntry.is_regular_file()) continue;
      auto ext = fileEntry.path().extension().string();
      if (ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".JPG") continue;

      std::string filePathStr = fileEntry.path().string();
      auto        inputData   = loadimage(filePathStr, width, height, channels);

      if (!inputData.empty()) {
        dataset->inputs.push_back(std::move(inputData));
        dataset->outputs.push_back(tag);
        filecount++;
      }
    }

    std::cout << i << " # filecount = " << filecount << " w: " << width << " h: " << height << " c: " << channels << std::endl;
  }

  std::cout << "Shuffling " << std::endl;

  dataset->shuffle();

  dataset->maxElements = 500;

  std::cout << "Done" << std::endl;

  return dataset;
}
