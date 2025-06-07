#include <filesystem>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <random>
#include <map>
#include <functional>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include "net.h"

#ifdef __linux__
#  define MMAP_READ
#endif

#ifdef MMAP_READ
#  include <sys/mman.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#endif


namespace fs = std::filesystem;

struct LazyDataSet : public DataSetStorage<std::string, std::vector<float>> {
  std::function<std::vector<float>(const std::string&)> loadDataFunctor;
  std::map<int, std::vector<float>>                     cache;

  virtual const std::vector<float>& getInput(int index) override {
    auto it = cache.find(index);
    if (it == cache.end())
      return cache[index] = loadDataFunctor(inputs[index]);
    return it->second;
  }
  virtual const std::vector<float>& getOutput(int index) override {
    return targets[index];
  }
};


#ifdef MMAP_READ
static std::vector<float> loadimageMMAP(const std::string& filepath) {
  int width, height, channels;

  int fd = open(filepath.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cerr << "Failed to open image file: " << filepath << "\n";
    return {};
  }

  struct stat sb;
  if (fstat(fd, &sb) < 0) {
    std::cerr << "Failed to get file size: " << filepath << "\n";
    close(fd);
    return {};
  }

  size_t fileSize = sb.st_size;

  void* mappedData = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd); // fd can be closed after mmap
  if (mappedData == MAP_FAILED) {
    std::cerr << "mmap failed for file: " << filepath << "\n";
    return {};
  }

  unsigned char* data = stbi_load_from_memory(
    reinterpret_cast<unsigned char*>(mappedData), static_cast<int>(fileSize),
    &width, &height, &channels, 1);

  munmap(mappedData, fileSize); // Unmap after stbi_load_from_memory has copied what it needs

  if (!data) {
    std::cerr << "stbi_load_from_memory failed: " << filepath << "\n";
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

#endif

static std::vector<float> loadimageRaw(const std::string& filepath) {
  int            width, height, channels;
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

#ifdef MMAP_READ
static auto loadimage = loadimageMMAP;
#else
static auto loadimage = loadimageRaw;
#endif

std::shared_ptr<DataSet> createStorageDataSet(const std::string& filepath) {
  std::cout << "Reading large dataset..." << std::endl;
  auto dataset = std::make_shared<LazyDataSet>();

  dataset->loadDataFunctor = loadimage;


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


      dataset->inputs.push_back(filePathStr);
      dataset->targets.push_back(tag);
      filecount++;
    }

    std::cout << i << " # filecount = " << filecount << " w: " << width << " h: " << height << " c: " << channels << std::endl;
  }

  std::cout << "Shuffling " << std::endl;

  dataset->shuffle();
  int n = 200;
  std::cout << "Trimming to " << n << std::endl;
  dataset->trim(n);

  std::cout << "Done" << std::endl;

  return dataset;
}
