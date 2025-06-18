#pragma once
#include <chrono>
#include <string>
#include <iostream>

class Timer {
  public:
  void tic() {
    start = std::chrono::high_resolution_clock::now();
  }

  double toc() const {
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
  }

  private:
  std::chrono::high_resolution_clock::time_point start;
};


struct TimerRegion {
  std::string name;
  Timer       timer;

  TimerRegion(const std::string& name) :
    name(name) {
    timer.tic();
  }

  ~TimerRegion() {
    std::cerr << name << " took: " << timer.toc() << "ms" << std::endl;
  }
};
