#include <tui/tui.hpp>
#include <imgui/imgui.h>
#include "testsuite.hpp"
#include <iostream>

struct ForwardTest {
  std::shared_ptr<MLP> network;

  void gui() {
  }
};

struct Application : public IApplication {
  std::vector<std::shared_ptr<MLPTrainer>> test;

  int current;

  Application() {
    test    = getTests();
    current = 1;
  }

  void render() override {
    ImGui::Begin("Network");
    test[current]->getNetwork()->GUI();
    test[current]->getNetwork()->Visualize();
    ImGui::End();
  }

  void update() override {}
  void init() override {}
};


int main() {
  engineCreate()->start(std::shared_ptr<IApplication>(new Application));
}
