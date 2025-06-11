#include <tui/tui.hpp>
#include <imgui/imgui.h>

struct Application : public IApplication {
  bool show = true;

  void render() override {
    ImGui::ShowDemoWindow(&show);
  }

  void update() override {}
  void init() override {}
};


int main() {
  engineCreate()->start(std::shared_ptr<IApplication>(new Application));
}
