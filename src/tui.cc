#include <tui/tui.hpp>

struct Application : public IApplication {

  void render() override {}
  void update() override {}
  void init() override {}
};


int main() {
  engineCreate()->start(std::shared_ptr<IApplication>(new Application));
}
