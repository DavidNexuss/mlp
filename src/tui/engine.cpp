#include "tui.hpp"
#include "vlWindow.hpp"

struct EngineImpl : public Engine {
  void start(std::shared_ptr<IApplication> application) override {

    glfwInit();

    vlWindowCreateInfo ci;
    ci.name     = "MLP atuo";
    ci.height   = 680;
    ci.width    = 720;
    auto window = vlWindowCreate(ci);

    application->init();

    while (!window->shouldClose() && !window->isKeyPressed(GLFW_KEY_ESCAPE)) {
      application->update();
      if (window->shouldRender()) {
        application->render();
      }
    }
  }
};

std::shared_ptr<Engine> engineCreate() {
  return std::shared_ptr<Engine>(new EngineImpl());
}
