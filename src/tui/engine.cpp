#include "tui.hpp"
#include "vlWindow.hpp"
#include <core/debug.hpp>
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>

struct EngineImpl : public Engine {
  void start(std::shared_ptr<IApplication> application) override {
    if (!glewInit()) {
      LOG("Failed to initialize glew\n");
      return;
    }
    if (!glfwInit()) {
      LOG("Failed to initialize glfw\n");
      return;
    }

    vlWindowCreateInfo ci;
    ci.name     = "MLP atuo";
    ci.height   = 680;
    ci.width    = 720;
    auto window = vlWindowCreate(ci);
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window->getInternal(), true);

    const char* glsl_version = "#version 330";
    ImGui_ImplOpenGL3_Init(glsl_version);

    application->init();

    while (!window->shouldClose() && !window->isKeyPressed(GLFW_KEY_ESCAPE)) {
      application->update();
      if (window->shouldRender()) {
        glViewport(0, 0, window->getFBOWidth(), window->getFBOHeight());
        glClearColor(0.3, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        application->render();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      }
      window->swapBuffers();
      window->pollEvents();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
  }
};

std::shared_ptr<Engine> engineCreate() {
  return std::shared_ptr<Engine>(new EngineImpl());
}
