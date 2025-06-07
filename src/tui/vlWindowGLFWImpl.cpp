#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <core/debug.hpp>
#include <unordered_map>
#include "vlWindow.hpp"

struct vlWindowImpl : public vlWindow {
  bool                         shouldRecreateSwapChain;
  GLFWwindow*                  window;
  int                          width;
  int                          height;
  int                          fboWidth;
  int                          fboHeight;
  int                          inputX;
  int                          inputY;
  float                        inputScrollX;
  float                        inputScrollY;
  std::unordered_map<int, int> inputPressedKeys;
  std::unordered_map<int, int> inputMouseKeys;

  static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    vlWindowImpl* user = (vlWindowImpl*)glfwGetWindowUserPointer(window);
    if (action == GLFW_PRESS)
      user->inputPressedKeys[key] = 1;
    if (action == GLFW_REPEAT)
      user->inputPressedKeys[key]++;
    if (action == GLFW_RELEASE)
      user->inputPressedKeys[key] = 0;
  }

  static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    vlWindowImpl* user = (vlWindowImpl*)glfwGetWindowUserPointer(window);
    user->inputX       = xpos;
    user->inputY       = ypos;
  }

  static void window_size_callback(GLFWwindow* window, int width, int height) {
    vlWindowImpl* user = (vlWindowImpl*)glfwGetWindowUserPointer(window);
    user->width        = width;
    user->height       = height;
  }

  static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    vlWindowImpl* user           = (vlWindowImpl*)glfwGetWindowUserPointer(window);
    user->inputMouseKeys[button] = action == GLFW_PRESS;
  }

  static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    vlWindowImpl* user = (vlWindowImpl*)glfwGetWindowUserPointer(window);
    user->inputScrollX += xoffset;
    user->inputScrollY += yoffset;
  }

  static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    vlWindowImpl* user = (vlWindowImpl*)glfwGetWindowUserPointer(window);
    user->fboWidth     = width;
    user->fboHeight    = height;
  }

  void setWidth(int width) override {}
  void setHeight(int height) override {}
  int  getWidth() override { return width; }
  int  getHeight() override { return height; }
  int  getFBOWidth() override { return fboWidth; }
  int  getFBOHeight() override { return fboHeight; }
  void setPos(int x, int y) override {}
  int  getX() override { return inputX; }
  int  getY() override { return inputY; }
  void begin() override {}
  void flush() override {}
  int  shouldClose() override { return glfwWindowShouldClose(window); }

  bool update() override {
    glfwPollEvents();
    return false;
  }

  float getScrollX() override {
    return inputScrollX;
  }

  float getScrollY() override {
    return inputScrollY;
  }

  int isMousePressed(int mousecode) override {
    return inputMouseKeys[mousecode];
  }

  int isKeyPressed(int keycode) override { return inputPressedKeys[keycode]; }

  ~vlWindowImpl() {
    glfwDestroyWindow(window);
    LOG("[VL] Window dispose\n");
  }

  vlWindowImpl(vlWindowCreateInfo params) {
    window = glfwCreateWindow(params.width, params.height, params.name.c_str(), 0, 0);

    glfwGetFramebufferSize(window, &fboWidth, &fboHeight);
    printf("[VL] vlWindow - Initial fbo window size: %d %d \n", fboWidth, fboHeight);

    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    LOG("[VL] Window created\n");
  }
};

std::shared_ptr<vlWindow> vlWindowCreate(vlWindowCreateInfo windowCI) {
  vlWindowImpl* impl = new vlWindowImpl(windowCI);

  return std::shared_ptr<vlWindow>(new vlWindowImpl(windowCI));
}
