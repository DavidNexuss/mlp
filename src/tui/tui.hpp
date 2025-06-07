#pragma once
#include <GL/gl.h>
#include <memory>

struct IApplication {
  virtual void render() = 0;
  virtual void update() = 0;
  virtual void init()   = 0;

  virtual ~IApplication() {}
};

struct Engine {
  virtual void start(std::shared_ptr<IApplication> application) = 0;
  virtual ~Engine() {}
};

std::shared_ptr<Engine> engineCreate();
