#pragma once
#include <string>
#include <GLFW/glfw3.h>
#include <memory>

struct vlWindowCreateInfo {
  int         width;
  int         height;
  std::string name;
};

struct vlWindow {

  //Getters
  virtual int   getWidth()     = 0;
  virtual int   getHeight()    = 0;
  virtual int   getFBOWidth()  = 0;
  virtual int   getFBOHeight() = 0;
  virtual int   getX()         = 0;
  virtual int   getY()         = 0;
  virtual float getScrollX()   = 0;
  virtual float getScrollY()   = 0;

  inline float getSX() { return getX() / (float)getWidth(); }
  inline float getSY() { return getY() / (float)getHeight(); }
  inline float getRA() { return getWidth() / (float)getHeight(); }

  inline bool shouldRender() { return getFBOWidth() > 0 && getFBOHeight() > 0; }

  virtual int shouldClose()                 = 0;
  virtual int isKeyPressed(int keycode)     = 0;
  virtual int isMousePressed(int mousecode) = 0;

  //setters
  virtual void setWidth(int width)   = 0;
  virtual void setHeight(int height) = 0;
  virtual void setPos(int x, int y)  = 0;

  //actions
  virtual void begin()  = 0;
  virtual void flush()  = 0;
  virtual bool update() = 0;

  virtual ~vlWindow() {}
};

std::shared_ptr<vlWindow> vlWindowCreate(vlWindowCreateInfo);
