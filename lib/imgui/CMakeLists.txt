cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ftree-vectorize -march=native")
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
set(CMAKE_CXX_STANDARD 17)

project(imgui)

include_directories(include)
include_directories(.)


file(GLOB_RECURSE SRC 
  *.cpp backends/*.cpp
)

add_library(imgui ${SRC})
