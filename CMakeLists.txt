cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

project(mlp)

find_package(OpenMP)

add_compile_options(-O3 -ftree-vectorize -march=native)
include_directories(lib)
include_directories(src)

file(GLOB_RECURSE SRC 
  src/*.cpp
)

add_executable(mlp ${SRC})
target_include_directories(mlp PUBLIC src)
target_link_libraries(mlp OpenMP::OpenMP_CXX glfw )
