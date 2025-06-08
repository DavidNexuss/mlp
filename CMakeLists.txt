cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ftree-vectorize -march=native")
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
set(CMAKE_CXX_STANDARD 17)

project(mlp LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 61) 
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit)


include_directories(${CUDA_INCLUDE_DIRS})
include_directories(lib)
include_directories(src)

file(GLOB_RECURSE SRC 
  src/*.cpp src/*.cu
)

add_executable(tui ${SRC} src/tui.cc)
target_include_directories(tui PUBLIC src)
target_link_libraries(tui PUBLIC glfw OpenMP::OpenMP_CXX cuda)

add_executable(cli ${SRC} src/cli.cc)
target_include_directories(cli PUBLIC src)

target_link_libraries(cli PUBLIC glfw OpenMP::OpenMP_CXX cuda)
