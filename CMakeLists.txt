cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ftree-vectorize -march=native")
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
set(CMAKE_CXX_STANDARD 17)

project(mlp)

include_directories(lib)
include_directories(src)

find_package(OpenMP)
if (FALSE)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB_RECURSE SRC 
  src/*.cpp
)

add_subdirectory(lib/glfw)
add_subdirectory(lib/tracy)

add_executable(tui ${SRC} src/tui.cc)
target_include_directories(tui PUBLIC src)
target_link_libraries(tui PUBLIC glfw TracyClient)
target_compile_definitions(tui PRIVATE TRACY_ENABLE)

add_executable(cli ${SRC} src/cli.cc)
target_include_directories(cli PUBLIC src)

target_link_libraries(cli PUBLIC glfw TracyClient)
target_compile_definitions(cli PRIVATE TRACY_ENABLE)
