#!/bin/sh
mkdir -p build/release
mkdir -p build/debug
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build/debug   -DCMAKE_BUILD_TYPE=Debug

ln -s build/release/compile_commands.json .
