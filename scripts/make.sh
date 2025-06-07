#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}-- $1${NC}"; }
doneok()  { echo -e "${GREEN}✔ $1${NC}"; }
error() { echo -e "${RED}✖ $1${NC}"; }

make_dir() {
  if [ ! -d "$1" ]; then
    mkdir -p "$1"
    doneok "Created directory: $1"
  else
    info "Directory exists: $1"
  fi
}

make_symlink() {
  if [ ! -e "$1" ]; then
    ln -s "$2" "$1"
    doneok "Linked $1 → $2"
  else
    info "Link exists: $1"
  fi
}

needs_reconfigure() {
  local build_dir="$1"
  local cache_file="$build_dir/CMakeCache.txt"
  if [ ! -f "$cache_file" ]; then
    return 0
  fi
  for f in CMakeLists.txt $(find . -maxdepth 1 -name "*.cmake" -o -name "*.txt"); do
    if [ "$f" -nt "$cache_file" ]; then
      return 0
    fi
  done
  return 1
}

configure_cmake() {
  local dir="$1"
  local type="$2"
  info "Configuring CMake ($type)..."
  cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE="$type" > /dev/null && 
  doneok "CMake configured for $type" || error "CMake failed to be configured for $type"
}

# Execution
make_dir build/release
make_dir build/debug

configure_cmake build/release Release
configure_cmake build/debug Debug

make_symlink build/release/assets ../../assets
make_symlink build/debug/assets ../../assets

make_symlink compile_commands.json build/release/compile_commands.json
