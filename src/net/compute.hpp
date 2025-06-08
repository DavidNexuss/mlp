#pragma once
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cuda.h>

template <typename T>
class cuda_vector {
  T*     data_;
  size_t size_;
  size_t capacity_;

  public:
  __host__ cuda_vector(size_t initial_capacity = 0) :
    data_(nullptr), size_(0), capacity_(0) {
    if (initial_capacity > 0) {
      reserve(initial_capacity);
    }
  }

  __host__ ~cuda_vector() {
    if (data_) {
      cudaFree(data_);
    }
  }

  cuda_vector(const cuda_vector&)            = delete;
  cuda_vector& operator=(const cuda_vector&) = delete;

  __host__ cuda_vector(cuda_vector&& other) :
    data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    other.data_     = nullptr;
    other.size_     = 0;
    other.capacity_ = 0;
  }

  __host__ cuda_vector& operator=(cuda_vector&& other) {
    if (this != &other) {
      if (data_) cudaFree(data_);
      data_           = other.data_;
      size_           = other.size_;
      capacity_       = other.capacity_;
      other.data_     = nullptr;
      other.size_     = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  __host__ void reserve(size_t new_capacity) {
    if (new_capacity <= capacity_) return;

    T* new_data;
    cudaMalloc(&new_data, new_capacity * sizeof(T));
    if (data_) {
      cudaMemcpy(new_data, data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
      cudaFree(data_);
    }

    data_     = new_data;
    capacity_ = new_capacity;
  }

  __host__ void push_back(const T& value) {
    if (size_ == capacity_) {
      reserve(capacity_ > 0 ? capacity_ * 2 : 4);
    }
    cudaMemcpy(data_ + size_, &value, sizeof(T), cudaMemcpyHostToDevice);
    ++size_;
  }

  __host__ __device__ T& operator[](size_t index) {
    return data_[index];
  }

  __host__ __device__ const T& operator[](size_t index) const {
    return data_[index];
  }

  __host__ __device__ size_t size() const {
    return size_;
  }

  __host__ __device__ T* data() {
    return data_;
  }

  __host__ __device__ const T* data() const {
    return data_;
  }
};
