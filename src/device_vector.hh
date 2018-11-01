#pragma once
#include <cstdint>

#include "device_lock.hh"

class DeviceVector
{
public:

    /**
     * Empty handlers used to construct the object on CPU side, then copy
     * it to the gpu.
     */
    DeviceVector();
    ~DeviceVector();

    __device__ void push(size_t val);
    __device__ constexpr size_t operator[](size_t idx) const { return data_[idx]; }

    /**
     * Simple getter to know how much data we stored
     */
    __device__ size_t size() const { return cur_size_; }

private:
    __device__ void expand();

private:
    size_t cur_size_ = 0;
    size_t max_size_ = 10;
    size_t *data_;
    mutable device_lock_t lock_;
};

using device_vec_t = DeviceVector;
