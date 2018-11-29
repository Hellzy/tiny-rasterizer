#pragma once
#include <cstdint>

#include "device_lock.hh"

/**
 * A simple vector class that allows to simulate c++ vector's behaviour on
 * GPU side, using locks to allow usage between blocks.
 */

class DeviceVector
{
public:

    /**
     * Simple constructor to allocate the device memory used by the class
     */
    DeviceVector();

    /**
     * Simple destructor to release the device memory used by the class
     */
    ~DeviceVector();

    __device__ void push(size_t val);
    __device__ constexpr size_t operator[](size_t idx) const { return data_[idx]; }

    /**
     * Simple getter to know how much data we stored
     */
    __device__ size_t size() const { return cur_size_; }

private:
    /**
     * Simple getter to know how much data we stored
     */
    __device__ void expand();

private:
    size_t cur_size_ = 0;
    size_t max_size_ = 100000;
    size_t *data_;
    mutable device_lock_t lock_;
};

using device_vec_t = DeviceVector;
