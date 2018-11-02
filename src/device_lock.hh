#pragma once
#include <cstdint>

/**
 * Simple wrapper around atomicCAS and atomicExch to simulate mutual
 * exclusion between blocks.
 */

class DeviceLock
{
public:
    __host__ DeviceLock();
    __host__ ~DeviceLock();

    __device__ void lock();
    __device__ void unlock();

private:
        int* lock_ = nullptr;
};

using device_lock_t = DeviceLock;
