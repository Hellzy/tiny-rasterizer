#include "device_lock.hh"

DeviceLock::DeviceLock()
{
    cudaMalloc(&lock_, sizeof(int));
    cudaMemset(lock_, 0, sizeof(int));
}

DeviceLock::~DeviceLock()
{
    cudaFree(lock_);
}

__device__
void DeviceLock::lock()
{
    while (atomicCAS(lock_, 0, 1) != 0)
        continue;
}

__device__
void DeviceLock::unlock()
{
    atomicExch(lock_, 0);
}
