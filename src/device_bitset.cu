#include "device_bitset.hh"

bool* DeviceBitset::mem_ = nullptr;
size_t DeviceBitset::chunk_size_ = 0;
size_t DeviceBitset::mem_idx_ = 0;

DeviceBitset::DeviceBitset()
{
    bits_ = mem_ + mem_idx_ * chunk_size_;
    ++mem_idx_;
}

void DeviceBitset::allocate(size_t nb, size_t size)
{
    cudaMalloc(&mem_, sizeof(bool) * nb * size);
    cudaMemset(mem_, 0, sizeof(bool) * nb * size);
    chunk_size_ = size;
}

void DeviceBitset::release_memory()
{
    cudaFree(mem_);
    mem_idx_ = 0;
    chunk_size_ = 0;
}

__device__ void DeviceBitset::set(size_t idx)
{
    bits_[idx] = true;
}
__device__ bool DeviceBitset::test(size_t idx) const
{
    return bits_[idx];
}
