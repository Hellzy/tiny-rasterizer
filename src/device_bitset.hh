#pragma once

class DeviceBitset
{
public:
    __host__ DeviceBitset();
    __host__ static void allocate(size_t nb, size_t size);
    __host__ static void release_memory();

    __device__ void set(size_t idx);
    __device__ bool test(size_t idx) const;

private:
    /* GPU memory shared by all bitsets */
    static bool* mem_;

    /* Next free available chunk of memory */
    static size_t mem_idx_;

    /* Size for a chunk of memory */
    static size_t chunk_size_;

    /* Offset in mem_ representing the bits of the current bitset */
    bool* bits_ = nullptr;
};

using bitset_t = DeviceBitset;
