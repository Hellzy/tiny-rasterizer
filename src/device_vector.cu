#include <cstdlib>
#include <cstring>

#include "device_vector.hh"

DeviceVector::DeviceVector()
{
    cudaMalloc(&data_, sizeof(size_t) * max_size_);
}

DeviceVector::~DeviceVector()
{
    cudaFree(data_);
}

__device__
void DeviceVector::push(size_t val)
{
    lock_.lock();
    if (cur_size_ == max_size_)
        expand();

    data_[cur_size_++] = val;
    lock_.unlock();
}

__device__
void DeviceVector::expand()
{
    max_size_ *= 2;
    size_t* new_ptr = static_cast<size_t*>(malloc(sizeof(size_t) * max_size_));
    memcpy(new_ptr, data_, sizeof(size_t) * cur_size_);
    free(data_);
    data_ = new_ptr;
}
