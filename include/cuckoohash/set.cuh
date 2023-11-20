#if !defined(CUCKOOHASH_SET_H_)
#define CUCKOOHASH_SET_H_

#include <thrust/device_vector.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cuckoohash/builtin.hpp"
#include "cuckoohash/hash.cuh"
#include "cuckoohash/slot.cuh"

namespace cuckoohash::set {

namespace detail {

template <const std::size_t T, const std::size_t C, const std::size_t U>
__device__ bool insert(slot::SlotView<T, C>& slot, std::uint32_t rehash, std::uint32_t key) {
  std::size_t reloc = 0;
  while (reloc < U) {
    auto hash = hash::Hash(rehash + reloc % T)(key);
    auto old = atomicExch(&slot.keys[reloc % T][hash % C], static_cast<int32_t>(key));
    if (old == slot::kEmpty || old == key) {
      return true;
    }
    ++reloc;
    key = old;
  }
  return false;
}

template <const std::size_t T, const std::size_t C, const std::size_t U>
__global__ void insert(slot::SlotView<T, C> slot, std::uint32_t rehash, std::uint32_t* colide,
                       const uint32_t* keys, const std::size_t size) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size && *colide == 0;
       i += blockDim.x * gridDim.x) {
    if (!insert<T, C, U>(slot, rehash, keys[i])) {
      atomicAdd(colide, 1);
    }
  }
}

template <const std::size_t T, const std::size_t C>
__device__ bool lookup(slot::SlotView<T, C>& slot, std::uint32_t rehash, std::uint32_t key) {
  for (std::size_t i = 0; i < T; ++i) {
    auto hash = hash::Hash(rehash + i)(key);
    if (slot.keys[i][hash % C] == key) {
      return true;
    }
  }
  return false;
}

template <const std::size_t T, const std::size_t C>
__global__ void lookup(slot::SlotView<T, C> slot, std::uint32_t rehash, bool* result,
                       const uint32_t* keys, const std::size_t size) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    result[i] = lookup<T, C>(slot, rehash, keys[i]);
  }
}

}  // namespace detail

template <const std::size_t T, const std::size_t C, const std::size_t U>
class Set {
 public:
  void insert(std::vector<std::uint32_t>& keys) {
    auto keys_device = thrust::device_vector<std::uint32_t>(keys.begin(), keys.end());
    auto view = slot::SlotView<T, C>(slot_);
    for (std::size_t i = 0; i < builtin::kRehashLimit; ++i) {
      std::uint32_t* colide;
      cudaMalloc(&colide, sizeof(std::uint32_t));
      cudaMemset(colide, 0, sizeof(std::uint32_t));
      detail::insert<T, C, U>
          <<<(keys_device.size() + builtin::kBlockSize - 1) / builtin::kBlockSize,
             builtin::kBlockSize>>>(view, rehash_ + static_cast<std::uint32_t>(i), colide,
                                    keys_device.data().get(), keys_device.size());
      std::uint32_t colide_host;
      cudaMemcpy(&colide_host, colide, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
      cudaFree(colide);
      if (colide_host == 0) {
        rehash_ += i;
        return;
      }
    }
    throw std::runtime_error("rehash limit exceeded");
  }

  std::vector<bool> lookup(std::vector<std::uint32_t>& keys) {
    auto keys_device = thrust::device_vector<std::uint32_t>(keys.begin(), keys.end());
    auto result = std::vector<bool>(keys.size());
    auto result_device = thrust::device_vector<bool>(keys.size());
    auto view = slot::SlotView<T, C>(slot_);
    detail::lookup<T, C><<<(keys_device.size() + builtin::kBlockSize - 1) / builtin::kBlockSize,
                           builtin::kBlockSize>>>(view, rehash_, result_device.data().get(),
                                                  keys_device.data().get(), keys_device.size());
    thrust::copy(result_device.begin(), result_device.end(), result.begin());
    return result;
  }

 private:
  slot::Slot<T, C> slot_;
  std::uint32_t rehash_ = 0;
};

}  // namespace cuckoohash::set

#endif  // CUCKOOHASH_SET_H_
