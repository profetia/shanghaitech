#if !defined(CUCKOOHASH_SET_H_)
#define CUCKOOHASH_SET_H_

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

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
struct InsertImpl : public thrust::unary_function<std::uint32_t, bool> {
  InsertImpl(slot::SlotView<T, C> slot, std::uint32_t rehash)
      : slot_(std::move(slot)), rehash_(rehash) {}

  __device__ bool operator()(std::uint32_t key) const {
    std::size_t reloc = 0;
    while (reloc < U) {
      auto hash = hash::Hash(rehash_ + reloc % T)(key);
      auto old = atomicExch(&slot_.keys[reloc % T][hash % C], static_cast<int32_t>(key));
      if (old == slot::kEmpty || old == key) {
        return true;
      }
      ++reloc;
      key = old;
    }
    return false;
  }

  slot::SlotView<T, C> slot_;
  std::uint32_t rehash_;
};

template <const std::size_t T, const std::size_t C, const std::size_t U>
struct LookupImpl : public thrust::unary_function<std::uint32_t, bool> {
  LookupImpl(slot::SlotView<T, C> slot, std::uint32_t rehash)
      : slot_(std::move(slot)), rehash_(rehash) {}

  __device__ bool operator()(std::uint32_t key) const {
    for (std::size_t i = 0; i < T; ++i) {
      auto hash = hash::Hash(rehash_ + i)(key);
      if (slot_.keys[i][hash % C] == key) {
        return true;
      }
    }
    return false;
  }

  slot::SlotView<T, C> slot_;
  std::uint32_t rehash_;
};

}  // namespace detail

template <const std::size_t T, const std::size_t C, const std::size_t U>
class Set {
 public:
  void insert(std::vector<std::uint32_t>& keys) {
    auto keys_device = thrust::device_vector<std::uint32_t>(keys.begin(), keys.end());
    insert(keys_device);
  }

  void insert(thrust::device_vector<std::uint32_t>& keys) {
    auto view = slot::SlotView<T, C>(slot_);
    for (std::size_t i = 0; i < builtin::kRehashLimit; ++i) {
      bool success = thrust::transform_reduce(
          thrust::device, keys.begin(), keys.end(),
          detail::InsertImpl<T, C, U>(view, rehash_ + static_cast<std::uint32_t>(i)), true,
          thrust::logical_and<bool>());
      if (success) {
        rehash_ += i;
        return;
      }
    }
    throw std::runtime_error("rehash limit exceeded");
  }

  thrust::device_vector<bool> lookup(std::vector<std::uint32_t>& keys) {
    auto keys_device = thrust::device_vector<std::uint32_t>(keys.begin(), keys.end());
    return lookup(keys_device);
  }

  thrust::device_vector<bool> lookup(thrust::device_vector<std::uint32_t>& keys) {
    auto result = thrust::device_vector<bool>(keys.size());
    auto view = slot::SlotView<T, C>(slot_);
    thrust::transform(thrust::device, keys.begin(), keys.end(), result.begin(),
                      detail::LookupImpl<T, C, U>(view, rehash_));
    return result;
  }

  void clear() {
    slot_.clear();
    rehash_ = 0;
  }

 private:
  slot::Slot<T, C> slot_;
  std::uint32_t rehash_ = 0;
};

}  // namespace cuckoohash::set

#endif  // CUCKOOHASH_SET_H_
