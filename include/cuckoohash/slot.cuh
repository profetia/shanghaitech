#ifndef CUCKOOHASH_SLOT_H_
#define CUCKOOHASH_SLOT_H_

#include <thrust/device_vector.h>

#include <cstddef>
#include <cstdint>

namespace cuckoohash::slot {

constexpr std::int32_t kEmpty = -1;

template <const std::size_t T, const std::size_t C>
struct Slot {
  thrust::device_vector<std::int32_t> keys[T];

  Slot() {
    for (std::size_t i = 0; i < T; ++i) {
      keys[i] = thrust::device_vector<std::int32_t>(C, kEmpty);
    }
  }
};

template <const std::size_t T, const std::size_t C>
struct SlotView {
  std::int32_t* keys[T];

  explicit SlotView(Slot<T, C>& slot) {
    for (std::size_t i = 0; i < T; ++i) {
      keys[i] = slot.keys[i].data().get();
    }
  }
};

}  // namespace cuckoohash::slot

#endif  // CUCKOOHASH_SLOT_H_
