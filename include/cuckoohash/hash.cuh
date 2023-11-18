// https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h

#ifndef CUCKOOHASH_HASH_H_
#define CUCKOOHASH_HASH_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cuckoohash::hash {

namespace detail {

constexpr std::uint32_t kPrime1 = 0x9E3779B1U;
constexpr std::uint32_t kPrime2 = 0x85EBCA77U;
constexpr std::uint32_t kPrime3 = 0xC2B2AE3DU;
constexpr std::uint32_t kPrime4 = 0x27D4EB2FU;
constexpr std::uint32_t kPrime5 = 0x165667B1U;

__host__ __device__ constexpr std::uint32_t rotate_left(const std::uint32_t x,
                                                        const std::uint32_t r) {
  return (x << r) | (x >> (32 - r));
}

}  // namespace detail

class Hash {
 public:
  __host__ __device__ explicit constexpr Hash(const std::uint32_t seed) : seed(seed) {}

  __host__ __device__ constexpr std::uint32_t operator()(const std::uint32_t key) const {
    return HashImpl(detail::kPrime1 * static_cast<std::uint32_t>(seed + 3) + detail::kPrime2)(key);
  }

 private:
  std::uint32_t seed;

  class HashImpl {
   public:
    __host__ __device__ explicit constexpr HashImpl(const std::uint32_t seed) : seed(seed) {}

    __host__ __device__ constexpr std::uint32_t operator()(const std::uint32_t key) const {
      std::uint32_t hash = seed + detail::kPrime5 + 4;

      hash += key * detail::kPrime3;
      hash = detail::rotate_left(hash, 17) * detail::kPrime4;

      const auto *bytes = static_cast<const std::uint8_t *>(static_cast<const void *>(&key));
      for (std::size_t i = 0; i < 4; ++i) {
        hash += bytes[i] * detail::kPrime5;
        hash = detail::rotate_left(hash, 11) * detail::kPrime1;
      }

      hash ^= hash >> 15;
      hash *= detail::kPrime2;
      hash ^= hash >> 13;
      hash *= detail::kPrime3;
      hash ^= hash >> 16;

      return hash;
    }

   private:
    std::uint32_t seed;
  };
};

}  // namespace cuckoohash::hash

#endif  // CUCKOOHASH_HASH_H_
