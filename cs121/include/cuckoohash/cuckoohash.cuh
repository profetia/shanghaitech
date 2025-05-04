#if !defined(CUCKOOHASH_CUCKOOHASH_H_)
#define CUCKOOHASH_CUCKOOHASH_H_

#include "cuckoohash/set.cuh"

namespace cuckoohash {

template <const std::size_t T, const std::size_t C, const std::size_t U>
using Set = set::Set<T, C, U>;

}  // namespace cuckoohash

#endif  // CUCKOOHASH_CUCKOOHASH_H_
