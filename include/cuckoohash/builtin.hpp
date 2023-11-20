#ifndef CUCKOOHASH_BUILTIN_H_
#define CUCKOOHASH_BUILTIN_H_

#include <cstddef>

namespace cuckoohash::builtin {

constexpr std::size_t kBlockSize = 256;

constexpr std::size_t kRehashLimit = 10;

}  // namespace cuckoohash::builtin

#endif  // CUCKOOHASH_BUILTIN_H_
