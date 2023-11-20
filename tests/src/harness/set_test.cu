#include <thrust/device_vector.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "cuckoohash/set.cuh"
#include "random.cuh"

int main() {
  constexpr std::size_t kT = 2;
  constexpr std::size_t kC = 1 << 5;
  constexpr std::size_t kS = 1 << 4;
  constexpr std::size_t kU = 4 * 4;

  auto source = cuckoohash_test::random_vector(kS);
  auto lookup = cuckoohash_test::shuffle(source);
  lookup = cuckoohash_test::blend(lookup, 0.4);

  auto cuckoo = cuckoohash::set::Set<kT, kC, kU>();
  auto set = std::unordered_set<std::uint32_t>();

  cuckoo.insert(source);
  set.insert(source.begin(), source.end());

  auto cuckoo_result = cuckoo.lookup(lookup);
  for (auto i = 0; i < kS; ++i) {
    auto expected = set.find(lookup[i]) != set.end();
    auto actual = cuckoo_result[i];
    assert(expected == actual);
  }

  std::cout << "[PASSED] - " << __FILE__ << std::endl;

  return 0;
}