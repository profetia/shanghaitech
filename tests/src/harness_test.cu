
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "cuckoohash/set.cuh"
#include "random.cuh"

template <std::size_t T, std::size_t C, std::size_t U, std::size_t S>
void harness_test() {
  auto source = cuckoohash_test::random::random_vector(S);
  auto lookup = cuckoohash_test::random::shuffle(source);
  lookup = cuckoohash_test::random::blend(lookup, 0.4);

  auto cuckoo = cuckoohash::set::Set<T, C, U>();
  auto set = std::unordered_set<std::uint32_t>();

  cuckoo.insert(source);
  set.insert(source.begin(), source.end());

  auto cuckoo_result = cuckoo.lookup(lookup);
  for (auto i = 0; i < S; ++i) {
    auto expected = set.find(lookup[i]) != set.end();
    auto actual = cuckoo_result[i];
    if (expected != actual) {
      std::cout << "expected: " << expected << ", actual: " << actual << std::endl;
    }
    assert(expected == actual);
  }

  std::cout << "[PASSED] - " << __FUNCTION__ << "<" << T << ", " << C << ", " << U << ", " << S
            << ">" << std::endl;
}

int main() {
  harness_test<2, 1 << 5, 4 * 4, 1 << 4>();
  harness_test<2, 1 << 10, 4 * 9, 1 << 9>();
  harness_test<2, 1 << 15, 4 * 14, 1 << 14>();
  harness_test<2, 1 << 20, 4 * 19, 1 << 19>();
  harness_test<2, 1 << 25, 4 * 24, 1 << 24>();
  return 0;
}