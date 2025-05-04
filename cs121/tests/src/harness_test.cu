
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "benchmark.cuh"
#include "cuckoohash/set.cuh"
#include "random.cuh"

template <std::size_t T, std::size_t C, std::size_t U, std::size_t S>
void harness_test() {
  auto source = cuckoohash_test::random::random_vector(1 << S);
  auto lookup = cuckoohash_test::random::shuffle(source);
  lookup = cuckoohash_test::random::blend(lookup, 0.4);

  auto cuckoo = cuckoohash::set::Set<T, 1 << C, U << 2>();
  auto set = std::unordered_set<std::uint32_t>();

  auto source_device = cuckoohash_test::benchmark::to_device(source);
  auto lookup_device = cuckoohash_test::benchmark::to_device(lookup);
  cuckoo.insert(source_device);
  auto result_device = cuckoo.lookup(lookup_device);
  auto result = cuckoohash_test::benchmark::to_host(result_device);

  set.insert(source.begin(), source.end());
  for (auto i = 0; i < 1 << S; ++i) {
    auto expected = set.find(lookup[i]) != set.end();
    auto actual = result[i];
    if (expected != actual) {
      std::cout << "expected: " << expected << ", actual: " << actual << std::endl;
    }
    assert(expected == actual);
  }

  std::cout << "[PASSED] - " << __FUNCTION__ << "<" << T << ", " << C << ", " << U << ", " << S
            << ">" << std::endl;
}

int main() {
  harness_test<2, 5, 4, 4>();
  harness_test<2, 10, 9, 9>();
  harness_test<2, 15, 14, 14>();
  harness_test<2, 20, 19, 19>();
  harness_test<2, 25, 24, 24>();
  return 0;
}