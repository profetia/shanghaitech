
#include <cstddef>
#include <iostream>
#include <vector>

#include "benchmark.cuh"
#include "cuckoohash/set.cuh"
#include "random.cuh"

template <const std::size_t T, const std::size_t R>
void lookup_test(std::vector<std::uint32_t>& source, std::vector<std::uint32_t>& another,
                 cuckoohash::set::Set<T, 1 << 25, 24 << 2>& cuckoo) {
  auto lookup =
      cuckoohash_test::random::blend(source, another, static_cast<double>(100U - R * 10) / 100.0f);
  auto lookup_device = cuckoohash_test::benchmark::to_device(lookup);
  try {
    auto benchmark = cuckoohash_test::benchmark::benchmark([&]() { cuckoo.lookup(lookup_device); });
    std::cout << "[PASSED] - " << __FUNCTION__ << "<" << T << ", " << R << ">" << std::endl;
    std::cout << cuckoohash_test::benchmark::report(benchmark, 1 << 24) << std::endl;
  } catch (const std::exception& e) {
    std::cout << "[FAILED] - " << __FUNCTION__ << "<" << T << ", " << R << ">" << std::endl;
    std::cout << e.what() << std::endl;
  }
}

int main() {
  auto source = cuckoohash_test::random::random_vector(1 << 24);
  auto source_device = cuckoohash_test::benchmark::to_device(source);
  auto another = cuckoohash_test::random::random_vector(1 << 24);
  {
    auto cuckoo = cuckoohash::set::Set<2, 1 << 25, 24 << 2>();
    cuckoo.insert(source_device);
    lookup_test<2, 0>(source, another, cuckoo);
    lookup_test<2, 1>(source, another, cuckoo);
    lookup_test<2, 2>(source, another, cuckoo);
    lookup_test<2, 3>(source, another, cuckoo);
    lookup_test<2, 4>(source, another, cuckoo);
    lookup_test<2, 5>(source, another, cuckoo);
    lookup_test<2, 6>(source, another, cuckoo);
    lookup_test<2, 7>(source, another, cuckoo);
    lookup_test<2, 8>(source, another, cuckoo);
    lookup_test<2, 9>(source, another, cuckoo);
    lookup_test<2, 10>(source, another, cuckoo);
  }

  {
    auto cuckoo = cuckoohash::set::Set<3, 1 << 25, 24 << 2>();
    cuckoo.insert(source_device);
    lookup_test<3, 0>(source, another, cuckoo);
    lookup_test<3, 1>(source, another, cuckoo);
    lookup_test<3, 2>(source, another, cuckoo);
    lookup_test<3, 3>(source, another, cuckoo);
    lookup_test<3, 4>(source, another, cuckoo);
    lookup_test<3, 5>(source, another, cuckoo);
    lookup_test<3, 6>(source, another, cuckoo);
    lookup_test<3, 7>(source, another, cuckoo);
    lookup_test<3, 8>(source, another, cuckoo);
    lookup_test<3, 9>(source, another, cuckoo);
    lookup_test<3, 10>(source, another, cuckoo);
  }

  return 0;
}