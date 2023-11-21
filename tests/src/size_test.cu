
#include <cstddef>
#include <iostream>
#include <vector>

#include "benchmark.cuh"
#include "cuckoohash/set.cuh"
#include "random.cuh"

template <const std::size_t T, const std::size_t R>
void size_test(thrust::device_vector<std::uint32_t>& source_device) {
  constexpr std::size_t C = (static_cast<double>(R) / 100) * (1 << 24);
  auto cuckoo = cuckoohash::set::Set<T, C, 24 << 2>();
  try {
    auto benchmark = cuckoohash_test::benchmark::profile([&](auto&& timer) {
      cuckoo.insert(source_device);
      auto elapsed = timer.elapsed();
      cuckoo.clear();
      return elapsed;
    });
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

  size_test<2, 101>(source_device);
  size_test<2, 102>(source_device);
  size_test<2, 105>(source_device);
  size_test<2, 110>(source_device);
  size_test<2, 120>(source_device);
  size_test<2, 130>(source_device);
  size_test<2, 140>(source_device);
  size_test<2, 150>(source_device);
  size_test<2, 160>(source_device);
  size_test<2, 170>(source_device);
  size_test<2, 180>(source_device);
  size_test<2, 190>(source_device);
  size_test<2, 200>(source_device);

  size_test<3, 101>(source_device);
  size_test<3, 102>(source_device);
  size_test<3, 105>(source_device);
  size_test<3, 110>(source_device);
  size_test<3, 120>(source_device);
  size_test<3, 130>(source_device);
  size_test<3, 140>(source_device);
  size_test<3, 150>(source_device);
  size_test<3, 160>(source_device);
  size_test<3, 170>(source_device);
  size_test<3, 180>(source_device);
  size_test<3, 190>(source_device);
  size_test<3, 200>(source_device);

  return 0;
}