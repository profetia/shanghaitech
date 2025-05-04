
#include <cstddef>
#include <iostream>
#include <vector>

#include "benchmark.cuh"
#include "cuckoohash/set.cuh"
#include "random.cuh"

template <const std::size_t T, const std::size_t R>
void bound_test(thrust::device_vector<std::uint32_t>& source_device) {
  constexpr std::size_t C = 1.4 * (1 << 24);
  constexpr std::size_t U = 24 * static_cast<double>(R) / 10 + 1;
  auto cuckoo = cuckoohash::set::Set<T, C, U>();
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

  bound_test<2, 2>(source_device);
  bound_test<2, 4>(source_device);
  bound_test<2, 6>(source_device);
  bound_test<2, 8>(source_device);
  bound_test<2, 10>(source_device);
  bound_test<2, 12>(source_device);
  bound_test<2, 14>(source_device);
  bound_test<2, 16>(source_device);
  bound_test<2, 18>(source_device);
  bound_test<2, 20>(source_device);
  bound_test<2, 22>(source_device);
  bound_test<2, 24>(source_device);
  bound_test<2, 36>(source_device);
  bound_test<2, 48>(source_device);
  bound_test<2, 72>(source_device);
  bound_test<2, 96>(source_device);
  bound_test<2, 144>(source_device);
  bound_test<2, 192>(source_device);

  bound_test<3, 2>(source_device);
  bound_test<3, 4>(source_device);
  bound_test<3, 6>(source_device);
  bound_test<3, 8>(source_device);
  bound_test<3, 10>(source_device);
  bound_test<3, 12>(source_device);
  bound_test<3, 14>(source_device);
  bound_test<3, 16>(source_device);
  bound_test<3, 18>(source_device);
  bound_test<3, 20>(source_device);
  bound_test<3, 22>(source_device);
  bound_test<3, 24>(source_device);
  bound_test<3, 36>(source_device);
  bound_test<3, 48>(source_device);
  bound_test<3, 72>(source_device);
  bound_test<3, 96>(source_device);
  bound_test<3, 144>(source_device);
  bound_test<3, 192>(source_device);

  return 0;
}