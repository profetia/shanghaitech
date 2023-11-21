
#include <cstddef>
#include <iostream>
#include <vector>

#include "benchmark.cuh"
#include "cuckoohash/set.cuh"
#include "random.cuh"

template <std::size_t T, std::size_t C, std::size_t U, std::size_t S>
void insert_test() {
  auto source = cuckoohash_test::random::random_vector(1 << S);
  auto source_device = cuckoohash_test::benchmark::to_device(source);
  auto cuckoo = cuckoohash::set::Set<T, 1 << C, U << 2>();
  try {
    auto benchmark = cuckoohash_test::benchmark::profile([&](auto&& timer) {
      cuckoo.insert(source_device);
      auto elapsed = timer.elapsed();
      cuckoo.clear();
      return elapsed;
    });
    std::cout << "[PASSED] - " << __FUNCTION__ << "<" << T << ", " << C << ", " << U << ", " << S
              << ">" << std::endl;
    std::cout << cuckoohash_test::benchmark::report(benchmark, 1 << S) << std::endl;
  } catch (const std::exception& e) {
    std::cout << "[FAILED] - " << __FUNCTION__ << "<" << T << ", " << C << ", " << U << ", " << S
              << ">" << std::endl;
    std::cout << e.what() << std::endl;
  }
}

int main() {
  insert_test<2, 25, 10, 10>();
  insert_test<2, 25, 11, 11>();
  insert_test<2, 25, 12, 12>();
  insert_test<2, 25, 13, 13>();
  insert_test<2, 25, 14, 14>();
  insert_test<2, 25, 15, 15>();
  insert_test<2, 25, 16, 16>();
  insert_test<2, 25, 17, 17>();
  insert_test<2, 25, 18, 18>();
  insert_test<2, 25, 19, 19>();
  insert_test<2, 25, 20, 20>();
  insert_test<2, 25, 21, 21>();
  insert_test<2, 25, 22, 22>();
  insert_test<2, 25, 23, 23>();
  insert_test<2, 25, 24, 24>();
  insert_test<3, 25, 10, 10>();
  insert_test<3, 25, 11, 11>();
  insert_test<3, 25, 12, 12>();
  insert_test<3, 25, 13, 13>();
  insert_test<3, 25, 14, 14>();
  insert_test<3, 25, 15, 15>();
  insert_test<3, 25, 16, 16>();
  insert_test<3, 25, 17, 17>();
  insert_test<3, 25, 18, 18>();
  insert_test<3, 25, 19, 19>();
  insert_test<3, 25, 20, 20>();
  insert_test<3, 25, 21, 21>();
  insert_test<3, 25, 22, 22>();
  insert_test<3, 25, 23, 23>();
  insert_test<3, 25, 24, 24>();
  return 0;
}