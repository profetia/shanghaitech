#ifndef CUCKOOHASH_TEST_RANDOM_H_
#define CUCKOOHASH_TEST_RANDOM_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace cuckoohash_test::random {

std::vector<std::uint32_t> random_vector(const std::size_t size) {
  auto vec = std::vector<std::uint32_t>(size);
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<std::uint32_t> dis{0, std::numeric_limits<std::int32_t>::max()};
  for (auto& i : vec) {
    i = dis(gen);
  }
  return vec;
}

std::vector<std::uint32_t> shuffle(std::vector<std::uint32_t>& vec) {
  auto shuffled = std::vector<std::uint32_t>(vec);
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::shuffle(shuffled.begin(), shuffled.end(), gen);
  return shuffled;
}

std::vector<std::uint32_t> blend(std::vector<std::uint32_t>& vec,
                                 std::vector<std::uint32_t>& another, double ratio) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<double> dis{0.0, 1.0};
  auto blended = std::vector<std::uint32_t>(vec);
  for (std::size_t i = 0; i < vec.size(); ++i) {
    if (dis(gen) < ratio) {
      blended[i] = another[i];
    }
  }
  return blended;
}

std::vector<std::uint32_t> blend(std::vector<std::uint32_t>& vec, double ratio) {
  auto another = random_vector(vec.size());
  return blend(vec, another, ratio);
}

}  // namespace cuckoohash_test::random

#endif  // CUCKOOHASH_TEST_RANDOM_H_
