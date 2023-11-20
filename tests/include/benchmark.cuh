#ifndef CUCKOOHASH_TEST_BENCHMARK_H_
#define CUCKOOHASH_TEST_BENCHMARK_H_

#include <cuda_runtime.h>

#include <string>
#include <tuple>

namespace cuckoohash_test::benchmark {

namespace detail {

class Instant {
 public:
  Instant() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }

  ~Instant() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  float elapsed() {
    cudaEventSynchronize(stop_);
    float elapsed = 0.0F;
    cudaEventElapsedTime(&elapsed, start_, stop_);
    return elapsed;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

}  // namespace detail

template <typename F, typename... Args>
float record(F&& f, Args&&... args) {
  auto timer = detail::Instant{};
  f(std::forward<Args>(args)...);
  return timer.elapsed();
}

template <typename F, typename... Args>
std::tuple<double, double> benchmark(F&& f, Args&&... args) {
  auto elapseds = std::vector<float>{};
  for (auto i = 0; i < 5; ++i) {
    elapseds.push_back(record(std::forward<F>(f), std::forward<Args>(args)...));
  }
  auto avg = std::accumulate(elapseds.begin(), elapseds.end(), 0.0F) / elapseds.size();
  auto stddev = std::sqrt(
      std::accumulate(elapseds.begin(), elapseds.end(), 0.0F,
                      [avg](auto acc, auto elapsed) { return acc + std::pow(elapsed - avg, 2); }) /
      elapseds.size());
  return {avg, stddev};
}

std::string report(const std::tuple<double, double>& benchmark, std::size_t ops) {
  auto [avg, stddev] = benchmark;
  auto mops = ops / 1e6 / avg;

  auto report = std::ostringstream{};
  report << set_precision(6) << mops << " Mops/s ("
         << "avg: " << avg << " ms, "
         << "stddev: " << stddev << " ms)";
  return report.str();
}

}  // namespace cuckoohash_test::benchmark

#endif  // CUCKOOHASH_TEST_BENCHMARK_H_
