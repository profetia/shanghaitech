#include <iostream>

#include "cuckoohash/tmp.cuh"

namespace cuckoohash {

int add_and_print(int a, int b) {
  std::cout << "a + b = " << a + b << std::endl;
  return a + b;
}

}  // namespace cuckoohash
