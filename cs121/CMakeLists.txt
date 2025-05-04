cmake_minimum_required(VERSION 3.21)

project(
    CuckooHash 
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CUDA_SEPARABLE_COMPILATION ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif ()

file(GLOB_RECURSE HEADERS include/cuckoohash/*.hpp include/cuckoohash/*.cuh)
file(GLOB_RECURSE SOURCES src/*.cpp src/*.cu)

file(GLOB_RECURSE BENCH_SOURCES benches/src/*.cpp benches/src/*.cu)
file(GLOB_RECURSE BENCH_HEADERS benches/include/*.hpp benches/include/*.cuh)

add_library(${CMAKE_PROJECT_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC include)

enable_testing()
add_subdirectory(tests)
