cmake_minimum_required(VERSION 3.21)

project(${CMAKE_PROJECT_NAME}Benches LANGUAGES CXX CUDA)

file(GLOB_RECURSE BENCH_HEADERS include/*.hpp include/*.cuh)
file(GLOB_RECURSE BENCH_SOURCES src/*.cpp src/*.cu)

foreach(file ${BENCH_SOURCES})
    string(REGEX REPLACE "(.*/)([a-zA-Z0-9_ ]+)(\.cpp|\.cu)" "\\2" bench_name ${file}) 
    add_executable(${bench_name}_Benches ${file} ${BENCH_HEADERS})
    
    target_link_libraries(
        ${bench_name}_Benches
        PUBLIC
        ${CMAKE_PROJECT_NAME}
    )

    add_test(
        NAME ${bench_name}
        COMMAND ${bench_name}_Benches
    )
endforeach()

