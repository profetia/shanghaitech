cmake_minimum_required(VERSION 3.21)

project(${CMAKE_PROJECT_NAME}Tests LANGUAGES CXX CUDA)

file(GLOB_RECURSE TEST_HEADERS include/*.hpp include/*.cuh)
file(GLOB_RECURSE TEST_SOURCES src/*.cpp src/*.cu)

foreach(file ${TEST_SOURCES})
    string(REGEX REPLACE "(.*/)([a-zA-Z0-9_ ]+)(\.cpp|\.cu)" "\\2" test_name ${file}) 
    add_executable(${test_name}_Tests ${file} ${TEST_HEADERS})
    target_include_directories(${test_name}_Tests PUBLIC include)
    
    target_link_libraries(
        ${test_name}_Tests
        PUBLIC
        ${CMAKE_PROJECT_NAME}
    )

    add_test(
        NAME ${test_name}
        COMMAND ${test_name}_Tests
    )
endforeach()

