cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Release)

project(bunjilearn VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(dataset)
add_subdirectory(layers)
add_subdirectory(metrics)
add_subdirectory(model)
add_subdirectory(util)
