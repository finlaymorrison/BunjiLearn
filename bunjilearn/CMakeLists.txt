cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

add_subdirectory(dataset)
add_subdirectory(layers)
add_subdirectory(metrics)
add_subdirectory(model)
add_subdirectory(util)

project(bunjilearn)
set(BUNJILEARN_VERSION_MAJOR 0 CACHE STRING "Major version of bunjilearn library")
set(BUNJILEARN_VERSION_MINOR 1 CACHE STRING "Minor version of bunjilearn library")
add_library(bunjilearn
    ${PROJECT_SOURCE_DIR}/empty.cpp
    )
target_link_libraries(bunjilearn dataset layers metrics model util)
