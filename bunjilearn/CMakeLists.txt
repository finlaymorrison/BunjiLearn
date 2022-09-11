cmake_minimum_required(VERSION 3.10)

add_subdirectory(dataset)
add_subdirectory(layers)
add_subdirectory(metrics)
add_subdirectory(model)
add_subdirectory(util)

project(bunjilearn-lib)

set(BUNJILEARN_VERSION_MAJOR 0 CACHE STRING "Major version of bunjilearn")
set(BUNJILEARN_VERSION_MINOR 1 CACHE STRING "Minor version of bunjilearn")
add_library(bunjilearn SHARED
    ${PROJECT_SOURCE_DIR}/empty.cpp
    )
target_link_libraries(bunjilearn dataset layers metrics model util)
