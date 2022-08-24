cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Release)

project(metrics VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(metrics
    ${PROJECT_SOURCE_DIR}/loss.cpp
    )

target_include_directories(metrics PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(metrics util)
