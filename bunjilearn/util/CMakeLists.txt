cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Release)

project(util VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(util
    ${PROJECT_SOURCE_DIR}/empty.cpp
    )

target_include_directories(util PUBLIC ${PROJECT_SOURCE_DIR}/include)
