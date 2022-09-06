cmake_minimum_required(VERSION 3.10)

project(util)
add_library(util
    ${PROJECT_SOURCE_DIR}/empty.cpp
    )
target_include_directories(util PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(util fmt-header-only)
