cmake_minimum_required(VERSION 3.10)

project(metrics)
add_library(metrics
    ${PROJECT_SOURCE_DIR}/loss.cpp
    ${PROJECT_SOURCE_DIR}/metric.cpp
    )
target_include_directories(metrics PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(metrics util)
