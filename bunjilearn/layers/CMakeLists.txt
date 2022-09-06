cmake_minimum_required(VERSION 3.10)

project(layers)
add_library(layers
    ${PROJECT_SOURCE_DIR}/dense.cpp
    ${PROJECT_SOURCE_DIR}/activation.cpp
    ${PROJECT_SOURCE_DIR}/flatten.cpp
    ${PROJECT_SOURCE_DIR}/layer.cpp
    )
target_include_directories(layers PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(layers util)
