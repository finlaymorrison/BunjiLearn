cmake_minimum_required(VERSION 3.10)

project(layers)
add_library(layers
    ${PROJECT_SOURCE_DIR}/dense.cpp
    ${PROJECT_SOURCE_DIR}/activation.cpp
    ${PROJECT_SOURCE_DIR}/flatten.cpp
    ${PROJECT_SOURCE_DIR}/layer.cpp
    ${PROJECT_SOURCE_DIR}/batch_norm.cpp
    ${PROJECT_SOURCE_DIR}/convolution.cpp
    ${PROJECT_SOURCE_DIR}/dropout.cpp
    ${PROJECT_SOURCE_DIR}/pool.cpp
    )
target_include_directories(layers PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(layers util)
