cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

project(layers)
add_library(layers
    ${PROJECT_SOURCE_DIR}/dense.cpp
    ${PROJECT_SOURCE_DIR}/activation.cpp
    ${PROJECT_SOURCE_DIR}/flatten.cpp
    ${PROJECT_SOURCE_DIR}/layer.cpp
    )
target_include_directories(layers PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(layers util)
