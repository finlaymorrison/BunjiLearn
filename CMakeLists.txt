cmake_minimum_required(VERSION 3.10)

project(ml_library VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")

configure_file(config.h.in config.h)

add_executable(ml_library ${PROJECT_SOURCE_DIR}/src/main.cpp)

add_library(ml_library_src
    ${PROJECT_SOURCE_DIR}/src/dataset.cpp
    ${PROJECT_SOURCE_DIR}/src/dense.cpp
    ${PROJECT_SOURCE_DIR}/src/activation.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer.cpp
    ${PROJECT_SOURCE_DIR}/src/network.cpp
    ${PROJECT_SOURCE_DIR}/src/loss.cpp
    ${PROJECT_SOURCE_DIR}/src/flatten.cpp
    ${PROJECT_SOURCE_DIR}/src/layer.cpp
    )

target_include_directories(ml_library PUBLIC ${PROJECT_BINARY_DIR})
target_include_directories(ml_library_src PUBLIC ${PROJECT_SOURCE_DIR}/src/include)
target_include_directories(ml_library_src PUBLIC ${PROJECT_SOURCE_DIR}/third_party/json/single_include)

target_link_libraries(ml_library ml_library_src)


