cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Release)

project(model VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(model
    ${PROJECT_SOURCE_DIR}/network.cpp
    ${PROJECT_SOURCE_DIR}/trainer.cpp
    )

target_include_directories(model PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(model util layers dataset metrics)
