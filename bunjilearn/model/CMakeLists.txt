cmake_minimum_required(VERSION 3.10)

project(model)
add_library(model SHARED
    ${PROJECT_SOURCE_DIR}/network.cpp
    ${PROJECT_SOURCE_DIR}/trainer.cpp
    )
target_include_directories(model PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(model util layers dataset metrics)
