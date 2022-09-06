cmake_minimum_required(VERSION 3.10)

project(dataset)
add_library(dataset
    ${PROJECT_SOURCE_DIR}/dataset.cpp
    )
target_include_directories(dataset PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(dataset util nlohmann_json)
