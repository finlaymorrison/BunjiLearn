cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

project(dataset VERSION 1.0)
add_library(dataset
    ${PROJECT_SOURCE_DIR}/dataset.cpp
    )
target_include_directories(dataset PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(dataset util nlohmann_json)
