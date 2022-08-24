cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Release)

project(ml_library VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file(config.h.in config.h)

add_executable(ml_library ${PROJECT_SOURCE_DIR}/mains/main.cpp)

add_subdirectory(bunjilearn)
add_subdirectory(third_party/json)

target_link_libraries(ml_library layers model dataset metrics)

target_include_directories(ml_library PUBLIC ${PROJECT_BINARY_DIR})
