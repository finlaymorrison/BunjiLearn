cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-Wall -Wextra -Wpedantic)

add_subdirectory(bunjilearn)
add_subdirectory(third_party/json)
add_subdirectory(third_party/fmt)

configure_file(config.h.in config.h)

project(ml_library)
add_executable(ml_library ${PROJECT_SOURCE_DIR}/mains/main.cpp)
target_link_libraries(ml_library bunjilearn)
target_include_directories(bunjilearn PUBLIC ${PROJECT_BINARY_DIR})
