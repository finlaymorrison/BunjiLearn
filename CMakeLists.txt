cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

add_subdirectory(bunjilearn)
add_subdirectory(third_party/json)
add_subdirectory(third_party/fmt)

project(ml_library VERSION 1.0)
configure_file(config.h.in config.h)

add_executable(ml_library ${PROJECT_SOURCE_DIR}/mains/main.cpp)
target_include_directories(ml_library PUBLIC ${PROJECT_BINARY_DIR})
target_link_libraries(ml_library bunjilearn)
