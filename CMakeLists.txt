cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

add_subdirectory(bunjilearn)
add_subdirectory(third_party/json)
add_subdirectory(third_party/fmt)

configure_file(config.h.in config.h)

project(ml_library)
get_directory_property(<variable> [DIRECTORY <dir>] <prop-name>)
add_executable(ml_library ${PROJECT_SOURCE_DIR}/mains/main.cpp)
target_link_libraries(ml_library bunjilearn)
target_include_directories(bunjilearn PUBLIC ${PROJECT_BINARY_DIR})
