cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-Wall -Wextra -Wpedantic)

add_subdirectory(third_party/json)
add_subdirectory(third_party/fmt)
add_subdirectory(third_party/googletest)
add_subdirectory(third_party/pybind11)
add_subdirectory(bunjilearn)
add_subdirectory(api)
add_subdirectory(tests)

project(bunjilearn)

add_executable(example ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(example bunjilearn)

enable_testing()
