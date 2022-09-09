cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-Wall -Wextra -Wpedantic)

add_subdirectory(bunjilearn)
add_subdirectory(third_party/json)
add_subdirectory(third_party/fmt)
add_subdirectory(third_party/googletest)

project(unit-testing)

enable_testing()

add_executable(tensor-tests ${PROJECT_SOURCE_DIR}/mains/tensor_test.cpp)

target_link_libraries(tensor-tests gtest bunjilearn)

add_test(
    NAME tensor-tests
    COMMAND tensor-tests
)
