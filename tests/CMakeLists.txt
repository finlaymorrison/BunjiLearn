cmake_minimum_required(VERSION 3.10)

project(bunilearn-tests)

add_executable(tensor-tests ${PROJECT_SOURCE_DIR}/tensor_test.cpp)
target_link_libraries(tensor-tests gtest gtest_main bunjilearn)

add_test(
    NAME tensor-tests
    COMMAND tensor-tests
)
