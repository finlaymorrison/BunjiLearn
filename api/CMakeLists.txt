cmake_minimum_required(VERSION 3.10)

project(api)

pybind11_add_module(bunjilearn_py api.cpp)
target_link_libraries(bunjilearn_py PUBLIC bunjilearn)
