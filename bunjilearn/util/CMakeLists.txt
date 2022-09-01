cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Release)

project(util VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(util
    ${PROJECT_SOURCE_DIR}/empty.cpp
    )

target_include_directories(util PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(util fmt-header-only)

option(BUNJI_LOG_INFO "Enable BUNJI_INF logging." ON)
option(BUNJI_LOG_DEBUG "Enable BUNJI_DBG logging." ON)
option(BUNJI_LOG_WARNING "Enable BUNJI_WRN logging." ON)
option(BUNJI_LOG_ERROR "Enable BUNJI_ERR logging." ON)

if(BUNJI_LOG_INFO)
  list(APPEND COMPILE_DEFINITIONS "BUNJI_LOG_INFO")
endif()

if(BUNJI_LOG_DEBUG)
  list(APPEND COMPILE_DEFINITIONS "BUNJI_LOG_DEBUG")
endif()

if(BUNJI_LOG_WARNING)
  list(APPEND COMPILE_DEFINITIONS "BUNJI_LOG_WARNING")
endif()

if(BUNJI_LOG_ERROR)
  list(APPEND COMPILE_DEFINITIONS "BUNJI_LOG_ERROR")
endif()

target_compile_definitions(util PUBLIC ${COMPILE_DEFINITIONS})
