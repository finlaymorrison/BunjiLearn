#pragma once

#include <fmt/core.h>
#include <fmt/color.h>

// makes sure that __FILENAME__ is the name of the file and not its path
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define BUNJI_LOG_IMPL(level, color, ...)                                      \
    do                                                                         \
    {                                                                          \
        fmt::print(color, "[{}] <{}:{}>: ", level, __FILENAME__, __LINE__);    \
        fmt::print(color, __VA_ARGS__);                                        \
        fmt::print("\n");                                                      \
    }                                                                          \
    while(false)

#define BUNJI_PRINT(...) fmt::print(__VA_ARGS__) 

#define BUNJI_INF(...) BUNJI_LOG_IMPL("I", fg(fmt::color::white), __VA_ARGS__)
#define BUNJI_DBG(...) BUNJI_LOG_IMPL("D", fg(fmt::color::blue), __VA_ARGS__)
#define BUNJI_WRN(...) BUNJI_LOG_IMPL("W", fg(fmt::color::orange), __VA_ARGS__)
#define BUNJI_ERR(...) BUNJI_LOG_IMPL("E", fg(fmt::color::red), __VA_ARGS__)
    