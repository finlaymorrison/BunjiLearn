#pragma once

#include <fmt/core.h>
#include <fmt/color.h>

// makes sure that __FILENAME__ is the name of the file and not its path
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define PRINT(...)                                                            \
    do                                                                        \
    {                                                                         \
        fmt::print(__VA_ARGS__);                                              \
        std::fflush(stdout);                                                  \
    }                                                                         \
    while(false)
    

// Prints output with debug information.
#define BUNJI_LOGDB(level, disp, ...)                                         \
    do                                                                        \
    {                                                                         \
        PRINT(disp, "[{}] <{}:{}>: ", level, __FILENAME__, __LINE__);         \
        PRINT(disp, __VA_ARGS__);                                             \
        PRINT("\n");                                                          \
    }                                                                         \
    while(false)

// Prints output without debug information.
#define BUNJI_LOG(disp, ...)                                                  \
    do                                                                        \
    {                                                                         \
        PRINT(disp, __VA_ARGS__);                                             \
        PRINT("\n");                                                          \
    }                                                                         \
    while(false)

// Resets write head to start of line and does not print a newline.
#define BUNJI_LOG_REPLACE(disp, ...)                                          \
    do                                                                        \
    {                                                                         \
        PRINT("\r");                                                          \
        PRINT(disp, __VA_ARGS__);                                             \
    }                                                                         \
    while(false)

#define BUNJI_LOG_NL fmt::print("\n");

#ifdef BUNJI_LOG_INFO
#define BUNJI_INF(...) BUNJI_LOG(fg(fmt::color::white), __VA_ARGS__)
#else
#define BUNJI_INF(...)
#endif

#ifdef BUNJI_LOG_DEBUG
#define BUNJI_DBG(...) BUNJI_LOGDB("D", fg(fmt::color::green), __VA_ARGS__)
#else
#define BUNJI_DBG(...)
#endif

#ifdef BUNJI_LOG_WARNING
#define BUNJI_WRN(...) BUNJI_LOGDB("W", fg(fmt::color::yellow), __VA_ARGS__)
#else
#define BUNJI_WRN(...)
#endif

#ifdef BUNJI_LOG_ERROR
#define BUNJI_ERR(...) BUNJI_LOGDB("E", fg(fmt::color::red), __VA_ARGS__)
#else
#define BUNJI_ERR(...)
#endif