#pragma once

#include <fmt/core.h>

// makes sure that __FILENAME__ is the name of the file and not its path
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define BUNJI_LOG(level, ...)                                           \
    do                                                                  \
    {                                                                   \
        fmt::print("[{}] <{}:{}>: ", level, __FILENAME__, __LINE__);    \
        fmt::print(__VA_ARGS__);                                        \
        fmt::print("\n");                                               \
    }                                                                   \
    while(false)

#define BUNJI_INF(...) BUNJI_LOG("I", __VA_ARGS__)
#define BUNJI_DBG(...) BUNJI_LOG("D", __VA_ARGS__)
#define BUNJI_WRN(...) BUNJI_LOG("W", __VA_ARGS__)
#define BUNJI_ERR(...) BUNJI_LOG("E", __VA_ARGS__)
    