#pragma once

#include <cstdio>
#include <cstring>

// makes sure that __FILENAME__ is the name of the file and not its path
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define BUNJI_LOG(level, ...)                                                    \
    do                                                                           \
    {                                                                            \
        printf("[%s] %s <%s:%d>: ", level, __FILENAME__, __func__, __LINE__);    \
        printf(__VA_ARGS__);                                                     \
        printf("\n");                                                            \
    }                                                                            \
    while(false)

#define BUNJI_DBG(...) BUNJI_LOG("D", __VA_ARGS__)
#define BUNJI_WRN(...) BUNJI_LOG("W", __VA_ARGS__)
#define BUNJI_ERR(...) BUNJI_LOG("E", __VA_ARGS__)
    