#pragma once
#include <cstdio>

#define COLOR_OK    "\033[0;32m" // Green
#define COLOR_FAIL  "\033[0;31m" // Red
#define COLOR_RESET "\033[0m"    // Reset to default color

#define TEST(expr)                                                                                 \
  do {                                                                                             \
    int status = (expr);                                                                           \
    if (status >= 0) {                                                                             \
      printf(COLOR_OK "[OK]" COLOR_RESET "%s\n", #expr);                                           \
    } else {                                                                                       \
      printf(COLOR_FAIL "[FAIL]" COLOR_RESET " %s, %s:%d %s\n", #expr, __FILE__, __LINE__, #expr); \
      fflush(stdout);                                                                              \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#define ASSERT(expr)                                                                            \
  do {                                                                                          \
    int status = (expr);                                                                        \
    if (status < 0) {                                                                           \
      printf(COLOR_FAIL "[FAIL] %s, %s:%d %s\n" COLOR_RESET, #expr, __FILE__, __LINE__, #expr); \
      fflush(stdout);                                                                           \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

#define C_ASSERT(expr)                                                                          \
  do {                                                                                          \
    int status = (expr);                                                                        \
    if (status != 0) {                                                                          \
      printf(COLOR_FAIL "[FAIL] %s, %s:%d %s\n" COLOR_RESET, #expr, __FILE__, __LINE__, #expr); \
      fflush(stdout);                                                                           \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

#define ERROR(...) \
  do { fprintf(stderr, "[ERROR] " __VA_ARGS__); } while (0)

#define WARNING(...) \
  do { fprintf(stderr, "[WARNING] " __VA_ARGS__); } while (0)

#define LOG(...) \
  do { fprintf(stderr, "[LOG] " __VA_ARGS__); } while (0)
#define EXIT() \
  do {         \
    exit(1);   \
  } while (0);
