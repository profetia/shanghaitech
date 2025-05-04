/* Copyright 2022-2023 Linshu Yang. 

This file is adopted from Tanc.

Tanc is an open source project licensed under the terms of the GNU General 
Public License v3.0. You can redistribute it and/or modify it conditioning on 
making available complete source code of licensed works and modifications, 
which include larger works using a licensed work, under the same license. 
Copyright and license notices must be preserved.

For more information, please refer to <https://github.com/yanglinshu/tanc/>.
*/

#ifndef __LIB_KERNEL_TANC_H
#define __LIB_KERNEL_TANC_H

#include <stdbool.h>
#include "tanc.conf.h"

/* Level of Log */
enum LogLevel {
  LOG_LEVEL_TRACE,
  LOG_LEVEL_DEBUG,
  LOG_LEVEL_INFO,
  LOG_LEVEL_WARN,
  LOG_LEVEL_ERROR,
  LOG_LEVEL_FATAL,
  LOG_LEVEL_NONE,
};

/* Return true if the log level is enabled */
inline bool log_is_enabled(enum LogLevel level) {
  return level >= LOG_LEVEL;
}

/* Convert log level to string */
inline const char* log_level_to_string(enum LogLevel level) {
  switch (level) {
    case LOG_LEVEL_TRACE: 
      return "TRACE";
    case LOG_LEVEL_DEBUG: 
      return "DEBUG";
    case LOG_LEVEL_INFO: 
      return "INFO";
    case LOG_LEVEL_WARN: 
      return "WARN";
    case LOG_LEVEL_ERROR: 
      return "ERROR";
    case LOG_LEVEL_FATAL: 
      return "FATAL";
    case LOG_LEVEL_NONE: 
      return "NONE";
    default: 
      return "UNKNOWN";
  }
}

/* Write log message header */
inline int log_message_header(enum LogLevel level, const char* file, int line, 
                              const char* function) {
  return LOG_WRITE_HANDLER("[%s] %s:%d %s(): ", log_level_to_string(level), 
                            file, line, function);
}

/* Write log message footer */
inline int log_message_footer(void) {
  return LOG_WRITE_HANDLER("\n");
}

/* Log message at level TRACE */
#define LOG_TRACE(body)                                                  \
  do {                                                                   \
    if (log_is_enabled(LOG_LEVEL_TRACE)) { /* Check if is enabled */     \
      log_message_header(LOG_LEVEL_TRACE, __FILE__, __LINE__, __func__); \
      LOG_WRITE_HANDLER body;                                            \
      log_message_footer();                                              \
    }                                                                    \
  } while (0)

/* Log message at level DEBUG */
#define LOG_DEBUG(body)                                                  \
  do {                                                                   \
    if (log_is_enabled(LOG_LEVEL_DEBUG)) { /* Check if is enabled */     \
      log_message_header(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __func__); \
      LOG_WRITE_HANDLER body;                                            \
      log_message_footer();                                              \
    }                                                                    \
  } while (0)

/* Log message at level INFO */
#define LOG_INFO(body)                                                  \
  do {                                                                  \
    if (log_is_enabled(LOG_LEVEL_INFO)) { /* Check if is enabled */     \
      log_message_header(LOG_LEVEL_INFO, __FILE__, __LINE__, __func__); \
      LOG_WRITE_HANDLER body;                                           \
      log_message_footer();                                             \
    }                                                                   \
  } while (0)

/* Log message at level WARN */
#define LOG_WARN(body)                                                  \
  do {                                                                  \
    if (log_is_enabled(LOG_LEVEL_WARN)) { /* Check if is enabled */     \
      log_message_header(LOG_LEVEL_WARN, __FILE__, __LINE__, __func__); \
      LOG_WRITE_HANDLER body;                                           \
      log_message_footer();                                             \
    }                                                                   \
  } while (0)

/* Log message at level ERROR */
#define LOG_ERROR(body)                                                  \
  do {                                                                   \
    if (log_is_enabled(LOG_LEVEL_ERROR)) { /* Check if is enabled */     \
      log_message_header(LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__); \
      LOG_WRITE_HANDLER body;                                            \
      log_message_footer();                                              \
    }                                                                    \
  } while (0)

/* Log message at level FATAL */
#define LOG_FATAL(body)                                                  \
  do {                                                                   \
    if (log_is_enabled(LOG_LEVEL_FATAL)) { /* Check if is enabled */     \
      log_message_header(LOG_LEVEL_FATAL, __FILE__, __LINE__, __func__); \
      LOG_WRITE_HANDLER body;                                            \
      log_message_footer();                                              \
      abort();                                                           \
    }                                                                    \
  } while (0)

/* Log the value of an expression at level DEBUG and return it */
#define DBG(format, expr)                                                     \
  ((log_is_enabled(LOG_LEVEL_DEBUG)                                           \
        ? (log_message_header(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __func__), \
           LOG_WRITE_HANDLER(#expr " = " format " (%s)", (expr), format),     \
           log_message_footer())                                              \
        : 0),                                                                 \
   (expr))

/* Log the value of an expression at level DEBUG without return it */
#define DBGL(format, expr) \
  LOG_DEBUG((#expr " = " format " (%s)", (expr), format))

#endif // __LIB_KERNEL_TANC_H