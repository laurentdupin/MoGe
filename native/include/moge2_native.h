#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(MOGE2_BUILD_DLL)
#    define MOGE2_API __declspec(dllexport)
#  else
#    define MOGE2_API __declspec(dllimport)
#  endif
#else
#  define MOGE2_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

MOGE2_API int moge2_get_transfer_counters(
    uint64_t* upload_bytes, uint64_t* download_bytes);

#ifdef __cplusplus
}
#endif
