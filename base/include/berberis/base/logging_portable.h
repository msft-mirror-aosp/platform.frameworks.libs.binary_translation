/*
 * Copyright (C) 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BERBERIS_BASE_LOGGING_PORTABLE_H_
#define BERBERIS_BASE_LOGGING_PORTABLE_H_

#include <cstdio>
#include <cstdlib>

#define __berberis_log_print(severity, ...) \
  ((void)(fprintf(stderr, LOG_TAG ": %s: ", severity), fprintf(stderr, __VA_ARGS__)))
#define __berberis_log_fatal(condition, ...) \
  ((void)(fprintf(stderr, LOG_TAG ": %s: ", condition), fprintf(stderr, __VA_ARGS__)), abort())

#define ALOGE(...) __berberis_log_print("E", __VA_ARGS__)
#define ALOGW(...) __berberis_log_print("W", __VA_ARGS__)
#define ALOGI(...) __berberis_log_print("I", __VA_ARGS__)
#define ALOGD(...) __berberis_log_print("D", __VA_ARGS__)
#define ALOGV(...) __berberis_log_print("V", __VA_ARGS__)

#define LOG_ALWAYS_FATAL_IF(cond, ...) ((cond) ? __berberis_log_fatal(#cond, __VA_ARGS__) : (void)0)

#define LOG_ALWAYS_FATAL(...) __berberis_log_fatal("fatal", __VA_ARGS__)

#endif  // BERBERIS_BASE_LOGGING_PORTABLE_H_
