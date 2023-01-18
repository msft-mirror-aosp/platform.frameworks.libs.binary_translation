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

#ifndef BERBERIS_BASE_LOGGING_ANDROID_H_
#define BERBERIS_BASE_LOGGING_ANDROID_H_

#include <cutils/log.h>

#include "berberis/base/scoped_errno.h"

// We must save errno and restore it when we are doing logging (b/27992399).
#undef LOG_PRI
#define LOG_PRI(priority, tag, ...) \
  (::berberis::ScopedErrno(), android_printLog(priority, tag, __VA_ARGS__))

#endif  // BERBERIS_BASE_LOGGING_ANDROID_H_
