/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include <pthread.h>

#include "berberis/base/checks.h"

namespace berberis {

// Manages the current guest thread.
pthread_key_t g_guest_thread_key;

namespace {

void GuestThreadDtor(void* /* arg */) {
  // TODO(b/280671643): Implement DetachCurrentThread().
  // DetachCurrentThread();
}

}  // namespace

// Not thread safe, not async signals safe!
void InitGuestThreadManager() {
  // Here we don't need pthread_once, which is not reentrant due to spinlocks.
  CHECK_EQ(0, pthread_key_create(&g_guest_thread_key, GuestThreadDtor));
}

}  // namespace berberis