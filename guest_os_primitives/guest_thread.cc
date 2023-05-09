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
#include <sys/types.h>  // pid_t
#include <mutex>

#include "berberis/base/checks.h"
#include "berberis/base/forever_map.h"

namespace berberis {

// TODO(b/280551726): Replace this forward declaration with a full class
// declaration included from guest_thread.h.
class GuestThread;

// Manages the current guest thread.
pthread_key_t g_guest_thread_key;

namespace {

typedef ForeverMap<pid_t, GuestThread*> GuestThreadMap;
GuestThreadMap g_guest_thread_map_;

std::mutex g_guest_thread_mutex_;

[[maybe_unused]] void ResetThreadTable(pid_t tid, GuestThread* thread) {
  std::lock_guard<std::mutex> lock(g_guest_thread_mutex_);
  g_guest_thread_map_.clear();
  g_guest_thread_map_[tid] = thread;
}

[[maybe_unused]] void InsertThread(pid_t tid, GuestThread* thread) {
  std::lock_guard<std::mutex> lock(g_guest_thread_mutex_);
  auto result = g_guest_thread_map_.insert({tid, thread});
  CHECK(result.second);
}

[[maybe_unused]] GuestThread* RemoveThread(pid_t tid) {
  std::lock_guard<std::mutex> lock(g_guest_thread_mutex_);
  auto it = g_guest_thread_map_.find(tid);
  CHECK(it != g_guest_thread_map_.end());
  GuestThread* thread = it->second;
  g_guest_thread_map_.erase(it);
  return thread;
}

[[maybe_unused]] GuestThread* FindThread(pid_t tid) {
  std::lock_guard<std::mutex> lock(g_guest_thread_mutex_);
  auto it = g_guest_thread_map_.find(tid);
  if (it == g_guest_thread_map_.end()) {
    return nullptr;
  }
  return it->second;
}

template <typename F>
[[maybe_unused]] void ForEachThread(const F& f) {
  std::lock_guard<std::mutex> lock(g_guest_thread_mutex_);
  for (const auto& v : g_guest_thread_map_) {
    f(v.first, v.second);
  }
}

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