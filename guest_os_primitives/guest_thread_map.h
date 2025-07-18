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

#ifndef BERBERIS_GUEST_OS_PRIMITIVES_GUEST_THREAD_MAP_H_
#define BERBERIS_GUEST_OS_PRIMITIVES_GUEST_THREAD_MAP_H_

#include <sys/types.h>  // pid_t
#include <mutex>

#include "berberis/base/forever_map.h"
#include "berberis/guest_os_primitives/guest_thread.h"

namespace berberis {

class GuestThreadMap {
 public:
  static GuestThreadMap* GetInstance();

  void ResetThreadTable(pid_t tid, GuestThread* thread);
  void InsertThread(pid_t tid, GuestThread* thread);
  GuestThread* RemoveThread(pid_t tid);
  GuestThread* FindThread(pid_t tid);

  // TODO(b/280551726): Replace template with function that accepts a function arg.
  template <typename F>
  void ForEachThread(const F& f) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& v : map_) {
      f(v.first, v.second);
    }
  }

 private:
  GuestThreadMap() = default;
  GuestThreadMap(const GuestThreadMap&) = delete;
  GuestThreadMap& operator=(const GuestThreadMap&) = delete;

  friend GuestThreadMap* NewForever<GuestThreadMap>();

  ForeverMap<pid_t, GuestThread*> map_;
  std::mutex mutex_;
};

}  // namespace berberis

#endif  // BERBERIS_GUEST_OS_PRIMITIVES_GUEST_THREAD_MAP_H_
