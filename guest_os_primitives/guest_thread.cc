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
#include "berberis/base/macros.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_os_primitives/guest_thread_manager.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/instrument/instrument.h"

#include "guest_thread_manager_impl.h"
#include "scoped_signal_blocker.h"

namespace berberis {

// Manages the current guest thread.
pthread_key_t g_guest_thread_key;

namespace {

// TODO(b/281746270): Consider consolidating these compile-time constants into single location.
// Size of stack for a single guest call.
constexpr size_t kGuestStackSize = 2 * 1024 * 1024;
// Size of stack guard buffer. Same as host's page size. 4K on all systems so far.
constexpr size_t kGuestStackGuardSize = 4 * 1024;

typedef ForeverMap<pid_t, GuestThread*> GuestThreadMap;
GuestThreadMap g_guest_thread_map_;

std::mutex g_guest_thread_mutex_;

[[maybe_unused]] void ResetThreadTable(pid_t tid, GuestThread* thread) {
  std::lock_guard<std::mutex> lock(g_guest_thread_mutex_);
  g_guest_thread_map_.clear();
  g_guest_thread_map_[tid] = thread;
}

void InsertThread(pid_t tid, GuestThread* thread) {
  std::lock_guard<std::mutex> lock(g_guest_thread_mutex_);
  auto result = g_guest_thread_map_.insert({tid, thread});
  CHECK(result.second);
}

GuestThread* RemoveThread(pid_t tid) {
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
  // TLS cache was cleared by pthread_exit.
  // TODO(b/280671643): Postpone detach to last pthread destructor iteration.
  // On previous iterations, simply restore TLS cache and return.
  DetachCurrentThread();
}

}  // namespace

// Not thread safe, not async signals safe!
void InitGuestThreadManager() {
  // Here we don't need pthread_once, which is not reentrant due to spinlocks.
  CHECK_EQ(0, pthread_key_create(&g_guest_thread_key, GuestThreadDtor));
}

// Common guest thread function attaches GuestThread lazily on first call and detaches in pthread
// key destructor (register_dtor = true).
//
// Guest signal handlers and guest pthread key destructors are special as they might be called when
// GuestThread is not yet attached or is already detached. Moreover, they cannot determine between
// latter cases. Thus, signal handlers and key destructors reuse GuestThread if it is attached,
// otherwise they attach AND detach themselves, so GuestThread attach state is preserved and
// GuestThread is never leaked (register_dtor = false).
//
// ATTENTION: When signal handler or key destructor attach GuestThread themselves, they might get
// GuestThread stack different from one used in thread function. It might confuse several
// (ill-formed?) apks, so we issue a warning.
//
// ATTENTION: Can be interrupted!
GuestThread* AttachCurrentThread(bool register_dtor, bool* attached) {
  // The following code is not reentrant!
  ScopedSignalBlocker signal_blocker;

  pid_t tid = GettidSyscall();
  GuestThread* thread = FindThread(tid);
  if (thread) {
    // Thread was already attached.
    *attached = false;
    return thread;
  }

  // Copy host stack size attributes.
  // TODO(b/30124680): pthread_getattr_np is bionic-only, what about glibc?
  size_t stack_size = kGuestStackSize;
  size_t guard_size = kGuestStackGuardSize;
#if defined(__BIONIC__)
  pthread_attr_t attr;
  CHECK_EQ(0, pthread_getattr_np(pthread_self(), &attr));
  CHECK_EQ(0, pthread_attr_getstacksize(&attr, &stack_size));
  CHECK_EQ(0, pthread_attr_getguardsize(&attr, &guard_size));
#endif
  thread = GuestThread::CreatePthread(nullptr, stack_size, guard_size);
  CHECK(thread);

  InsertCurrentThread(thread, register_dtor);
  thread->InitStaticTls();

  // If thread is attached in HandleHostSignal we must run guest handler
  // immediately because we detach guest thread before exit from HandleHostSignal.
  // All non-reentrant code in runtime must be protected with ScopedPendingSignalsEnabler.
  thread->state()->pending_signals_status = kPendingSignalsDisabled;
  // AttachCurrentThread is never called from generated code.
  thread->state()->residence = kOutsideGeneratedCode;

  *attached = true;
  return thread;
}

void InsertCurrentThread(GuestThread* thread, bool register_dtor) {
  pid_t tid = GettidSyscall();

  // The following code is not reentrant!
  ScopedSignalBlocker signal_blocker;

  // Thread should not be already in the table!
  // If signal came after we checked tls cache or table but before we blocked signals, it should
  // have attached AND detached the thread!
  InsertThread(tid, thread);
  if (register_dtor) {
    CHECK_EQ(0, pthread_setspecific(g_guest_thread_key, thread));
  }
  if (kInstrumentGuestThread) {
    // TODO(b/280498513): Call instrumentation hook(s) here.
  }

  TRACE("guest thread attached %d", tid);
}

// ATTENTION: Can be interrupted!
void DetachCurrentThread() {
  pid_t tid = GettidSyscall();

  // The following code is not reentrant!
  ScopedSignalBlocker signal_blocker;

  // Remove thread from global table.
  GuestThread* thread = RemoveThread(tid);
  if (kInstrumentGuestThread) {
    // TODO(b/280498513): Call instrumentation hook(s) here.
  }

  TRACE("guest thread detached %d", tid);
  GuestThread::Destroy(thread);
}

GuestThread* GuestThread::CreatePthread(void* stack, size_t stack_size, size_t guard_size) {
  // TODO(b/280551726): Implement.
  UNUSED(stack, stack_size, guard_size);
  return nullptr;
}

void GuestThread::Destroy(GuestThread* thread) {
  // TODO(b/280551726): Implement.
  UNUSED(thread);
}

void GuestThread::InitStaticTls() {
  // TODO(b/280551726): Implement.
}

}  // namespace berberis