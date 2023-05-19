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

#include <mutex>

#include "berberis/base/macros.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_signal.h"
#include "berberis/guest_os_primitives/guest_thread.h"

#include "guest_signal_action.h"

namespace berberis {

namespace {

GuestSignalAction g_signal_actions[Guest__KERNEL__NSIG];
std::mutex g_signal_actions_guard_mutex;

// Can be interrupted by another HandleHostSignal!
void HandleHostSignal(int sig, siginfo_t* info, void* context) {
  // TODO(b/283499233): Implement.
  UNUSED(sig, info, context);
}

bool IsReservedSignal(int signal) {
  switch (signal) {
    // Disallow guest action for SIGABRT to simplify debugging (b/32167022).
    case SIGABRT:
#if defined(__BIONIC__)
    // Disallow overwriting the host profiler handler from guest code. Otherwise
    // guest __libc_init_profiling_handlers() would install its own handler, which
    // is not yet supported for guest code (at least need a proxy for
    // heapprofd_client.so) and fundamentally cannot be supported for host code.
    // TODO(b/167966989): Instead intercept __libc_init_profiling_handlers.
    case BIONIC_SIGNAL_PROFILER:
#endif
      return true;
  }
  return false;
}

}  // namespace

bool GuestThread::ProcessAndDisablePendingSignals() {
  // TODO(b/280551353): Implement.
  return true;  // Previous state: enabled.
};

bool GuestThread::TestAndEnablePendingSignals() {
  // TODO(b/280551353): Implement.
  return false;  // Previous state: disabled.
};

bool SetGuestSignalHandler(int signal,
                           const Guest_sigaction* act,
                           Guest_sigaction* old_act,
                           int* error) {
  if (signal < 1 || signal > Guest__KERNEL__NSIG) {
    *error = EINVAL;
    return false;
  }

  if (act && IsReservedSignal(signal)) {
    TRACE("sigaction for reserved signal %d not set", signal);
    act = nullptr;
  }

  std::lock_guard<std::mutex> lock(g_signal_actions_guard_mutex);
  return g_signal_actions[signal - 1].Change(signal, act, HandleHostSignal, old_act, error);
}

}  // namespace berberis