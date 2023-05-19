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

#include <atomic>
#include <mutex>

#include "berberis/base/macros.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_signal.h"
#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_state/guest_state_opaque.h"

#include "guest_signal_action.h"

namespace berberis {

namespace {

GuestSignalAction g_signal_actions[Guest__KERNEL__NSIG];
std::mutex g_signal_actions_guard_mutex;

const Guest_sigaction* FindSignalHandler(int signal) {
  CHECK_GT(signal, 0);
  CHECK_LE(signal, Guest__KERNEL__NSIG);
  std::lock_guard<std::mutex> lock(g_signal_actions_guard_mutex);
  return &g_signal_actions[signal - 1].GetClaimedGuestAction();
}

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

void ProcessGuestSignal(GuestThread* thread, const Guest_sigaction* sa, siginfo_t* info) {
  // TODO(b/283499233): Implement.
  UNUSED(thread, sa, info);
}

}  // namespace

bool GuestThread::ProcessAndDisablePendingSignals() {
  for (;;) {
    // If pending signals are not present, cas should disable them.
    // Otherwise, process pending signals and try again.
    uint8_t old_status = kPendingSignalsEnabled;
    if (GetPendingSignalsStatusAtomic(state_)->compare_exchange_weak(
            old_status, kPendingSignalsDisabled, std::memory_order_acq_rel)) {
      return true;
    }
    if (old_status == kPendingSignalsDisabled) {
      return false;
    }
    ProcessPendingSignalsImpl();
  }
}

bool GuestThread::TestAndEnablePendingSignals() {
  // If pending signals are disabled, cas should mark them enabled.
  // Otherwise, pending signals are already enabled.
  uint8_t old_status = kPendingSignalsDisabled;
  return !GetPendingSignalsStatusAtomic(state_)->compare_exchange_strong(
      old_status, kPendingSignalsEnabled, std::memory_order_acq_rel);
}

// Return if another iteration is needed.
// ATTENTION: Can be interrupted by SetSignal!
void GuestThread::ProcessPendingSignalsImpl() {
  // Clear pending signals status and queue.
  // ATTENTION: It is important to change status before the queue!
  // Otherwise if interrupted by SetSignal, we might end up with
  // no pending signals status but with non-empty queue!
  GetPendingSignalsStatusAtomic(state_)->store(kPendingSignalsEnabled, std::memory_order_relaxed);

  siginfo_t* signal_info;
  while ((signal_info = pending_signals_.DequeueSignalUnsafe())) {
    const Guest_sigaction* sa = FindSignalHandler(signal_info->si_signo);
    ProcessGuestSignal(this, sa, signal_info);
    pending_signals_.FreeSignal(signal_info);
  }
}

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