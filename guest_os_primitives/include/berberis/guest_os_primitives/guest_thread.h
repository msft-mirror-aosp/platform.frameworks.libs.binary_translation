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

#ifndef BERBERIS_GUEST_OS_PRIMITIVES_GUEST_THREAD_H_
#define BERBERIS_GUEST_OS_PRIMITIVES_GUEST_THREAD_H_

#include <setjmp.h>  // jmp_buf

#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/signal_queue.h"

struct NativeBridgeStaticTlsConfig;

namespace berberis {

struct GuestCallExecution {
  GuestCallExecution* parent = nullptr;
  GuestAddr sp = 0;
  jmp_buf buf;
};

// ATTENTION: GuestThread object can only be used by the current thread!
class GuestThread {
 public:
  static GuestThread* CreatePthread(void* stack, size_t stack_size, size_t guard_size);
  static void Destroy(GuestThread* thread);

  // Initialize *current* guest thread.
  void InitStaticTls();

  const ThreadState* state() const { return &state_; }
  ThreadState* state() { return &state_; }

 private:
  GuestThread() = default;
  static GuestThread* Create();

  bool AllocStack(void* stack, size_t stack_size, size_t guard_size);
  bool AllocShadowCallStack();
  bool AllocStaticTls();

  // Host stack. Valid for cloned threads only.
  void* host_stack_ = nullptr;

  // Stack.
  void* stack_ = nullptr;
  size_t stack_size_ = 0;
  size_t guard_size_ = 0;
  size_t mmap_size_ = 0;
  GuestAddr stack_top_ = 0;

  void* static_tls_ = nullptr;

  // Shadow call stack.
  void* scs_region_ = nullptr;
  GuestAddr scs_base_ = 0;

  ThreadState state_;

  SignalQueue pending_signals_;

  GuestCallExecution* guest_call_execution_ = nullptr;

  void* sig_alt_stack_ = nullptr;
  size_t sig_alt_stack_size_ = 0;
};

}  // namespace berberis

#endif  // BERBERIS_GUEST_OS_PRIMITIVES_GUEST_THREAD_H_