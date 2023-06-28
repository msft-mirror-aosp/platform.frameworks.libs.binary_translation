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

#include "berberis/guest_state/guest_state_opaque.h"

#include "berberis/base/checks.h"
#include "berberis/base/mmap.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"

#include <atomic>   // std::memory_order_relaxed
#include <cstddef>  // size_t
#include <cstdint>  // uint_least8_t
#include <cstring>  // memset

namespace berberis {

namespace {

constexpr size_t kThreadStatePageAlignedSize = AlignUpPageSize(sizeof(ThreadState));

void InitThreadState(ThreadState* state) {
  // This is needed to set all flag values to 0.
  memset(&(state->cpu), 0, sizeof(CPUState));

  // TODO(b/281864904): Add architecture-defined hook for initializing host MXCSR.

  // ATTENTION: Set fields specific for current thread when actually attaching to host thread!
  state->thread = nullptr;
  // TODO(b/281864904): Initialize TLS pointer to zero here.

  state->pending_signals_status.store(kPendingSignalsDisabled, std::memory_order_relaxed);
  state->residence = kOutsideGeneratedCode;
  state->instrument_data = nullptr;
}

}  // namespace

ThreadState* CreateThreadState() {
  void* storage = Mmap(kThreadStatePageAlignedSize);
  if (storage == MAP_FAILED) {
    return nullptr;
  }
  ThreadState* state = new (storage) ThreadState;
  CHECK(state);

  InitThreadState(state);
  return state;
};

void DestroyThreadState(ThreadState* state) {
  CHECK(state);
  MunmapOrDie(state, kThreadStatePageAlignedSize);
}

class GuestThread;
void SetGuestThread(ThreadState* state, GuestThread* thread) {
  state->thread = thread;
}

GuestThread* GetGuestThread(const ThreadState* state) {
  return state->thread;
}

GuestThreadResidence GetResidence(ThreadState* state) {
  return state->residence;
}

void SetResidence(ThreadState* state, GuestThreadResidence residence) {
  state->residence = residence;
}

std::atomic<uint_least8_t>* GetPendingSignalsStatusAtomic(ThreadState* state) {
  return &state->pending_signals_status;
}

bool ArePendingSignalsPresent(const ThreadState* state) {
  return state->pending_signals_status.load(std::memory_order_relaxed) == kPendingSignalsPresent;
}

CPUState* GetCPUState(ThreadState* state) {
  return &state->cpu;
}

void SetStackRegister(CPUState* cpu, GuestAddr val) {
  SetXReg<SP>(*cpu, val);
}

GuestAddr GetStackRegister(CPUState* cpu) {
  return GetXReg<SP>(*cpu);
}

void SetLinkRegister(CPUState* cpu, GuestAddr val) {
  SetXReg<RA>(*cpu, val);
}

GuestAddr GetLinkRegister(const CPUState* cpu) {
  return GetXReg<RA>(*cpu);
}

void SetInsnAddr(CPUState* cpu, GuestAddr addr) {
  cpu->insn_addr = addr;
}

GuestAddr GetInsnAddr(const CPUState* cpu) {
  return cpu->insn_addr;
}

}  // namespace berberis
