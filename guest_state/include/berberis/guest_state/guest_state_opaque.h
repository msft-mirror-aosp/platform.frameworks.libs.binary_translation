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

#ifndef BERBERIS_GUEST_STATE_GUEST_STATE_OPAQUE_H_
#define BERBERIS_GUEST_STATE_GUEST_STATE_OPAQUE_H_

#include <cstdint>

#include "berberis/guest_state/guest_addr.h"

namespace berberis {

struct CPUState;
struct ThreadState;

ThreadState* CreateThreadState();
void DestroyThreadState(ThreadState* state);

class GuestThread;
void SetGuestThread(ThreadState* state, GuestThread* thread);
GuestThread* GetGuestThread(const ThreadState* state);

// Track whether we are in generated code or not.
enum GuestThreadResidence : uint8_t {
  kOutsideGeneratedCode = 0,
  kInsideGeneratedCode = 1,
};

GuestThreadResidence GetResidence(ThreadState* state);
void SetResidence(ThreadState* state, GuestThreadResidence residence);

// TODO(b/28058920): Refactor into GuestThread.
// Pending signals status state machine:
//   disabled <-> enabled <-> enabled and pending signals present
enum PendingSignalsStatus : uint8_t {
  kPendingSignalsDisabled = 0,  // initial value, must be 0
  kPendingSignalsEnabled,
  kPendingSignalsPresent,  // implies enabled
};

void SetPendingSignalsStatus(ThreadState* state, PendingSignalsStatus status);

CPUState* GetCPUState(ThreadState* state);

void SetLinkRegister(CPUState* cpu, GuestAddr val);

GuestAddr GetLinkRegister(const CPUState* cpu);

void SetInsnAddr(CPUState* cpu, GuestAddr addr);

GuestAddr GetInsnAddr(const CPUState* cpu);

// TODO(b/28058920): Refactor into GuestThread.
bool ArePendingSignalsPresent(const ThreadState* state);

}  // namespace berberis

#endif  // BERBERIS_GUEST_STATE_GUEST_STATE_OPAQUE_H_
