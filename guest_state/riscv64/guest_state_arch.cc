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

#include "berberis/guest_state/guest_state.h"

namespace berberis {

void SetStackRegister(CPUState* cpu, GuestAddr val) {
  SetXReg<SP>(*cpu, val);
}

GuestAddr GetStackRegister(CPUState* cpu) {
  return GetXReg<SP>(*cpu);
}

void SetLinkRegister(CPUState* cpu, GuestAddr val) {
  SetXReg<RA>(*cpu, val);
}

GuestAddr GetLinkRegister(const CPUState& cpu) {
  return GetXReg<RA>(cpu);
}

void SetTlsAddr(ThreadState* state, GuestAddr addr) {
  SetXReg<TP>(state->cpu, addr);
}

GuestAddr GetTlsAddr(const ThreadState* state) {
  return GetXReg<TP>(state->cpu);
}

void InitFloatingPointState() {
  // TODO(b/276787675): Initialize host MXCSR register once riscv64 intrinsics are supported.
}

}  // namespace berberis