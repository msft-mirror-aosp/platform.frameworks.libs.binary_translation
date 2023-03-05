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

#ifndef BERBERIS_GUEST_STATE_GUEST_STATE_RISCV64_H_
#define BERBERIS_GUEST_STATE_GUEST_STATE_RISCV64_H_

#include <cstdint>

#include "berberis/base/macros.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

struct CPUState {
  // x1 to x31.
  uint64_t x[31];
  GuestAddr insn_addr;
};

template <uint8_t kIndex>
inline uint64_t GetXReg(const CPUState& state) {
  static_assert(kIndex > 0);
  static_assert((kIndex - 1) < arraysize(state.x));
  return state.x[kIndex - 1];
}

template <uint8_t kIndex>
inline void SetXReg(CPUState& state, uint64_t val) {
  static_assert(kIndex > 0);
  static_assert((kIndex - 1) < arraysize(state.x));
  state.x[kIndex - 1] = val;
}

struct ThreadState {
  CPUState cpu;
};

}  // namespace berberis

#endif  // BERBERIS_GUEST_STATE_GUEST_STATE_RISCV64_H_
