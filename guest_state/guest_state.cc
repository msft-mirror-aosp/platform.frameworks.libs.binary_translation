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

#include <atomic>   // std::memory_order_relaxed
#include <cstring>  // memset

namespace berberis {

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

}  // namespace berberis