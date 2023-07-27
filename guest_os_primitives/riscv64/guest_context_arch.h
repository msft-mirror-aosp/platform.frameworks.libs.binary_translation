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

#ifndef BERBERIS_GUEST_OS_PRIMITIVES_GUEST_CONTEXT_ARCH_H_
#define BERBERIS_GUEST_OS_PRIMITIVES_GUEST_CONTEXT_ARCH_H_

#include "berberis/base/checks.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

// TODO(b/283499233): Properly implement this class for riscv64.
class GuestContext {
 public:
  GuestContext() = default;
  GuestContext(const GuestContext&) = delete;
  GuestContext& operator=(const GuestContext&) = delete;

  void Save(const CPUState* /*cpu*/) { FATAL("unimplemented"); }
  void Restore(CPUState* /*cpu*/) const { FATAL("unimplemented"); }

  void* ptr() { return &ctx_; }

 private:
  struct Guest_ucontext;
  Guest_ucontext* ctx_ = nullptr;

  CPUState* cpu_ = nullptr;
};

}  // namespace berberis

#endif  // BERBERIS_GUEST_OS_PRIMITIVES_GUEST_CONTEXT_ARCH_H_