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

#ifndef BERBERIS_LITE_TRANSLATOR_RISCV64_ALLOCATOR_H_
#define BERBERIS_LITE_TRANSLATOR_RISCV64_ALLOCATOR_H_

#include <algorithm>  // std::max
#include <optional>

#include "berberis/assembler/x86_64.h"

namespace berberis {

template <typename RegType>
class Allocator;

template <>
class Allocator<x86_64::Assembler::Register> {
 public:
  std::optional<x86_64::Assembler::Register> Alloc() {
    if (regs_allocated_ + max_temp_regs_allocated >= kNumRegister) {
      return std::nullopt;
    }
    return std::optional<x86_64::Assembler::Register>(kRegs[regs_allocated_++]);
  }

  std::optional<x86_64::Assembler::Register> AllocTemp() {
    if (regs_allocated_ + temp_regs_allocated >= kNumRegister) {
      return std::nullopt;
    }
    auto res =
        std::optional<x86_64::Assembler::Register>(kRegs[kNumRegister - 1 - temp_regs_allocated]);
    temp_regs_allocated++;
    max_temp_regs_allocated = std::max(max_temp_regs_allocated, temp_regs_allocated);
    return res;
  }

  void FreeTemps() { temp_regs_allocated = 0; }

 private:
  // TODO(286261771): Add rdx to registers, push it on stack in all instances that are clobbering
  // it.
  static constexpr x86_64::Assembler::Register kRegs[] = {x86_64::Assembler::rbx,
                                                          x86_64::Assembler::rsi,
                                                          x86_64::Assembler::rdi,
                                                          x86_64::Assembler::r8,
                                                          x86_64::Assembler::r9,
                                                          x86_64::Assembler::r10,
                                                          x86_64::Assembler::r11,
                                                          x86_64::Assembler::r12,
                                                          x86_64::Assembler::r13,
                                                          x86_64::Assembler::r14,
                                                          x86_64::Assembler::r15};
  static constexpr unsigned kNumRegister = arraysize(kRegs);

  unsigned int regs_allocated_ = 0;
  unsigned int temp_regs_allocated = 0;
  unsigned int max_temp_regs_allocated = 0;
};

}  // namespace berberis

#endif  // BERBERIS_LITE_TRANSLATOR_RISCV64_ALLOCATOR_H_
