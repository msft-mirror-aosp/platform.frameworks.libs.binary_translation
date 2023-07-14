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

#ifndef RISCV64_TO_X86_64_NDK_TRANSLATION_INTRINSICS_TEXT_ASSEMBLER_H_
#define RISCV64_TO_X86_64_NDK_TRANSLATION_INTRINSICS_TEXT_ASSEMBLER_H_

#include <stdio.h>

#include "berberis/intrinsics/common_to_x86/text_assembler_common.h"

namespace berberis {

class TextAssembler : public TextAssemblerX86<TextAssembler> {
 public:
  TextAssembler(int indent, FILE* out) : TextAssemblerX86(indent, out) {}

// Instructions.
#include "gen_text_assembler_x86_64-inl.h"  // NOLINT generated file

  static constexpr bool need_gpr_macroassembler_mxcsr_scratch() {
    return need_gpr_macroassembler_mxcsr_scratch_;
  }

  // Unhide Movq(Mem, XMMReg) and Movq(XMMReg, Mem) hidden by Movq(Reg, Imm) and many others.
  using TextAssemblerX86::Movq;

  static constexpr char kArchName[] = "RISCV64_TO_X86_64";
  static constexpr char kNamespaceName[] = "berberis";

 protected:
  static constexpr bool need_gpr_macroassembler_mxcsr_scratch_ = false;
  typedef RegisterTemplate<kEsp, false> RegisterDefaultBit;

 private:
  using Assembler = TextAssembler;
  DISALLOW_IMPLICIT_CONSTRUCTORS(TextAssembler);
  friend TextAssemblerX86;
};

void MakeExtraGuestFunctions(FILE*) {}

}  // namespace berberis

#endif  // RISCV64_TO_X86_64_NDK_TRANSLATION_INTRINSICS_TEXT_ASSEMBLER_H_
