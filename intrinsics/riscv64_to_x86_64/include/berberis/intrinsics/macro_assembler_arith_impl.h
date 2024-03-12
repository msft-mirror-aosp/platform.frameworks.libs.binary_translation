/*
 * Copyright (C) 2020 The Android Open Source Project
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

#ifndef RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_ARITH_IMPL_H_
#define RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_ARITH_IMPL_H_

#include <climits>
#include <type_traits>

#include "berberis/base/bit_util.h"
#include "berberis/intrinsics/macro_assembler.h"

namespace berberis {

// Divisor comes in "src", dividend comes in gpr_a, result is returned in gpr_a.
// gpr_d and FLAGS are clobbered by that macroinstruction.
template <typename Assembler>
template <typename IntType>
void MacroAssembler<Assembler>::MacroDiv(Register src) {
  Label* zero = MakeLabel();
  Label* done = MakeLabel();
  Test<IntType>(src, src);
  Jcc(Condition::kZero, *zero);

  if constexpr (std::is_signed_v<IntType>) {
    Label* do_idiv = MakeLabel();
    // If min int32_t/int64_t is divided by -1 then in risc-v the result is
    // the dividend, but x86 will raise an exception. Handle this case separately.
    Cmp<IntType>(src, int8_t{-1});
    Jcc(Condition::kNotEqual, *do_idiv);

    if constexpr (std::is_same_v<IntType, int64_t>) {
      Cmp<IntType>(gpr_a,
                   {.disp = constants_pool::kVectorConst<std::numeric_limits<IntType>::min()>});
    } else {
      Cmp<IntType>(gpr_a, std::numeric_limits<IntType>::min());
    }
    Jcc(Condition::kEqual, *done);

    Bind(do_idiv);
    // We need to sign-extend eax into edx to ensure 64-bit/128-bit dividend is correct.
    if constexpr (std::is_same_v<IntType, int32_t>) {
      Cdq();
    } else if constexpr (std::is_same_v<IntType, int64_t>) {
      // 32bit assembler wouldn't have Cqo, but because of constexpr we are not seeking it there.
      Cqo();
    } else {
      static_assert(kDependentTypeFalse<IntType>, "Unsupported format");
    }
  } else {
    // We need to zero-extend eax into edx to ensure 64-bit/128-bit dividend is correct.
    Xor<uint32_t>(gpr_d, gpr_d);
  }

  Div<IntType>(src);
  Jmp(*done);

  Bind(zero);
  Mov<IntType>(gpr_a, int64_t{-1});

  Bind(done);
}
}  // namespace berberis

#endif  // RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_ARITH_IMPL_H_
