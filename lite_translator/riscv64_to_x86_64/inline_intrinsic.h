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

#ifndef BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_INLINE_INTRINSIC_H_
#define BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_INLINE_INTRINSIC_H_

#include <cstdint>
#include <tuple>
#include <type_traits>

#include "berberis/intrinsics/guest_fp_flags.h"

#include "berberis/assembler/x86_64.h"

namespace berberis {

template <auto kFunc>
class InlineIntrinsic {
 public:
  template <typename RegAllocator, typename ResType, typename... ArgType>
  static bool TryInline(MacroAssembler<x86_64::Assembler>* as,
                        RegAllocator* reg_alloc,
                        ResType result,
                        ArgType... args) {
    std::tuple args_tuple = std::make_tuple(args...);
    if constexpr (IsTagEq<&intrinsics::FMul<intrinsics::Float64>>) {
      auto dst = result;
      auto [rm, frm, src1, src2] = args_tuple;

      if (rm == FPFlags::DYN) {
        return false;
      }

      as->Movdqa(dst, src1);
      as->Mulsd(dst, src2);
      return true;
    }
    // reg_alloc does nothing for now, is used for later implemented instructions
    UNUSED(reg_alloc);
    return false;
  }

 private:
  template <auto kFunction>
  class FunctionCompareTag;

  template <auto kOtherFunction>
  static constexpr bool IsTagEq =
      std::is_same_v<FunctionCompareTag<kFunc>, FunctionCompareTag<kOtherFunction>>;
};

template <auto kFunction, typename RegAllocator, typename ResType, typename... ArgType>
bool TryInlineIntrinsic(MacroAssembler<x86_64::Assembler>* as,
                        RegAllocator* reg_alloc,
                        ResType result,
                        ArgType... args) {
  return InlineIntrinsic<kFunction>::TryInline(as, reg_alloc, result, args...);
}

}  // namespace berberis

#endif  // BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_CALL_INTRINSIC_H_
