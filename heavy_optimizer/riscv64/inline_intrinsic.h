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

#ifndef BERBERIS_HEAVY_OPTIMIZER_RISCV64_INLINE_INTRINSIC_H_
#define BERBERIS_HEAVY_OPTIMIZER_RISCV64_INLINE_INTRINSIC_H_

#include <cfenv>
#include <cstdint>
#include <type_traits>

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"

namespace berberis {

template <auto kFunc>
class InlineIntrinsic {
 public:
  template <typename ResType, typename FlagRegister, typename... ArgType>
  static bool TryInline(x86_64::MachineIRBuilder* /* builder */,
                        ResType /* result */,
                        FlagRegister /* flag_register */,
                        ArgType... /* args */) {
    // TODO(b/291126189) Implement intrinsics
    return false;
  }

  template <typename FlagRegister, typename... ArgType>
  static bool TryInline(x86_64::MachineIRBuilder* /* builder */,
                        FlagRegister /* flag_register */,
                        ArgType... /* args */) {
    // TODO(b/291126189) Implement intrinsics
    return false;
  }

 private:
  // Comparison of pointers which point to different functions is generally not a
  // constexpr since such functions can be merged in object code (comparing
  // pointers to the same function is constexpr). This helper compares them using
  // templates explicitly telling that we are not worried about such subtleties here.
  template <auto kFunction>
  class FunctionCompareTag;

  // Note, if we define it as a variable clang doesn't consider it a constexpr in TryInline().
  template <auto kOtherFunction>
  static constexpr bool IsTagEq() {
    return std::is_same_v<FunctionCompareTag<kFunc>, FunctionCompareTag<kOtherFunction>>;
  }
};

template <auto kFunction, typename ResType, typename FlagRegister, typename... ArgType>
bool TryInlineIntrinsicForHeavyOptimizer(x86_64::MachineIRBuilder* builder,
                                         ResType result,
                                         FlagRegister flag_register,
                                         ArgType... args) {
  return InlineIntrinsic<kFunction>::TryInline(builder, result, flag_register, args...);
}

template <auto kFunction, typename FlagRegister, typename... ArgType>
bool TryInlineIntrinsicForHeavyOptimizer(x86_64::MachineIRBuilder* builder,
                                         FlagRegister flag_register,
                                         ArgType... args) {
  return InlineIntrinsic<kFunction>::TryInline(builder, flag_register, args...);
}

}  // namespace berberis

#endif  // BERBERIS_HEAVY_OPTIMIZER_RISCV64_INLINE_INTRINSIC_H_