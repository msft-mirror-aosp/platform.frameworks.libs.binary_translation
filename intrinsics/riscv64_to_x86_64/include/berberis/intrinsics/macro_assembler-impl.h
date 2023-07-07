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

#ifndef RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_IMPL_H_
#define RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_IMPL_H_

#include "berberis/base/bit_util.h"
#include "berberis/intrinsics/macro_assembler.h"

namespace berberis {

namespace constants_pool {

// Constant suitable for NaNBoxing of RISC-V 32bit float with PXor.
extern const int32_t kNanBoxFloat32;
extern const int32_t kNanBoxedNaNsFloat32;

}  // namespace constants_pool

template <typename Assembler>
template <typename FloatType>
void MacroAssembler<Assembler>::MacroUnboxNaN(XMMRegister result, XMMRegister src) {
  static_assert(std::is_same_v<FloatType, intrinsics::Float32>);

  Pmov(result, src);
  Pcmpeq<typename TypeTraits<FloatType>::Int>(result, {.disp = constants_pool::kNanBoxFloat32});
  Pshufd(result, result, kShuffleDDBB);
  Pand(src, result);
  Pandn(result, {.disp = constants_pool::kNanBoxedNaNsFloat32});
  Por(result, src);
}

template <typename Assembler>
template <typename FloatType>
void MacroAssembler<Assembler>::MacroUnboxNaNAVX(XMMRegister result, XMMRegister src) {
  static_assert(std::is_same_v<FloatType, intrinsics::Float32>);

  Vpcmpeq<typename TypeTraits<FloatType>::Int>(result, src, {.disp = constants_pool::kNanBoxFloat32});
  Vpshufd(result, result, kShuffleDDBB);
  Vpand(src, src, result);
  Vpandn(result, result, {.disp = constants_pool::kNanBoxedNaNsFloat32});
  Vpor(result, result, src);
}

}  // namespace berberis

#endif  // RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_IMPL_H_
