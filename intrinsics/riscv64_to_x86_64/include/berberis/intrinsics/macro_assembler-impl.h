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

// Constant suitable for NaN boxing of RISC-V 32bit float with PXor.
// Note: technically we only need to Nan-box Float32 since we don't support Float16 yet.
template <typename FloatType>
extern const int32_t kNanBox;
template <>
extern const int32_t kNanBox<intrinsics::Float32>;
template <typename FloatType>
extern const int32_t kNanBoxedNans;
template <>
extern const int32_t kNanBoxedNans<intrinsics::Float32>;
template <typename FloatType>
extern const int32_t kCanonicalNans;

// Canonical NaNs. Float32 and Float64 are supported.
template <>
extern const int32_t kCanonicalNans<intrinsics::Float32>;
template <>
extern const int32_t kCanonicalNans<intrinsics::Float64>;

}  // namespace constants_pool

template <typename Assembler>
template <typename FloatType>
void MacroAssembler<Assembler>::CanonicalizeNan(XMMRegister result, XMMRegister src) {
  Pmov(result, src);
  Cmpords<FloatType>(result, src);
  Pand(src, result);
  Pandn(result, {.disp = constants_pool::kCanonicalNans<FloatType>});
  Por(result, src);
}

template <typename Assembler>
template <typename FloatType>
void MacroAssembler<Assembler>::CanonicalizeNanAVX(XMMRegister result, XMMRegister src) {
  Vcmpords<FloatType>(result, src, src);
  Vpand(src, src, result);
  Vpandn(result, result, {.disp = constants_pool::kCanonicalNans<FloatType>});
  Vpor(result, result, src);
}

template <typename Assembler>
template <typename FloatType>
void MacroAssembler<Assembler>::MacroUnboxNan(XMMRegister result, XMMRegister src) {
  static_assert(std::is_same_v<FloatType, Float32>);

  Pmov(result, src);
  Pcmpeq<typename TypeTraits<FloatType>::Int>(result, {.disp = constants_pool::kNanBox<Float32>});
  Pshufd(result, result, kShuffleDDBB);
  Pand(src, result);
  Pandn(result, {.disp = constants_pool::kNanBoxedNans<Float32>});
  Por(result, src);
}

template <typename Assembler>
template <typename FloatType>
void MacroAssembler<Assembler>::MacroUnboxNanAVX(XMMRegister result, XMMRegister src) {
  static_assert(std::is_same_v<FloatType, Float32>);

  Vpcmpeq<typename TypeTraits<FloatType>::Int>(
      result, src, {.disp = constants_pool::kNanBox<Float32>});
  Vpshufd(result, result, kShuffleDDBB);
  Vpand(src, src, result);
  Vpandn(result, result, {.disp = constants_pool::kNanBoxedNans<Float32>});
  Vpor(result, result, src);
}

}  // namespace berberis

#endif  // RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_IMPL_H_
