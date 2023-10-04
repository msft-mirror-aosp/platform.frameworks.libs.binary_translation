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

#ifndef RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_CONSTANTS_POOL_H_
#define RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_CONSTANTS_POOL_H_

#include "berberis/base/bit_util.h"
#include "berberis/intrinsics/macro_assembler.h"

namespace berberis::constants_pool {

// Constant suitable for NaN boxing of RISC-V 32bit float with PXor.
// Note: technically we only need to Nan-box Float32 since we don't support Float16 yet.
template <typename FloatType>
extern const int32_t kNanBox;
template <>
extern const int32_t kNanBox<intrinsics::Float32>;

// Canonically Nan boxed canonical NaN.
// Note: technically we only need to Nan-box Float32 since we don't support Float16 yet.
template <typename FloatType>
extern const int32_t kNanBoxedNans;
template <>
extern const int32_t kNanBoxedNans<intrinsics::Float32>;

// Canonical NaNs. Float32 and Float64 are supported.
template <typename FloatType>
extern const int32_t kCanonicalNans;
template <>
extern const int32_t kCanonicalNans<intrinsics::Float32>;
template <>
extern const int32_t kCanonicalNans<intrinsics::Float64>;

// Helper constant for BsrToClz conversion. 63 for int32_t, 127 for int64_t.
template <typename IntType>
extern const int32_t kBsrToClz;
template <>
extern const int32_t kBsrToClz<int32_t>;
template <>
extern const int32_t kBsrToClz<int64_t>;

// Helper constant for width of the type. 32 for int32_t, 64 for int64_t.
template <typename IntType>
extern const int32_t kWidthInBits;
template <>
extern const int32_t kWidthInBits<int32_t>;
template <>
extern const int32_t kWidthInBits<int64_t>;

extern const int32_t kZero;
extern const int32_t kMaxUInt;

// 64 bit constants for use with arithmetic operations.
// Used because only 32 bit immediates are supported on x86-64.
template <auto Value>
extern const int32_t kConst;
template <>
extern const int32_t kConst<uint64_t{0x8000'0000'0000'00ff}>;

extern const int32_t kRiscVToX87Exceptions;
extern const int32_t kX87ToRiscVExceptions;

}  // namespace berberis::constants_pool

#endif  // RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_CONSTANTS_POOL_H_
