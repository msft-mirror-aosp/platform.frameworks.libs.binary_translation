/*
 * Copyright (C) 2024 The Android Open Source Project
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

// Once JIT is ready, this file should be automatically generated by
// gen_text_asm_intrinsics.cc

#ifndef RISCV64_TO_ARM64_BERBERIS_INTRINSICS_H_
#define RISCV64_TO_ARM64_BERBERIS_INTRINSICS_H_

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "berberis/intrinsics/riscv64_to_all/intrinsics.h"

namespace berberis {

namespace intrinsics {

inline uint64_t ShiftedOne(uint64_t shift_amount) {
  return uint64_t{1} << (shift_amount % 64);
}

inline std::tuple<uint64_t> Bclr(uint64_t in1, uint64_t in2) {
  // Clear the specified bit.
  return {in1 & ~ShiftedOne(in2)};
};

inline std::tuple<uint64_t> Bext(uint64_t in1, uint64_t in2) {
  // Return whether the bit is set.
  return {(in1 & ShiftedOne(in2)) ? 1 : 0};
};

inline std::tuple<uint64_t> Binv(uint64_t in1, uint64_t in2) {
  // Toggle the specified bit.
  return {in1 ^ ShiftedOne(in2)};
};

inline std::tuple<uint64_t> Bset(uint64_t in1, uint64_t in2) {
  // Set the specified bit.
  return {in1 | ShiftedOne(in2)};
};

template <typename T, enum PreferredIntrinsicsImplementation>
inline std::tuple<T> Div(T in1, T in2) {
  static_assert(std::is_integral_v<T>);

  if (in2 == 0) {
    return ~T{0};
  } else if (std::is_signed_v<T> && in2 == -1 && in1 == std::numeric_limits<T>::min()) {
    return {std::numeric_limits<T>::min()};
  }
  return {in1 / in2};
};

template <typename T, enum PreferredIntrinsicsImplementation>
inline std::tuple<T> Max(T in1, T in2) {
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>);
  return {std::max(in1, in2)};
};

template <typename T, enum PreferredIntrinsicsImplementation>
inline std::tuple<T> Min(T in1, T in2) {
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>);
  return {std::min(in1, in2)};
};

template <typename T, enum PreferredIntrinsicsImplementation>
inline std::tuple<T> Rem(T in1, T in2) {
  static_assert(std::is_integral_v<T>);

  if (in2 == 0) {
    return {in1};
  } else if (std::is_signed_v<T> && in2 == -1 && in1 == std::numeric_limits<T>::min()) {
    return {0};
  }
  return {in1 % in2};
};

inline std::tuple<uint64_t> Rev8(uint64_t in1) {
  return {__builtin_bswap64(in1)};
};

template <typename T, enum PreferredIntrinsicsImplementation>
inline std::tuple<T> Rol(T in1, int8_t in2) {
  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>);
  // We need unsigned shifts, so that shifted-in bits are filled with zeroes.
  if (std::is_same_v<T, int32_t>) {
    return {(static_cast<uint32_t>(in1) << (in2 % 32)) |
            (static_cast<uint32_t>(in1) >> (32 - (in2 % 32)))};
  } else {
    return {(static_cast<uint64_t>(in1) << (in2 % 64)) |
            (static_cast<uint64_t>(in1) >> (64 - (in2 % 64)))};
  }
};

template <typename T, enum PreferredIntrinsicsImplementation>
inline std::tuple<T> Ror(T in1, int8_t in2) {
  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>);
  // We need unsigned shifts, so that shifted-in bits are filled with zeroes.
  if (std::is_same_v<T, int32_t>) {
    return {(static_cast<uint32_t>(in1) >> (in2 % 32)) |
            (static_cast<uint32_t>(in1) << (32 - (in2 % 32)))};
  } else {
    return {(static_cast<uint64_t>(in1) >> (in2 % 64)) |
            (static_cast<uint64_t>(in1) << (64 - (in2 % 64)))};
  }
};

template <typename T, enum PreferredIntrinsicsImplementation>
inline std::tuple<int64_t> Sext(T in1) {
  static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>);
  return {static_cast<int64_t>(in1)};
};

inline std::tuple<uint64_t> Sh1add(uint64_t in1, uint64_t in2) {
  return {uint64_t{in1} * 2 + in2};
};

inline std::tuple<uint64_t> Sh1adduw(uint32_t in1, uint64_t in2) {
  return Sh1add(uint64_t{in1}, in2);
};

inline std::tuple<uint64_t> Sh2add(uint64_t in1, uint64_t in2) {
  return {uint64_t{in1} * 4 + in2};
};

inline std::tuple<uint64_t> Sh2adduw(uint32_t in1, uint64_t in2) {
  return Sh2add(uint64_t{in1}, in2);
};

inline std::tuple<uint64_t> Sh3add(uint64_t in1, uint64_t in2) {
  return {uint64_t{in1} * 8 + in2};
};

inline std::tuple<uint64_t> Sh3adduw(uint32_t in1, uint64_t in2) {
  return Sh3add(uint64_t{in1}, in2);
};

template <typename T, enum PreferredIntrinsicsImplementation>
inline std::tuple<uint64_t> Zext(T in1) {
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint16_t> ||
                std::is_same_v<T, uint8_t>);
  return {static_cast<uint64_t>(in1)};
};

}  // namespace intrinsics

}  // namespace berberis

#endif  // RISCV64_TO_ARM64_BERBERIS_INTRINSICS_H_
