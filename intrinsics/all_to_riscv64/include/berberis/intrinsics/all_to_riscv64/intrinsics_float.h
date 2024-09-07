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

#ifndef BERBERIS_INTRINSICS_ALL_TO_RISCV64_INTRINSICS_FLOAT_H_
#define BERBERIS_INTRINSICS_ALL_TO_RISCV64_INTRINSICS_FLOAT_H_

#include <cinttypes>
#include <cmath>

#include "berberis/base/bit_util.h"
#include "berberis/base/logging.h"
#include "berberis/intrinsics/common/intrinsics_float.h"  // Float32/Float64
#include "berberis/intrinsics/guest_rounding_modes.h"     // FE_HOSTROUND/FE_TIESAWAY

namespace berberis::intrinsics {

#define MAKE_BINARY_OPERATOR(guest_name, operator_name, assignment_name)                         \
                                                                                                 \
  inline Float32 operator operator_name(const Float32& v1, const Float32& v2) {                  \
    Float32 result;                                                                              \
    asm("f" #guest_name ".s %0, %1, %2" : "=f"(result.value_) : "f"(v1.value_), "f"(v2.value_)); \
    return result;                                                                               \
  }                                                                                              \
                                                                                                 \
  inline Float32& operator assignment_name(Float32 & v1, const Float32 & v2) {                   \
    asm("f" #guest_name ".s %0, %1, %2" : "=f"(v1.value_) : "f"(v1.value_), "f"(v2.value_));     \
    return v1;                                                                                   \
  }                                                                                              \
                                                                                                 \
  inline Float64 operator operator_name(const Float64& v1, const Float64& v2) {                  \
    Float64 result;                                                                              \
    asm("f" #guest_name ".d %0, %1, %2" : "=f"(result.value_) : "f"(v1.value_), "f"(v2.value_)); \
    return result;                                                                               \
  }                                                                                              \
                                                                                                 \
  inline Float64& operator assignment_name(Float64 & v1, const Float64 & v2) {                   \
    asm("f" #guest_name ".d %0, %1, %2" : "=f"(v1.value_) : "f"(v1.value_), "f"(v2.value_));     \
    return v1;                                                                                   \
  }

MAKE_BINARY_OPERATOR(add, +, +=)
MAKE_BINARY_OPERATOR(sub, -, -=)
MAKE_BINARY_OPERATOR(mul, *, *=)
MAKE_BINARY_OPERATOR(div, /, /=)

#undef MAKE_BINARY_OPERATOR

inline bool operator<(const Float32& v1, const Float32& v2) {
  bool result;
  asm("flt.s %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator<(const Float64& v1, const Float64& v2) {
  bool result;
  asm("flt.d %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator>(const Float32& v1, const Float32& v2) {
  bool result;
  asm("flt.s %0, %1, %2" : "=r"(result) : "f"(v2.value_), "f"(v1.value_));
  return result;
}

inline bool operator>(const Float64& v1, const Float64& v2) {
  bool result;
  asm("flt.d %0, %1, %2" : "=r"(result) : "f"(v2.value_), "f"(v1.value_));
  return result;
}

inline bool operator<=(const Float32& v1, const Float32& v2) {
  bool result;
  asm("fle.s %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator<=(const Float64& v1, const Float64& v2) {
  bool result;
  asm("fle.d %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator>=(const Float32& v1, const Float32& v2) {
  bool result;
  asm("fle.s %0, %1, %2" : "=r"(result) : "f"(v2.value_), "f"(v1.value_));
  return result;
}

inline bool operator>=(const Float64& v1, const Float64& v2) {
  bool result;
  asm("fle.d %0, %1, %2" : "=r"(result) : "f"(v2.value_), "f"(v1.value_));
  return result;
}

inline bool operator==(const Float32& v1, const Float32& v2) {
  bool result;
  asm("feq.s %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator==(const Float64& v1, const Float64& v2) {
  bool result;
  asm("feq.d %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return result;
}

inline bool operator!=(const Float32& v1, const Float32& v2) {
  bool result;
  asm("feq.s %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return !result;
}

inline bool operator!=(const Float64& v1, const Float64& v2) {
  bool result;
  asm("feq.d %0, %1, %2" : "=r"(result) : "f"(v1.value_), "f"(v2.value_));
  return !result;
}

// It's NOT safe to use ANY functions which return float or double.  That's because IA32 ABI uses
// x87 stack to pass arguments (and does that even with -mfpmath=sse) and NaN float and
// double values would be corrupted if pushed on it.

inline Float32 Negative(const Float32& v) {
  Float32 result;
  asm("fneg.s %0, %1" : "=f"(result.value_) : "f"(v.value_));
  return result;
}

inline Float64 Negative(const Float64& v) {
  Float64 result;
  asm("fneg.d %0, %1" : "=f"(result.value_) : "f"(v.value_));
  return result;
}

// Since the '_value' attribute of wrapped float types (Float32/64) has private visibility, the
// chosen workaround was to pass float/double to helper method. This works fine since this code will
// only be run on riscv64 hardware, thus avoiding the issues with IA32 ABI.

#define ROUND_FLOAT(rounding_type, method_name, float_type, float_suffix, int_suffix)      \
                                                                                           \
  inline float_type method_name(float_type value) {                                        \
    uint64_t tmp0;                                                                         \
    float_type tmp1, tmp2 = 1 / std::numeric_limits<float_type>::epsilon();                \
    asm("fabs." #float_suffix                                                              \
        "  %[tmp1], %[value]\n"                                                            \
        "flt." #float_suffix                                                               \
        "   %[tmp0], %[tmp1], %[tmp2]\n"                                                   \
        "beqz    %[tmp0], 0f\n"                                                            \
        "fcvt." #int_suffix "." #float_suffix "        %[tmp0], %[value], " #rounding_type \
        "\n"                                                                               \
        "fcvt." #float_suffix "." #int_suffix "        %[tmp2], %[tmp0], " #rounding_type  \
        "\n"                                                                               \
        "fsgnj." #float_suffix                                                             \
        " %[value], %[tmp2], %[value]\n"                                                   \
        "0:\n"                                                                             \
        : [tmp0] "=r"(tmp0), [value] "=f"(value), [tmp1] "=f"(tmp1), [tmp2] "=f"(tmp2)     \
        : "[tmp0]"(tmp0), "[value]"(value), "[tmp1]"(tmp1), "[tmp2]"(tmp2));               \
    return value;                                                                          \
  }

ROUND_FLOAT(rdn, FRoundDown, float, s, w)
ROUND_FLOAT(rup, FRoundUp, float, s, w)
ROUND_FLOAT(dyn, FRoundHost, float, s, w)
ROUND_FLOAT(rtz, FRoundZero, float, s, w)
ROUND_FLOAT(rne, FRoundNearest, float, s, w)

ROUND_FLOAT(rdn, FRoundDown, double, d, l)
ROUND_FLOAT(rup, FRoundUp, double, d, l)
ROUND_FLOAT(dyn, FRoundHost, double, d, l)
ROUND_FLOAT(rtz, FRoundZero, double, d, l)
ROUND_FLOAT(rne, FRoundNearest, double, d, l)

inline Float32 FPRound(const Float32& value, uint32_t round_control) {
  switch (round_control) {
    case FE_HOSTROUND:
      return Float32(FRoundHost(value.value_));
    case FE_TONEAREST:
      return Float32(FRoundNearest(value.value_));
    case FE_DOWNWARD:
      return Float32(FRoundDown(value.value_));
    case FE_UPWARD:
      return Float32(FRoundUp(value.value_));
    case FE_TOWARDZERO:
      return Float32(FRoundZero(value.value_));
    case FE_TIESAWAY:
      // TODO(b/146437763): Might fail if value doesn't have a floating part.
      if (value == FPRound(value, FE_DOWNWARD) + Float32(0.5)) {
        return value > Float32(0.0) ? FPRound(value, FE_UPWARD) : FPRound(value, FE_DOWNWARD);
      }

      // Any other case can be handled by to-nearest rounding.
      return FPRound(value, FE_TONEAREST);
    default:
      FATAL("Unknown round_control in FPRound!");
  }
}

inline Float64 FPRound(const Float64& value, uint32_t round_control) {
  switch (round_control) {
    case FE_HOSTROUND:
      return Float64(FRoundHost(value.value_));
    case FE_TONEAREST:
      return Float64(FRoundNearest(value.value_));
    case FE_DOWNWARD:
      return Float64(FRoundDown(value.value_));
    case FE_UPWARD:
      return Float64(FRoundUp(value.value_));
    case FE_TOWARDZERO:
      return Float64(FRoundZero(value.value_));
    case FE_TIESAWAY:
      // Since riscv64 does not support this rounding mode exactly, we must manually handle the
      // tie-aways (from (-)x.5)
      // TODO(b/364539415): Make Float32 and Float64 versions consistent
      if (value == FPRound(value, FE_DOWNWARD)) {
        // Value is already an integer and can be returned as-is. Checking this first avoids
        // dealing with numbers too large to be able to have a fractional part.
        return value;
      } else if (value == FPRound(value, FE_DOWNWARD) + Float64(0.5)) {
        // Fraction part is exactly 1/2, in which case we need to tie-away
        return value > Float64(0.0) ? FPRound(value, FE_UPWARD) : FPRound(value, FE_DOWNWARD);
      }

      // Any other case can be handled by to-nearest rounding.
      return FPRound(value, FE_TONEAREST);
    default:
      FATAL("Unknown round_control in FPRound!");
  }
}

#undef ROUND_FLOAT

}  // namespace berberis::intrinsics

#endif  // BERBERIS_INTRINSICS_ALL_TO_RISCV64_INTRINSICS_FLOAT_H_
