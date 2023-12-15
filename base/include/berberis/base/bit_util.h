/*
 * Copyright (C) 2015 The Android Open Source Project
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

#ifndef BERBERIS_BASE_BIT_UTIL_H_
#define BERBERIS_BASE_BIT_UTIL_H_

#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "berberis/base/checks.h"

namespace berberis {

template <typename T>
constexpr bool IsPowerOf2(T x) {
  static_assert(std::is_integral_v<T>, "IsPowerOf2: T must be integral");
  DCHECK(x != 0);
  return (x & (x - 1)) == 0;
}

template <typename T>
constexpr T AlignDown(T x, size_t align) {
  static_assert(std::is_integral_v<T>, "AlignDown: T must be integral");
  DCHECK(IsPowerOf2(align));
  return x & ~(align - 1);
}

template <typename T>
constexpr T AlignUp(T x, size_t align) {
  return AlignDown(x + align - 1, align);
}

template <typename T>
constexpr bool IsAligned(T x, size_t align) {
  return AlignDown(x, align) == x;
}

// Helper to align pointers.
template <typename T>
constexpr T* AlignDown(T* p, size_t align) {
  return reinterpret_cast<T*>(AlignDown(reinterpret_cast<uintptr_t>(p), align));
}

// Helper to align pointers.
template <typename T>
constexpr T* AlignUp(T* p, size_t align) {
  return reinterpret_cast<T*>(AlignUp(reinterpret_cast<uintptr_t>(p), align));
}

// Helper to align pointers.
template <typename T>
constexpr bool IsAligned(T* p, size_t align) {
  return IsAligned(reinterpret_cast<uintptr_t>(p), align);
}

template <typename T>
constexpr T BitUtilLog2(T x) {
  static_assert(std::is_integral_v<T>, "Log2: T must be integral");
  CHECK(IsPowerOf2(x));
  // TODO(b/260725458): Use std::countr_zero after C++20 becomes available
  return __builtin_ctz(x);
}

// Verify that argument value fits into a target.
template <typename ResultType, typename ArgumentType>
inline bool IsInRange(ArgumentType x) {
  // Note: conversion from wider integer type into narrow integer type is always
  // defined.  Conversion to unsigned produces well-defined result while conversion
  // to signed type produces implementation-defined result but in both cases value
  // is guaranteed to be unchanged if it can be represented in the destination type
  // and is *some* valid value if it's unrepesentable.
  //
  // Quote from the standard (including "note" in the standard):
  //   If the destination type is unsigned, the resulting value is the least unsigned
  // integer congruent to the source integer (modulo 2ⁿ where n is the number of bits
  // used to represent the unsigned type). [ Note: In a two’s complement representation,
  // this conversion is conceptual and there is no change in the bit pattern (if there
  // is no truncation). — end note ]
  //   If the destination type is signed, the value is unchanged if it can be represented
  // in the destination type; otherwise, the value is implementation-defined.

  return static_cast<ResultType>(x) == x;
}

// bit_cast<Dest, Source> is a well-defined equivalent of address-casting:
//   *reinterpret_cast<Dest*>(&source)
// See chromium base/macros.h for details.
template <class Dest, class Source>
inline Dest bit_cast(const Source& source) {
  static_assert(sizeof(Dest) == sizeof(Source),
                "bit_cast: source and destination must be of same size");
  static_assert(std::is_trivially_copyable_v<Dest>,
                "bit_cast: destination must be trivially copyable");
  static_assert(std::is_trivially_copyable_v<Source>,
                "bit_cast: source must be trivially copyable");
  Dest dest;
  memcpy(&dest, &source, sizeof(dest));
  return dest;
}

// Saturating and wrapping integers.
//   1. Never trigger UB, even in case of overflow.
//   2. Only support mixed types when both are of the same type (e.g. SatInt8 and SatInt16 or
//      Int8 and Int64 are allowed, but SatInt8 and Int8 are forbidden and Int32 and Uint32
//      require explicit casting, too).
//   3. Results are performed after type expansion.

template <typename Base>
class Wrapping;

template <typename Base>
class Saturating {
 public:
  using BaseType = Base;
  using SignedType = Saturating<std::make_signed_t<BaseType>>;
  using UnsignedType = Saturating<std::make_unsigned_t<BaseType>>;
  static constexpr bool kIsSigned = std::is_signed_v<BaseType>;

  static_assert(std::is_integral_v<BaseType>);

  template <typename IntType,
            typename = std::enable_if_t<std::is_integral_v<IntType> &&
                                        ((sizeof(BaseType) < sizeof(IntType) &&
                                          std::is_signed_v<IntType> == kIsSigned) ||
                                         sizeof(IntType) == sizeof(BaseType))>>
  [[nodiscard]] constexpr operator IntType() const {
    return static_cast<IntType>(value);
  }
  template <typename IntType,
            typename = std::enable_if_t<std::is_integral_v<IntType> &&
                                        ((sizeof(BaseType) < sizeof(IntType) &&
                                          std::is_signed_v<IntType> == kIsSigned) ||
                                         (sizeof(BaseType) == sizeof(IntType))) &&
                                        !std::is_same_v<IntType, BaseType>>>
  [[nodiscard]] constexpr operator Saturating<IntType>() const {
    return {static_cast<IntType>(value)};
  }
  template <typename IntType,
            typename = std::enable_if_t<std::is_integral_v<IntType> &&
                                        sizeof(BaseType) == sizeof(IntType)>>
  [[nodiscard]] constexpr operator Wrapping<IntType>() const {
    return {static_cast<IntType>(value)};
  }

  [[nodiscard]] friend constexpr bool operator==(Saturating lhs, Saturating rhs) {
    return lhs.value == rhs.value;
  }
  [[nodiscard]] friend constexpr bool operator!=(Saturating lhs, Saturating rhs) {
    return lhs.value != rhs.value;
  }
  [[nodiscard]] friend constexpr bool operator<(Saturating lhs, Saturating rhs) {
    return lhs.value < rhs.value;
  }
  [[nodiscard]] friend constexpr bool operator<=(Saturating lhs, Saturating rhs) {
    return lhs.value <= rhs.value;
  }
  [[nodiscard]] friend constexpr bool operator>(Saturating lhs, Saturating rhs) {
    return lhs.value > rhs.value;
  }
  [[nodiscard]] friend constexpr bool operator>=(Saturating lhs, Saturating rhs) {
    return lhs.value >= rhs.value;
  }
  [[nodiscard]] friend constexpr Saturating& operator+=(Saturating& lhs, Saturating rhs) {
    lhs = lhs + rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Saturating operator+(Saturating lhs, Saturating rhs) {
    BaseType result;
    bool overflow = __builtin_add_overflow(lhs.value, rhs.value, &result);
    if (overflow) {
      if constexpr (kIsSigned) {
        if (result < 0) {
          result = std::numeric_limits<BaseType>::max();
        } else {
          result = std::numeric_limits<BaseType>::min();
        }
      } else {
        result = std::numeric_limits<BaseType>::max();
      }
    }
    return {result};
  }
  [[nodiscard]] friend constexpr Saturating& operator-=(Saturating& lhs, Saturating rhs) {
    lhs = lhs - rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Saturating operator-(Saturating lhs) {
    if constexpr (kIsSigned) {
      if (lhs.value == std::numeric_limits<BaseType>::min()) {
        return {std::numeric_limits<BaseType>::max()};
      }
      return {-lhs.value};
    }
    return 0;
  }
  [[nodiscard]] friend constexpr Saturating operator-(Saturating lhs, Saturating rhs) {
    BaseType result;
    bool overflow = __builtin_sub_overflow(lhs.value, rhs.value, &result);
    if (overflow) {
      if constexpr (kIsSigned) {
        if (result < 0) {
          result = std::numeric_limits<BaseType>::max();
        } else {
          result = std::numeric_limits<BaseType>::min();
        }
      } else {
        result = 0;
      }
    }
    return {result};
  }
  [[nodiscard]] friend constexpr Saturating& operator*=(Saturating& lhs, Saturating rhs) {
    lhs = lhs * rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Saturating operator*(Saturating lhs, Saturating rhs) {
    BaseType result;
    bool overflow = __builtin_mul_overflow(lhs.value, rhs.value, &result);
    if (overflow) {
      if constexpr (kIsSigned) {
        if (lhs.value < 0 != rhs.value < 0) {
          result = std::numeric_limits<BaseType>::min();
        } else {
          result = std::numeric_limits<BaseType>::max();
        }
      } else {
        result = std::numeric_limits<BaseType>::max();
      }
    }
    return {result};
  }
  [[nodiscard]] friend constexpr Saturating& operator/=(Saturating& lhs, Saturating rhs) {
    lhs = lhs / rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Saturating operator/(Saturating lhs, Saturating rhs) {
    if constexpr (kIsSigned) {
      if (lhs.value == std::numeric_limits<BaseType>::min() && rhs.value == -1) {
        return {std::numeric_limits<BaseType>::max()};
      }
    }
    return {BaseType(lhs.value / rhs.value)};
  }
  [[nodiscard]] friend constexpr Saturating& operator%=(Saturating& lhs, Saturating rhs) {
    lhs = lhs % rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Saturating operator%(Saturating lhs, Saturating rhs) {
    if constexpr (kIsSigned) {
      if (lhs.value == std::numeric_limits<BaseType>::min() && rhs.value == -1) {
        return {1};
      }
    }
    return {BaseType(lhs.value % rhs.value)};
  }
  BaseType value = 0;
};

template <typename Base>
class Wrapping {
 public:
  using BaseType = Base;
  using SignedType = Wrapping<std::make_signed_t<BaseType>>;
  using UnsignedType = Wrapping<std::make_unsigned_t<BaseType>>;
  static constexpr bool kIsSigned = std::is_signed_v<BaseType>;

  static_assert(std::is_integral_v<BaseType>);

  template <typename IntType,
            typename = std::enable_if_t<std::is_integral_v<IntType> &&
                                        ((sizeof(BaseType) < sizeof(IntType) &&
                                          std::is_signed_v<IntType> == kIsSigned) ||
                                         sizeof(IntType) == sizeof(BaseType))>>
  [[nodiscard]] constexpr operator IntType() const {
    return static_cast<IntType>(value);
  }
  template <typename IntType,
            typename = std::enable_if_t<std::is_integral_v<IntType> &&
                                        sizeof(BaseType) == sizeof(IntType)>>
  [[nodiscard]] constexpr operator Saturating<IntType>() const {
    return {static_cast<IntType>(value)};
  }
  template <typename IntType,
            typename = std::enable_if_t<std::is_integral_v<IntType> &&
                                        ((sizeof(BaseType) < sizeof(IntType) &&
                                          std::is_signed_v<IntType> == kIsSigned) ||
                                         (sizeof(BaseType) == sizeof(IntType))) &&
                                        !std::is_same_v<IntType, BaseType>>>
  [[nodiscard]] constexpr operator Wrapping<IntType>() const {
    return {static_cast<IntType>(value)};
  }

  [[nodiscard]] friend constexpr bool operator==(Wrapping lhs, Wrapping rhs) {
    return lhs.value == rhs.value;
  }
  [[nodiscard]] friend constexpr bool operator!=(Wrapping lhs, Wrapping rhs) {
    return lhs.value != rhs.value;
  }
  [[nodiscard]] friend constexpr bool operator<(Wrapping lhs, Wrapping rhs) {
    return lhs.value < rhs.value;
  }
  [[nodiscard]] friend constexpr bool operator<=(Wrapping lhs, Wrapping rhs) {
    return lhs.value <= rhs.value;
  }
  [[nodiscard]] friend constexpr bool operator>(Wrapping lhs, Wrapping rhs) {
    return lhs.value > rhs.value;
  }
  [[nodiscard]] friend constexpr bool operator>=(Wrapping lhs, Wrapping rhs) {
    return lhs.value >= rhs.value;
  }
  // Note:
  //   1. We use __builtin_xxx_overflow instead of simple +, -, or * operators because
  //      __builtin_xxx_overflow produces well-defined result in case of overflow while
  //      +, -, * are triggering undefined behavior conditions.
  //   2. All operator xxx= are implemented in terms of opernator xxx
  [[nodiscard]] friend constexpr Wrapping& operator+=(Wrapping& lhs, Wrapping rhs) {
    lhs = lhs + rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Wrapping operator+(Wrapping lhs, Wrapping rhs) {
    BaseType result;
    __builtin_add_overflow(lhs.value, rhs.value, &result);
    return {result};
  }
  [[nodiscard]] friend constexpr Wrapping& operator-=(Wrapping& lhs, Wrapping rhs) {
    lhs = lhs - rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Wrapping operator-(Wrapping lhs) {
    BaseType result;
    __builtin_sub_overflow(BaseType{0}, lhs.value, &result);
    return {result};
  }
  [[nodiscard]] friend constexpr Wrapping operator-(Wrapping lhs, Wrapping rhs) {
    BaseType result;
    __builtin_sub_overflow(lhs.value, rhs.value, &result);
    return {result};
  }
  [[nodiscard]] friend constexpr Wrapping& operator*=(Wrapping& lhs, Wrapping rhs) {
    lhs = lhs * rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Wrapping operator*(Wrapping lhs, Wrapping rhs) {
    BaseType result;
    __builtin_mul_overflow(lhs.value, rhs.value, &result);
    return {result};
  }
  [[nodiscard]] friend constexpr Wrapping& operator/=(Wrapping& lhs, Wrapping rhs) {
    lhs = lhs / rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Wrapping operator/(Wrapping lhs, Wrapping rhs) {
    if constexpr (kIsSigned) {
      if (lhs.value == std::numeric_limits<BaseType>::min() && rhs.value == -1) {
        return {std::numeric_limits<BaseType>::min()};
      }
    }
    return {BaseType(lhs.value / rhs.value)};
  }
  [[nodiscard]] friend constexpr Wrapping& operator%=(Wrapping& lhs, Wrapping rhs) {
    lhs = lhs % rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Wrapping operator%(Wrapping lhs, Wrapping rhs) {
    if constexpr (kIsSigned) {
      if (lhs.value == std::numeric_limits<BaseType>::min() && rhs.value == -1) {
        return {0};
      }
    }
    return {BaseType(lhs.value % rhs.value)};
  }
  [[nodiscard]] friend constexpr Wrapping& operator<<=(Wrapping& lhs, Wrapping rhs) {
    lhs = lhs << rhs;
    return lhs;
  }
  template <typename IntType>
  [[nodiscard]] friend constexpr Wrapping operator<<(Wrapping lhs, Wrapping<IntType> rhs) {
    return {BaseType(lhs.value << (rhs.value & (sizeof(BaseType) * CHAR_BIT - 1)))};
  }
  [[nodiscard]] friend constexpr Wrapping& operator>>=(Wrapping& lhs, Wrapping rhs) {
    lhs = lhs >> rhs;
    return lhs;
  }
  template <typename IntType>
  [[nodiscard]] friend constexpr Wrapping operator>>(Wrapping lhs, Wrapping<IntType> rhs) {
    return {BaseType(lhs.value >> (rhs.value & (sizeof(BaseType) * CHAR_BIT - 1)))};
  }
  [[nodiscard]] friend constexpr Wrapping& operator&=(Wrapping& lhs, Wrapping rhs) {
    lhs = lhs & rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Wrapping operator&(Wrapping lhs, Wrapping rhs) {
    return {BaseType(lhs.value & rhs.value)};
  }
  [[nodiscard]] friend constexpr Wrapping& operator|=(Wrapping& lhs, Wrapping rhs) {
    lhs = lhs | rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Wrapping operator|(Wrapping lhs, Wrapping rhs) {
    return {BaseType(lhs.value | rhs.value)};
  }
  [[nodiscard]] friend constexpr Wrapping& operator^=(Wrapping& lhs, Wrapping rhs) {
    lhs = lhs ^ rhs;
    return lhs;
  }
  [[nodiscard]] friend constexpr Wrapping operator^(Wrapping lhs, Wrapping rhs) {
    return {BaseType(lhs.value ^ rhs.value)};
  }
  [[nodiscard]] friend constexpr Wrapping operator~(Wrapping lhs) { return {BaseType(~lhs.value)}; }
  BaseType value = 0;
};

using SatInt8 = Saturating<int8_t>;
using SatUInt8 = Saturating<uint8_t>;
using SatInt16 = Saturating<int16_t>;
using SatUInt16 = Saturating<uint16_t>;
using SatInt32 = Saturating<int32_t>;
using SatUInt32 = Saturating<uint32_t>;
using SatInt64 = Saturating<int64_t>;
using SatUInt64 = Saturating<uint64_t>;
#if defined(__x86_64__)
using SatInt128 = Saturating<__int128>;
using SatUInt128 = Saturating<unsigned __int128>;
#endif

using Int8 = Wrapping<int8_t>;
using UInt8 = Wrapping<uint8_t>;
using Int16 = Wrapping<int16_t>;
using UInt16 = Wrapping<uint16_t>;
using Int32 = Wrapping<int32_t>;
using UInt32 = Wrapping<uint32_t>;
using Int64 = Wrapping<int64_t>;
using UInt64 = Wrapping<uint64_t>;
#if defined(__x86_64__)
using Int128 = Wrapping<__int128>;
using UInt128 = Wrapping<unsigned __int128>;
#endif

template <typename ResultType, typename IntType>
auto MaybeTruncateTo(IntType src)
    -> std::enable_if_t<std::is_integral_v<IntType> &&
                            sizeof(typename ResultType::BaseType) <= sizeof(IntType),
                        ResultType> {
  return ResultType{static_cast<ResultType::BaseType>(src)};
}

template <typename ResultType, typename IntType>
auto TruncateTo(IntType src)
    -> std::enable_if_t<std::is_integral_v<IntType> &&
                            sizeof(typename ResultType::BaseType) < sizeof(IntType),
                        ResultType> {
  return ResultType{static_cast<ResultType::BaseType>(src)};
}

template <typename T>
struct TypeTraits;

template <typename BaseType>
[[nodiscard]] constexpr auto Widen(Saturating<BaseType> source)
    -> Saturating<typename TypeTraits<BaseType>::Wide> {
  return {source.value};
}

template <typename BaseType>
[[nodiscard]] constexpr auto Widen(Wrapping<BaseType> source)
    -> Wrapping<typename TypeTraits<BaseType>::Wide> {
  return {source.value};
}

template <typename BaseType>
[[nodiscard]] constexpr auto Narrow(Saturating<BaseType> source)
    -> Saturating<typename TypeTraits<BaseType>::Narrow> {
  if constexpr (Saturating<BaseType>::kIsSigned) {
    if (source.value < std::numeric_limits<typename TypeTraits<BaseType>::Narrow>::min()) {
      return {std::numeric_limits<typename TypeTraits<BaseType>::Narrow>::min()};
    }
  }
  if (source.value > std::numeric_limits<typename TypeTraits<BaseType>::Narrow>::max()) {
    return {std::numeric_limits<typename TypeTraits<BaseType>::Narrow>::max()};
  }
  return {static_cast<typename TypeTraits<BaseType>::Narrow>(source.value)};
}

template <typename BaseType>
[[nodiscard]] constexpr auto Narrow(Wrapping<BaseType> source)
    -> Wrapping<typename TypeTraits<BaseType>::Narrow> {
  return {static_cast<typename TypeTraits<BaseType>::Narrow>(source.value)};
}

}  // namespace berberis

#endif  // BERBERIS_BASE_BIT_UTIL_H_
