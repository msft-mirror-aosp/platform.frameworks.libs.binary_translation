/*
 * Copyright (C) 2016 The Android Open Source Project
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

#ifndef BERBERIS_INTRINSICS_SIMD_REGISTER_H_
#define BERBERIS_INTRINSICS_SIMD_REGISTER_H_

#if defined(__i386__) || defined(__x86_64)
#include "xmmintrin.h"
#endif

#include <stdint.h>
#include <string.h>

#include "berberis/intrinsics/common/intrinsics_float.h"

namespace berberis {

class SIMD128Register;

/*
 * We want to use partial specialization for SIMD128Register::[GS]et, but it's
 * it's not allowed for class members.  Use helper functions instead.
 */
template <typename T>
T SIMD128RegisterGet(const SIMD128Register* reg, int index) = delete;
template <typename T>
T SIMD128RegisterSet(SIMD128Register* reg, T elem, int index) = delete;

class SIMD128Register {
 public:
  // TODO(b/260725458): use explicit(sizeof(T) == 16) instead of three constructors when C++20 would
  // be available.
  template <typename T, typename = std::enable_if_t<sizeof(T) < 16>>
  explicit SIMD128Register(T elem) {
    Set<T>(elem, 0);
  }
  template <typename T,
            typename = std::enable_if_t<sizeof(T) == 16 &&
                                        !std::is_same_v<std::decay_t<T>, SIMD128Register>>>
  SIMD128Register(T&& elem) {
    Set<T>(elem);
  }
  SIMD128Register() = default;
  SIMD128Register(const SIMD128Register&) = default;
  SIMD128Register(SIMD128Register&&) = default;
  SIMD128Register& operator=(const SIMD128Register&) = default;
  SIMD128Register& operator=(SIMD128Register&&) = default;

  template <typename T>
  auto Get(int index) const -> std::enable_if_t<sizeof(T) < 16, std::decay_t<T>> {
    return SIMD128RegisterGet<std::decay_t<T>>(this, index);
  }
  template <typename T>
  auto Set(T elem, int index) -> std::enable_if_t<sizeof(T) < 16, std::decay_t<T>> {
    return SIMD128RegisterSet<T>(this, elem, index);
  }
  template <typename T>
  auto Get() const -> std::enable_if_t<sizeof(T) == 16, std::decay_t<T>> {
    return SIMD128RegisterGet<std::decay_t<T>>(this, 0);
  }
  template <typename T>
  auto Get(int index) const -> std::enable_if_t<sizeof(T) == 16, std::decay_t<T>> {
    CHECK_EQ(index, 0);
    return SIMD128RegisterGet<std::decay_t<T>>(this, 0);
  }
  template <typename T>
  auto Set(T elem) -> std::enable_if_t<sizeof(T) == 16, std::decay_t<T>> {
    return SIMD128RegisterSet<std::decay_t<T>>(this, elem, 0);
  }
  template <typename T>
  auto Set(T elem, int index) -> std::enable_if_t<sizeof(T) == 16, std::decay_t<T>> {
    CHECK_EQ(index, 0);
    return SIMD128RegisterSet<std::decay_t<T>>(this, elem, 0);
  }
  template <typename T>
  friend bool operator==(T lhs, SIMD128Register rhs) {
    // Note comparison of two vectors return vector of the same type. In such a case we need to
    // merge many bools that we got.
    if constexpr (sizeof(decltype(lhs == rhs.template Get<T>())) == sizeof(SIMD128Register)) {
      // On CPUs with _mm_movemask_epi8 (native, like on x86, or emulated, like on Power)
      // _mm_movemask_epi8 return 0xffff if and only if all comparisons returned true.
      return _mm_movemask_epi8(lhs == rhs.template Get<T>()) == 0xffff;
    } else {
      return lhs == rhs.Get<T>();
    }
  }
  template <typename T>
  friend bool operator!=(T lhs, SIMD128Register rhs) {
    // Note comparison of two vectors return vector of the same type. In such a case we need to
    // merge many bools that we got.
    if constexpr (sizeof(decltype(lhs != rhs.template Get<T>())) == sizeof(SIMD128Register)) {
      // On CPUs with _mm_movemask_epi8 (native, like on x86, or emulated, like on Power)
      // _mm_movemask_epi8 return 0xffff if and only if all comparisons returned true.
      return _mm_movemask_epi8(lhs == rhs.template Get<T>()) != 0xffff;
    } else {
      return lhs != rhs.Get<T>();
    }
  }
  template <typename T>
  friend bool operator==(SIMD128Register lhs, T rhs) {
    // Note comparison of two vectors return vector of the same type. In such a case we need to
    // merge many bools that we got.
    if constexpr (sizeof(decltype(lhs.template Get<T>() == rhs)) == sizeof(SIMD128Register)) {
      // On CPUs with _mm_movemask_epi8 (native, like on x86, or emulated, like on Power)
      // _mm_movemask_epi8 return 0xffff if and only if all comparisons returned true.
      return _mm_movemask_epi8(lhs.template Get<T>() == rhs) == 0xffff;
    } else {
      return lhs.Get<T>() == rhs;
    }
  }
  template <typename T>
  friend bool operator!=(SIMD128Register lhs, T rhs) {
    // Note comparison of two vectors return vector of the same type. In such a case we need to
    // merge many bools that we got.
    if constexpr (sizeof(decltype(lhs.template Get<T>() == rhs)) == sizeof(SIMD128Register)) {
      // On CPUs with _mm_movemask_epi8 (native, like on x86, or emulated, like on Power)
      // _mm_movemask_epi8 return 0xffff if and only if all comparisons returned true.
      return _mm_movemask_epi8(lhs.template Get<T>() == rhs) != 0xffff;
    } else {
      return lhs.Get<T>() != rhs;
    }
  }

 private:
  union {
#ifdef __GNUC__
    // Note: we are violating strict aliasing rules in the code below (Get and Set function) thus we
    // need to mask these fields "may_alias". Unknown attributes could be silently ignored by the
    // compiler. We protect definitions with #ifdef __GNU__ to make sure may_alias is not ignored.
    [[gnu::vector_size(16), gnu::may_alias]] int8_t int8;
    [[gnu::vector_size(16), gnu::may_alias]] uint8_t uint8;
    [[gnu::vector_size(16), gnu::may_alias]] int16_t int16;
    [[gnu::vector_size(16), gnu::may_alias]] uint16_t uint16;
    [[gnu::vector_size(16), gnu::may_alias]] int32_t int32;
    [[gnu::vector_size(16), gnu::may_alias]] uint32_t uint32;
    [[gnu::vector_size(16), gnu::may_alias]] int64_t int64;
    [[gnu::vector_size(16), gnu::may_alias]] uint64_t uint64;
#if defined(__x86_64)
    [[gnu::vector_size(16), gnu::may_alias]] __int128_t int128;
    [[gnu::vector_size(16), gnu::may_alias]] __uint128_t uint128;
#endif
    // Note: we couldn't use Float32/Float64 here because [[gnu::vector]] only works with
    // raw integer or FP-types.
    [[gnu::vector_size(16), gnu::may_alias]] float float32;
    [[gnu::vector_size(16), gnu::may_alias]] double float64;
#else
#error Unsupported compiler.
#endif
  };
  template <typename T>
  friend T SIMD128RegisterGet(const SIMD128Register* reg, int index);
  template <typename T>
  friend T SIMD128RegisterSet(SIMD128Register* reg, T elem, int index);
};

static_assert(sizeof(SIMD128Register) == 16, "Unexpected size of SIMD128Register");

#if defined(__i386__)
static_assert(alignof(SIMD128Register) == 16, "Unexpected align of SIMD128Register");
#elif defined(__x86_64)
static_assert(alignof(SIMD128Register) == 16, "Unexpected align of SIMD128Register");
#else
#error Unsupported architecture
#endif

/*
 * Partial specializations of SIMD128Register getters/setters for most types
 *
 * GNU C makes it possible to use unions to quickly and efficiently
 * operate with subvalues of different types:
 *   http://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#Type-punning
 * Unfortunately it's not a valid ANSI C code thus we always do that via
 * Get<type>(index) and Set<type>(value, index) accessors.
 *
 * For other compilers one will need to use memcpy to guarantee safety.
 */
#ifdef __GNUC__
#define SIMD_128_STDINT_REGISTER_GETTER_SETTER(TYPE, MEMBER)                          \
  template <>                                                                         \
  inline TYPE SIMD128RegisterGet<TYPE>(const SIMD128Register* reg, int index) {       \
    CHECK_LT(unsigned(index), sizeof(*reg) / sizeof(TYPE));                           \
    return reg->MEMBER[index];                                                        \
  }                                                                                   \
  template <>                                                                         \
  inline TYPE SIMD128RegisterSet<TYPE>(SIMD128Register * reg, TYPE elem, int index) { \
    CHECK_LT(unsigned(index), sizeof(*reg) / sizeof(TYPE));                           \
    return reg->MEMBER[index] = elem;                                                 \
  }
#define SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(TYPE, MEMBER)                         \
  template <>                                                                         \
  inline TYPE SIMD128RegisterGet<TYPE>(const SIMD128Register* reg, int index) {       \
    CHECK_LT(unsigned(index), sizeof(*reg) / sizeof(TYPE));                           \
    return {reg->MEMBER[index]};                                                      \
  }                                                                                   \
  template <>                                                                         \
  inline TYPE SIMD128RegisterSet<TYPE>(SIMD128Register * reg, TYPE elem, int index) { \
    CHECK_LT(unsigned(index), sizeof(*reg) / sizeof(TYPE));                           \
    return {reg->MEMBER[index] = elem};                                               \
  }
#define SIMD_128_FLOAT_REGISTER_GETTER_SETTER(TYPE, MEMBER_TYPE, MEMBER)              \
  template <>                                                                         \
  inline TYPE SIMD128RegisterGet<TYPE>(const SIMD128Register* reg, int index) {       \
    CHECK_LT(unsigned(index), sizeof(*reg) / sizeof(TYPE));                           \
    static_assert(sizeof(TYPE) == sizeof(MEMBER_TYPE));                               \
    /* Don't use bit_cast because it's unsafe if -O0 is used. */                      \
    /* See intrinsics_float.h for explanation. */                                     \
    TYPE elem;                                                                        \
    MEMBER_TYPE melem;                                                                \
    melem = reg->MEMBER[index];                                                       \
    memcpy(&elem, &melem, sizeof(TYPE));                                              \
    return elem;                                                                      \
  }                                                                                   \
  template <>                                                                         \
  inline TYPE SIMD128RegisterSet<TYPE>(SIMD128Register * reg, TYPE elem, int index) { \
    CHECK_LT(unsigned(index), sizeof(*reg) / sizeof(TYPE));                           \
    static_assert(sizeof(TYPE) == sizeof(MEMBER_TYPE));                               \
    /* Don't use bit_cast because it's unsafe if -O0 is used. */                      \
    /* See intrinsics_float.h for explanation. */                                     \
    MEMBER_TYPE melem;                                                                \
    memcpy(&melem, &elem, sizeof(TYPE));                                              \
    reg->MEMBER[index] = melem;                                                       \
    return elem;                                                                      \
  }
#define SIMD_128_FULL_REGISTER_GETTER_SETTER(TYPE, MEMBER)                            \
  template <>                                                                         \
  inline TYPE SIMD128RegisterGet<TYPE>(const SIMD128Register* reg, int index) {       \
    CHECK_EQ(index, 0);                                                               \
    return reg->MEMBER;                                                               \
  }                                                                                   \
  template <>                                                                         \
  inline TYPE SIMD128RegisterSet<TYPE>(SIMD128Register * reg, TYPE elem, int index) { \
    CHECK_EQ(index, 0);                                                               \
    return reg->MEMBER = elem;                                                        \
  }
#endif
SIMD_128_STDINT_REGISTER_GETTER_SETTER(int8_t, int8);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(Int8, int8);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(SatInt8, int8);
SIMD_128_STDINT_REGISTER_GETTER_SETTER(uint8_t, uint8);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(UInt8, uint8);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(SatUInt8, uint8);
SIMD_128_STDINT_REGISTER_GETTER_SETTER(int16_t, int16);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(Int16, int16);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(SatInt16, int16);
SIMD_128_STDINT_REGISTER_GETTER_SETTER(uint16_t, uint16);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(UInt16, uint16);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(SatUInt16, uint16);
SIMD_128_STDINT_REGISTER_GETTER_SETTER(int32_t, int32);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(Int32, int32);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(SatInt32, int32);
SIMD_128_STDINT_REGISTER_GETTER_SETTER(uint32_t, uint32);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(UInt32, uint32);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(SatUInt32, uint32);
SIMD_128_STDINT_REGISTER_GETTER_SETTER(int64_t, int64);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(Int64, int64);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(SatInt64, int64);
SIMD_128_STDINT_REGISTER_GETTER_SETTER(uint64_t, uint64);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(UInt64, uint64);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(SatUInt64, uint64);
#if defined(__x86_64__)
SIMD_128_STDINT_REGISTER_GETTER_SETTER(__int128_t, int128);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(Int128, int128);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(SatInt128, int128);
SIMD_128_STDINT_REGISTER_GETTER_SETTER(__uint128_t, uint128);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(UInt128, uint128);
SIMD_128_SAFEINT_REGISTER_GETTER_SETTER(SatUInt128, uint128);
#endif
#if defined(__i386__) || defined(__x86_64__)
SIMD_128_FULL_REGISTER_GETTER_SETTER(__v16qi, int8);
SIMD_128_FULL_REGISTER_GETTER_SETTER(__v16qu, uint8);
SIMD_128_FULL_REGISTER_GETTER_SETTER(__v8hi, int16);
SIMD_128_FULL_REGISTER_GETTER_SETTER(__v8hu, uint16);
SIMD_128_FULL_REGISTER_GETTER_SETTER(__v4si, int32);
SIMD_128_FULL_REGISTER_GETTER_SETTER(__v4su, uint32);
SIMD_128_FULL_REGISTER_GETTER_SETTER(__v2du, uint64);
SIMD_128_FULL_REGISTER_GETTER_SETTER(__v2df, float64);
SIMD_128_FULL_REGISTER_GETTER_SETTER(__m128i, int64);
SIMD_128_FULL_REGISTER_GETTER_SETTER(__m128, float32);
#endif
SIMD_128_FLOAT_REGISTER_GETTER_SETTER(intrinsics::Float32, float, float32);
SIMD_128_FLOAT_REGISTER_GETTER_SETTER(intrinsics::Float64, double, float64);
#undef SIMD_128_FULL_REGISTER_GETTER_SETTER
#undef SIMD_128_fLOAT_REGISTER_GETTER_SETTER
#undef SIMD_128_SAFEINT_REGISTER_GETTER_SETTER
#undef SIMD_128_STDINT_REGISTER_GETTER_SETTER

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_SIMD_REGISTER_H_
