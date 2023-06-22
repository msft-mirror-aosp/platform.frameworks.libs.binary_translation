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

#include <stdint.h>
#include <string.h>

#include "berberis/intrinsics/intrinsics_float.h"

namespace berberis {

class SIMD128Register;

/*
 * We want to use partial specialization for SIMD128Register::[GS]et, but it's
 * it's not allowed for class members.  Use helper functions instead.
 */
template <typename T>
T SIMD128RegisterGet(const SIMD128Register* reg, int index);
template <typename T>
T SIMD128RegisterSet(SIMD128Register* reg, T elem, int index);

class SIMD128Register {
 public:
  template <typename T>
  explicit SIMD128Register(T elem) {
    Set<T>(elem, 0);
  }
  SIMD128Register() = default;
  SIMD128Register(const SIMD128Register&) = default;
  SIMD128Register(SIMD128Register&&) = default;
  SIMD128Register& operator=(const SIMD128Register&) = default;
  SIMD128Register& operator=(SIMD128Register&&) = default;

  template <typename T>
  T Get(int index) const {
    return SIMD128RegisterGet<T>(this, index);
  }
  template <typename T>
  T Set(T elem, int index) {
    return SIMD128RegisterSet<T>(this, elem, index);
  }
  template <typename T>
  auto Get() const -> std::enable_if_t<sizeof(T) == 16, T> {
    return SIMD128RegisterGet<T>(this, 0);
  }
  template <typename T>
  auto Set(T elem) -> std::enable_if_t<sizeof(T) == 16, T> {
    return SIMD128RegisterSet<T>(this, elem, 0);
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
#define SIMD_128_REGISTER_GETTER_SETTER(TYPE, MEMBER)                                 \
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
#define SIMD_128_REGISTER_GETTER_SETTЕR(TYPE, MEMBER_TYPE, MEMBER)                    \
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
#endif
SIMD_128_REGISTER_GETTER_SETTER(int8_t, int8);
SIMD_128_REGISTER_GETTER_SETTER(uint8_t, uint8);
SIMD_128_REGISTER_GETTER_SETTER(int16_t, int16);
SIMD_128_REGISTER_GETTER_SETTER(uint16_t, uint16);
SIMD_128_REGISTER_GETTER_SETTER(int32_t, int32);
SIMD_128_REGISTER_GETTER_SETTER(uint32_t, uint32);
SIMD_128_REGISTER_GETTER_SETTER(int64_t, int64);
SIMD_128_REGISTER_GETTER_SETTER(uint64_t, uint64);
#if defined(__x86_64)
SIMD_128_REGISTER_GETTER_SETTER(__uint128_t, uint128);
#endif
SIMD_128_REGISTER_GETTER_SETTЕR(intrinsics::Float32, float, float32);
SIMD_128_REGISTER_GETTER_SETTЕR(intrinsics::Float64, double, float64);
#undef SIMD_128_REGISTER_GETTER_SETTER

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_SIMD_REGISTER_H_
