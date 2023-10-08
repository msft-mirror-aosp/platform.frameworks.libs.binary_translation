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

#ifndef BERBERIS_INTRINSICS_RISCV64_VECTOR_INTRINSICS_H_
#define BERBERIS_INTRINSICS_RISCV64_VECTOR_INTRINSICS_H_

#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/intrinsics_float.h"    // Float32/Float64
#include "berberis/intrinsics/riscv64/intrinsics.h"  // PreferredIntrinsicsImplementation
#include "berberis/intrinsics/simd_register.h"
#include "berberis/intrinsics/type_traits.h"

namespace berberis::intrinsics {

enum class TailProcessing {
  kUndisturbed = 0,
  kAgnostic = 1,
};

enum class InactiveProcessing {
  kUndisturbed = 0,
  kAgnostic = 1,
};

template <typename ElementType>
int MaskForRegisterInSequence(SIMD128Register mask, size_t register_in_sequence) {
  if constexpr (sizeof(ElementType) == sizeof(uint8_t)) {
    return mask.Get<uint16_t>(register_in_sequence);
  } else if constexpr (sizeof(ElementType) == sizeof(uint16_t)) {
    return mask.Get<uint8_t>(register_in_sequence);
  } else if constexpr (sizeof(ElementType) == sizeof(uint32_t)) {
    return mask.Get<uint32_t>(0) >> (register_in_sequence * 4);
  } else if constexpr (sizeof(ElementType) == sizeof(uint64_t)) {
    return mask.Get<uint32_t>(0) >> (register_in_sequence * 2);
  } else {
    static_assert(kDependentTypeFalse<ElementType>, "Unsupported vector element type");
  }
}

template <typename ElementType>
inline ElementType VectorElement(SIMD128Register src, int index) {
  return src.Get<ElementType>(index);
}

template <typename ElementType>
inline ElementType VectorElement(ElementType src, int) {
  return src;
}

// TODO(b/260725458): Pass lambda as template argument after C++20 would become available.
template <typename ElementType, TailProcessing vta, typename Lambda, typename... SourceType>
inline std::tuple<SIMD128Register> VectorArithmetic(Lambda lambda,
                                                    int vstart,
                                                    int vl,
                                                    SIMD128Register result,
                                                    SourceType... source) {
  static_assert(((std::is_same_v<SourceType, SIMD128Register> ||
                  std::is_same_v<SourceType, ElementType>)&&...));
  constexpr auto fill_value = [] {
    if constexpr (std::is_integral_v<ElementType>) {
      return std::numeric_limits<std::make_unsigned_t<ElementType>>::max();
    } else {
      return std::numeric_limits<
          std::make_unsigned_t<typename TypeTraits<ElementType>::Int>>::max();
    }
  }();
  if (vstart < 0) {
    vstart = 0;
  }
  if (vl > static_cast<int>(16 / sizeof(ElementType))) {
    vl = 16 / sizeof(ElementType);
  }
  if (vstart == 0 && vl == static_cast<int>(16 / sizeof(ElementType))) {
    for (int index = vstart; index < vl; ++index) {
      result.Set<ElementType>(lambda(VectorElement<ElementType>(source, index)...), index);
    }
  } else {
    #pragma clang loop unroll(disable)
    for (int index = vstart; index < vl; ++index) {
      result.Set<ElementType>(lambda(VectorElement<ElementType>(source, index)...), index);
    }
    if constexpr (vta == TailProcessing::kAgnostic) {
      if (vl < static_cast<int>(16 / sizeof(ElementType))) {
        #pragma clang loop unroll(disable)
        for (int index = vl; index < 16 / static_cast<int>(sizeof(ElementType)); ++index) {
          result.Set<ElementType>(fill_value, index);
        }
      }
    }
  }
  return result;
}

// TODO(b/260725458): Pass lambda as template argument after C++20 would become available.
template <typename ElementType,
          TailProcessing vta,
          InactiveProcessing vma,
          typename Lambda,
          typename... SourceType>
inline std::tuple<SIMD128Register> VectorArithmetic(Lambda lambda,
                                                    int vstart,
                                                    int vl,
                                                    int mask,
                                                    SIMD128Register result,
                                                    SourceType... source) {
  static_assert(((std::is_same_v<SourceType, SIMD128Register> ||
                  std::is_same_v<SourceType, ElementType>)&&...));
  constexpr auto fill_value = [] {
    if constexpr (std::is_integral_v<ElementType>) {
      return std::numeric_limits<std::make_unsigned_t<ElementType>>::max();
    } else {
      return std::numeric_limits<
          std::make_unsigned_t<typename TypeTraits<ElementType>::Int>>::max();
    }
  }();
  if (vstart < 0) {
    vstart = 0;
  }
  if (vl > static_cast<int>(16 / sizeof(ElementType))) {
    vl = 16 / sizeof(ElementType);
  }
  #pragma clang loop unroll(disable)
  for (int index = vstart; index < vl; ++index) {
    if (mask & (1 << index)) {
      result.Set<ElementType>(lambda(VectorElement<ElementType>(source, index)...), index);
    } else if constexpr (vma == InactiveProcessing::kAgnostic) {
      result.Set<ElementType>(fill_value, index);
    }
  }
  if constexpr (vta == TailProcessing::kAgnostic) {
    if (vl < static_cast<int>(16 / sizeof(ElementType))) {
      #pragma clang loop unroll(disable)
      for (int index = vl; index < 16 / static_cast<int>(sizeof(ElementType)); ++index) {
        result.Set<ElementType>(fill_value, index);
      }
    }
  }
  return result;
}

#define DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS(...) __VA_ARGS__
#define DEFINE_ARITHMETIC_INTRINSIC(Name, arithmetic, parameters, arguments)                      \
                                                                                                  \
  template <typename ElementType,                                                                 \
            TailProcessing vta,                                                                   \
            enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>       \
  inline std::tuple<SIMD128Register> Name(int vstart,                                             \
                                          int vl,                                                 \
                                          SIMD128Register result,                                 \
                                          DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS parameters) { \
    return VectorArithmetic<ElementType, vta>(                                                    \
        [](auto... args) {                                                                        \
          static_assert((std::is_same_v<decltype(args), ElementType> && ...));                    \
          arithmetic;                                                                             \
        },                                                                                        \
        vstart,                                                                                   \
        vl,                                                                                       \
        result,                                                                                   \
        DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS arguments);                                     \
  }                                                                                               \
                                                                                                  \
  template <typename ElementType,                                                                 \
            TailProcessing vta,                                                                   \
            InactiveProcessing vma,                                                               \
            enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>       \
  inline std::tuple<SIMD128Register> Name##m(                                                     \
      int vstart,                                                                                 \
      int vl,                                                                                     \
      int mask,                                                                                   \
      SIMD128Register result,                                                                     \
      DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS parameters) {                                     \
    return VectorArithmetic<ElementType, vta, vma>(                                               \
        [](auto... args) {                                                                        \
          static_assert((std::is_same_v<decltype(args), ElementType> && ...));                    \
          arithmetic;                                                                             \
        },                                                                                        \
        vstart,                                                                                   \
        vl,                                                                                       \
        mask,                                                                                     \
        result,                                                                                   \
        DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS arguments);                                     \
  }

DEFINE_ARITHMETIC_INTRINSIC(Vaddvv,
                            return (args + ...),
                            (SIMD128Register src1, SIMD128Register src2),
                            (src1, src2))
DEFINE_ARITHMETIC_INTRINSIC(Vaddvx,
                            return (args + ...),
                            (SIMD128Register src1, ElementType src2),
                            (src1, src2))
DEFINE_ARITHMETIC_INTRINSIC(Vsubvv,
                            return (args - ...),
                            (SIMD128Register src1, SIMD128Register src2),
                            (src1, src2))
DEFINE_ARITHMETIC_INTRINSIC(Vsubvx,
                            return (args - ...),
                            (SIMD128Register src1, ElementType src2),
                            (src1, src2))
DEFINE_ARITHMETIC_INTRINSIC(Vrsubvx,
                            ({ auto [arg1, arg2] = std::tuple{args...}; return (arg2 - arg1); }),
                            (SIMD128Register src1, ElementType src2),
                            (src1, src2))

#undef DEFINE_ARITHMETIC_INTRINSIC
#undef DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS

}  // namespace berberis::intrinsics

#endif  // BERBERIS_INTRINSICS_RISCV64_VECTOR_INTRINSICS_H_
