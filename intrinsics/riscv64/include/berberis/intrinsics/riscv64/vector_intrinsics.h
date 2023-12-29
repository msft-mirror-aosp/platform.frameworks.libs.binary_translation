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

#include <climits>  // CHAR_BIT
#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include "berberis/base/bit_util.h"
#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/intrinsics.h"        // PreferredIntrinsicsImplementation
#include "berberis/intrinsics/intrinsics_float.h"  // Float32/Float64
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
[[nodiscard]] int MaskForRegisterInSequence(SIMD128Register mask, size_t register_in_sequence) {
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

// Naïve implementation for tests.  Also used on not-x86 platforms.
[[nodiscard]] inline SIMD128Register MakeBitmaskFromVlForTests(size_t vl) {
  if (vl == 128) {
    return SIMD128Register(__int128(0));
  } else {
    return SIMD128Register((~__int128(0)) << vl);
  }
}

#ifndef __x86_64__
[[nodiscard]] inline SIMD128Register MakeBitmaskFromVl(size_t vl) {
  return MakeBitmaskFromVlForTests(vl);
}
#endif

// Naïve implementation for tests.  Also used on not-x86 platforms.
template <typename ElementType>
[[nodiscard]] inline SIMD128Register BitMaskToSimdMaskForTests(size_t mask) {
  constexpr ElementType kZeroValue = ElementType{0};
  constexpr ElementType kFillValue = ~ElementType{0};
  SIMD128Register result;
  for (size_t index = 0; index < 16 / sizeof(ElementType); ++index) {
    size_t bit = 1 << index;
    if (mask & bit) {
      result.Set(kFillValue, index);
    } else {
      result.Set(kZeroValue, index);
    }
  }
  return result;
}

#ifndef __x86_64__
template <typename ElementType>
[[nodiscard]] inline SIMD128Register BitMaskToSimdMask(size_t mask) {
  return BitMaskToSimdMaskForTests<ElementType>(mask);
}
#endif

template <auto kElement>
[[nodiscard]] inline SIMD128Register VectorMaskedElementToForTests(SIMD128Register simd_mask,
                                                                   SIMD128Register result) {
  using ElementType = decltype(kElement);
  constexpr int kElementsCount = static_cast<int>(16 / sizeof(ElementType));
  for (int index = 0; index < kElementsCount; ++index) {
    if (!simd_mask.Get<ElementType>(index)) {
      result.Set(kElement, index);
    }
  }
  return result;
}

#ifndef __x86_64__
template <typename ElementType>
[[nodiscard]] inline SIMD128Register VectorMaskedElementTo(SIMD128Register simd_mask,
                                                           SIMD128Register result) {
  return VectorMaskedElementToForTests(simd_mask, result);
}
#endif

template <typename ElementType>
[[nodiscard]] inline ElementType VectorElement(SIMD128Register src, int index) {
  return src.Get<ElementType>(index);
}

template <typename ElementType>
[[nodiscard]] inline ElementType VectorElement(ElementType src, int) {
  return src;
}

template <typename ElementType, TailProcessing vta>
SIMD128Register VectorMasking(int vstart, int vl, SIMD128Register dest, SIMD128Register result) {
  constexpr int kElementsCount = static_cast<int>(16 / sizeof(ElementType));
  if (vstart < 0) {
    vstart = 0;
  }
  if (vl < 0) {
    vl = 0;
  }
  if (vl > kElementsCount) {
    vl = kElementsCount;
  }
  if (vstart == 0) [[likely]] {
    if (vl == 16) [[likely]] {
      return result;
    }
    SIMD128Register tail_bitmask = MakeBitmaskFromVl(vl * sizeof(ElementType) * 8);
    if constexpr (vta == TailProcessing::kAgnostic) {
      dest = result | tail_bitmask;
    } else {
      dest = (dest & tail_bitmask) | (result & ~tail_bitmask);
    }
  } else if (vstart > vl) [[unlikely]] {
    if (vl == 16) [[likely]] {
      return dest;
    }
    if constexpr (vta == TailProcessing::kAgnostic) {
      SIMD128Register tail_bitmask = MakeBitmaskFromVl(vl * sizeof(ElementType) * 8);
      dest |= tail_bitmask;
    }
  } else {
    SIMD128Register start_bitmask = MakeBitmaskFromVl(vstart * sizeof(ElementType) * 8);
    SIMD128Register tail_bitmask = MakeBitmaskFromVl(vl * sizeof(ElementType) * 8);
    if constexpr (vta == TailProcessing::kAgnostic) {
      dest = (dest & ~start_bitmask) | (result & start_bitmask) | tail_bitmask;
    } else {
      dest = (dest & (~start_bitmask | tail_bitmask)) | (result & start_bitmask & ~tail_bitmask);
    }
  }
  return dest;
}

template <typename ElementType, TailProcessing vta, InactiveProcessing vma>
SIMD128Register VectorMasking(int vstart,
                              int vl,
                              int mask,
                              SIMD128Register dest,
                              SIMD128Register result) {
  constexpr int kElementsCount = static_cast<int>(16 / sizeof(ElementType));
  if (vstart < 0) {
    vstart = 0;
  }
  if (vl < 0) {
    vl = 0;
  }
  if (vl > kElementsCount) {
    vl = kElementsCount;
  }
  SIMD128Register simd_mask = BitMaskToSimdMask<ElementType>(mask);
  if (vstart == 0) [[likely]] {
    SIMD128Register tail_bitmask = MakeBitmaskFromVl(vl * sizeof(ElementType) * 8);
    if constexpr (vta == TailProcessing::kAgnostic) {
      if constexpr (vma == InactiveProcessing::kAgnostic) {
        dest = result | ~simd_mask | tail_bitmask;
      } else {
        dest = (dest & ~simd_mask) | (result & simd_mask) | tail_bitmask;
      }
    } else {
      if constexpr (vma == InactiveProcessing::kAgnostic) {
        dest = (dest & tail_bitmask) | ((result | ~simd_mask) & ~tail_bitmask);
      } else {
        dest = (dest & (~simd_mask | tail_bitmask)) | (result & simd_mask & ~tail_bitmask);
      }
    }
  } else if (vstart > vl) [[unlikely]] {
    if (vl == 16) [[likely]] {
      return dest;
    }
    if constexpr (vta == TailProcessing::kAgnostic) {
      SIMD128Register tail_bitmask = MakeBitmaskFromVl(vl * sizeof(ElementType) * 8);
      dest |= tail_bitmask;
    }
  } else {
    SIMD128Register start_bitmask = MakeBitmaskFromVl(vstart * sizeof(ElementType) * 8);
    SIMD128Register tail_bitmask = MakeBitmaskFromVl(vl * sizeof(ElementType) * 8);
    if constexpr (vta == TailProcessing::kAgnostic) {
      if constexpr (vma == InactiveProcessing::kAgnostic) {
        dest = (dest & ~start_bitmask) | ((result | ~simd_mask) & start_bitmask) | tail_bitmask;
      } else {
        dest = (dest & (~simd_mask | ~start_bitmask)) | (result & simd_mask & start_bitmask) |
               tail_bitmask;
      }
    } else {
      if constexpr (vma == InactiveProcessing::kAgnostic) {
        dest = (dest & (~start_bitmask | tail_bitmask)) |
               ((result | ~simd_mask) & start_bitmask & ~tail_bitmask);
      } else {
        dest = (dest & (~simd_mask | ~start_bitmask | tail_bitmask)) |
               (result & simd_mask & start_bitmask & ~tail_bitmask);
      }
    }
  }
  return dest;
}

// TODO(b/260725458): Pass lambda as template argument after C++20 would become available.
template <typename ElementType, TailProcessing vta, typename Lambda, typename... SourceType>
inline std::tuple<SIMD128Register> VectorProcessing(Lambda lambda,
                                                    int vstart,
                                                    int vl,
                                                    SIMD128Register dest,
                                                    SourceType... source) {
  static_assert(((std::is_same_v<SourceType, SIMD128Register> ||
                  std::is_same_v<SourceType, ElementType>)&&...));
  SIMD128Register result;
  constexpr int kElementsCount = static_cast<int>(16 / sizeof(ElementType));
  for (int index = 0; index < kElementsCount; ++index) {
    result.Set(lambda(VectorElement<ElementType>(source, index)...), index);
  }
  return {VectorMasking<ElementType, vta>(vstart, vl, dest, result)};
}

// TODO(b/260725458): Pass lambda as template argument after C++20 would become available.
template <typename ElementType,
          TailProcessing vta,
          InactiveProcessing vma,
          typename Lambda,
          typename... SourceType>
inline std::tuple<SIMD128Register> VectorProcessing(Lambda lambda,
                                                    int vstart,
                                                    int vl,
                                                    int mask,
                                                    SIMD128Register dest,
                                                    SourceType... source) {
  static_assert(((std::is_same_v<SourceType, SIMD128Register> ||
                  std::is_same_v<SourceType, ElementType>)&&...));
  SIMD128Register result;
  constexpr int kElementsCount = static_cast<int>(16 / sizeof(ElementType));
  for (int index = 0; index < kElementsCount; ++index) {
    result.Set(lambda(VectorElement<ElementType>(source, index)...), index);
  }
  return {VectorMasking<ElementType, vta, vma>(vstart, vl, mask, dest, result)};
}

#define DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS(...) __VA_ARGS__
#define DEFINE_ARITHMETIC_INTRINSIC(Name, arithmetic, parameters, arguments)                      \
                                                                                                  \
  template <typename ElementType,                                                                 \
            TailProcessing vta,                                                                   \
            enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>       \
  inline std::tuple<SIMD128Register> Name(int vstart,                                             \
                                          int vl,                                                 \
                                          SIMD128Register dest,                                   \
                                          DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS parameters) { \
    return VectorProcessing<ElementType, vta>(                                                    \
        [](auto... args) {                                                                        \
          static_assert((std::is_same_v<decltype(args), ElementType> && ...));                    \
          arithmetic;                                                                             \
        },                                                                                        \
        vstart,                                                                                   \
        vl,                                                                                       \
        dest,                                                                                     \
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
      SIMD128Register dest,                                                                       \
      DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS parameters) {                                     \
    return VectorProcessing<ElementType, vta, vma>(                                               \
        [](auto... args) {                                                                        \
          static_assert((std::is_same_v<decltype(args), ElementType> && ...));                    \
          arithmetic;                                                                             \
        },                                                                                        \
        vstart,                                                                                   \
        vl,                                                                                       \
        mask,                                                                                     \
        dest,                                                                                     \
        DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS arguments);                                     \
  }

#define DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(name, ...)                 \
  DEFINE_ARITHMETIC_INTRINSIC(V##name##vv, return ({ __VA_ARGS__; }); \
                              , (SIMD128Register src1, SIMD128Register src2), (src1, src2))
#define DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(name, ...)                 \
  DEFINE_ARITHMETIC_INTRINSIC(V##name##vx, return ({ __VA_ARGS__; }); \
                              , (SIMD128Register src1, ElementType src2), (src1, src2))

#define DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(name, ...)                 \
  DEFINE_ARITHMETIC_INTRINSIC(V##name##vv, return ({ __VA_ARGS__; }); \
                              , (SIMD128Register src1, SIMD128Register src2), (src1, src2, dest))
#define DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(name, ...)                 \
  DEFINE_ARITHMETIC_INTRINSIC(V##name##vx, return ({ __VA_ARGS__; }); \
                              , (SIMD128Register src1, ElementType src2), (src1, src2, dest))

DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(add, (args + ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(add, (args + ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(rsub, auto [arg1, arg2] = std::tuple{args...}; (arg2 - arg1))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(sub, (args - ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(sub, (args - ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(and, (args & ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(and, (args & ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(or, (args | ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(or, (args | ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(xor, (args ^ ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(xor, (args ^ ...))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(mseq,
                                   ElementType{
                                       static_cast<typename ElementType::BaseType>((args == ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(mseq,
                                   ElementType{
                                       static_cast<typename ElementType::BaseType>((args == ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(msne,
                                   ElementType{
                                       static_cast<typename ElementType::BaseType>((args != ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(msne,
                                   ElementType{
                                       static_cast<typename ElementType::BaseType>((args != ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(mslt,
                                   ElementType{
                                       static_cast<typename ElementType::BaseType>((args < ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(mslt,
                                   ElementType{
                                       static_cast<typename ElementType::BaseType>((args < ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(msle,
                                   ElementType{
                                       static_cast<typename ElementType::BaseType>((args <= ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(msle,
                                   ElementType{
                                       static_cast<typename ElementType::BaseType>((args <= ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(msgt,
                                   ElementType{
                                       static_cast<typename ElementType::BaseType>((args > ...))})
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(sl, auto [arg1, arg2] = std::tuple{args...}; (arg1 << arg2))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(sl, auto [arg1, arg2] = std::tuple{args...}; (arg1 << arg2))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(sr, auto [arg1, arg2] = std::tuple{args...}; (arg1 >> arg2))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(sr, auto [arg1, arg2] = std::tuple{args...}; (arg1 >> arg2))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(macc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   ((arg2 * arg1) + arg3))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(macc, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   ((arg2 * arg1) + arg3))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(nmsac, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (-(arg2 * arg1) + arg3))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(nmsac, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (-(arg2 * arg1) + arg3))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(madd, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   ((arg2 * arg3) + arg1))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(madd, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   ((arg2 * arg3) + arg1))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VV(nmsub, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (-(arg2 * arg3) + arg1))
DEFINE_3OP_ARITHMETIC_INTRINSIC_VX(nmsub, auto [arg1, arg2, arg3] = std::tuple{args...};
                                   (-(arg2 * arg3) + arg1))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(min, (std::min(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(min, (std::min(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VV(max, (std::max(args...)))
DEFINE_2OP_ARITHMETIC_INTRINSIC_VX(max, (std::max(args...)))

#undef DEFINE_ARITHMETIC_INTRINSIC
#undef DEFINE_ARITHMETIC_PARAMETERS_OR_ARGUMENTS
#undef DEFINE_2OP_ARITHMETIC_INTRINSIC_VV
#undef DEFINE_2OP_ARITHMETIC_INTRINSIC_VX

}  // namespace berberis::intrinsics

#endif  // BERBERIS_INTRINSICS_RISCV64_VECTOR_INTRINSICS_H_
