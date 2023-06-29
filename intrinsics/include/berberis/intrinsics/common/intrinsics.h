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

#ifndef BERBERIS_INTRINSICS_COMMON_INTRINSICS_H_
#define BERBERIS_INTRINSICS_COMMON_INTRINSICS_H_

#include <limits>
#include <tuple>
#include <type_traits>

#include "berberis/base/bit_util.h"
#include "berberis/intrinsics/intrinsics_float.h"  // Float32/Float64/ProcessNans
#include "berberis/intrinsics/type_traits.h"

namespace berberis {

class SIMD128Register;

namespace intrinsics {

// A solution for the inability to call generic implementation from specialization.
// Declaration:
//   template <typename Type,
//             int size,
//             enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
//   inline std::tuple<SIMD128Register> VectorMultiplyByScalarInt(SIMD128Register op1,
//                                                                SIMD128Register op2);
// Normal use only specifies two arguments, e.g. VectorMultiplyByScalarInt<uint32_t, 2>,
// but assembler implementation can (if SSE 4.1 is not available) do the following call:
//   return VectorMultiplyByScalarInt<uint32_t, 2, kUseCppImplementation>(in0, in1);
//
// Because PreferredIntrinsicsImplementation argument has non-default value we have call to the
// generic C-based implementation here.

enum PreferredIntrinsicsImplementation {
  kUseAssemblerImplementationIfPossible,
  kUseCppImplementation
};

#include "berberis/intrinsics/intrinsics-inl.h"  // NOLINT: generated file!

template <typename FloatType, enum PreferredIntrinsicsImplementation>
std::tuple<uint64_t> FClass(FloatType arg) {
  using IntType = std::make_unsigned_t<typename TypeTraits<FloatType>::Int>;
  constexpr IntType quiet_bit =
      __builtin_bit_cast(IntType, std::numeric_limits<FloatType>::quiet_NaN()) &
      ~__builtin_bit_cast(IntType, std::numeric_limits<FloatType>::signaling_NaN());
  const IntType raw_bits = bit_cast<IntType>(arg);

  switch (FPClassify(arg)) {
    case intrinsics::FPInfo::kNaN:
      return (raw_bits & quiet_bit) ? 0b10'0000'0000 : 0b01'0000'0000;
    case intrinsics::FPInfo::kInfinite:
      return intrinsics::SignBit(arg) ? 0b00'0000'0001 : 0b00'1000'0000;
    case intrinsics::FPInfo::kNormal:
      return intrinsics::SignBit(arg) ? 0b00'0000'0010 : 0b00'0100'0000;
    case intrinsics::FPInfo::kSubnormal:
      return intrinsics::SignBit(arg) ? 0b00'0000'0100 : 0b00'0010'0000;
    case intrinsics::FPInfo::kZero:
      return intrinsics::SignBit(arg) ? 0b00'0000'1000 : 0b00'0001'0000;
  }
}

template <typename TargetOperandType,
          typename SourceOperandType,
          enum PreferredIntrinsicsImplementation>
std::tuple<TargetOperandType> FCvtFloatToFloat(uint8_t rm, uint8_t frm, SourceOperandType arg) {
  static_assert(std::is_same_v<Float32, SourceOperandType> ||
                std::is_same_v<Float64, SourceOperandType>);
  static_assert(std::is_same_v<Float32, TargetOperandType> ||
                std::is_same_v<Float64, TargetOperandType>);
  if constexpr (sizeof(TargetOperandType) > sizeof(SourceOperandType)) {
    // Conversion from narrow type to wide one ignores rm because all possible values from narrow
    // type fit in the wide type.
    return TargetOperandType(arg);
  } else {
    return intrinsics::ExecuteFloatOperation<TargetOperandType>(
        rm, frm, [](auto x) { return typename TypeTraits<decltype(x)>::Narrow(x); }, arg);
  }
}

template <typename TargetOperandType,
          typename SourceOperandType,
          enum PreferredIntrinsicsImplementation>
std::tuple<TargetOperandType> FCvtFloatToInteger(uint8_t /*rm*/,
                                                 uint8_t /*frm*/,
                                                 SourceOperandType arg) {
  static_assert(std::is_same_v<Float32, SourceOperandType> ||
                std::is_same_v<Float64, SourceOperandType>);
  static_assert(std::is_integral_v<TargetOperandType>);
  // TODO(265372622): handle rm properly in integer-to-float and float-to-integer cases.
  TargetOperandType result = static_cast<TargetOperandType>(arg);
  return static_cast<std::make_signed_t<TargetOperandType>>(result);
}

template <typename TargetOperandType,
          typename SourceOperandType,
          enum PreferredIntrinsicsImplementation>
std::tuple<TargetOperandType> FCvtIntegerToFloat(uint8_t /*rm*/,
                                                 uint8_t /*frm*/,
                                                 SourceOperandType arg) {
  static_assert(std::is_integral_v<SourceOperandType>);
  static_assert(std::is_same_v<Float32, TargetOperandType> ||
                std::is_same_v<Float64, TargetOperandType>);
  // TODO(265372622): handle rm properly in integer-to-float and float-to-integer cases.
  TargetOperandType result = static_cast<TargetOperandType>(arg);
  return result;
}

template <typename FloatType,
          enum PreferredIntrinsicsImplementation kPreferredIntrinsicsImplementation>
std::tuple<FloatType> FMAdd(uint8_t rm,
                            uint8_t frm,
                            FloatType arg1,
                            FloatType arg2,
                            FloatType arg3) {
  return intrinsics::ExecuteFloatOperation<FloatType>(
      rm,
      frm,
      [](auto x, auto y, auto z) {
        return std::get<0>(FMAddDyn<decltype(x), kPreferredIntrinsicsImplementation>(x, y, z));
      },
      arg1,
      arg2,
      arg3);
}

template <typename FloatType, enum PreferredIntrinsicsImplementation>
std::tuple<FloatType> FMAddDyn(FloatType arg1, FloatType arg2, FloatType arg3) {
  return {intrinsics::MulAdd(arg1, arg2, arg3)};
}

template <typename FloatType, enum PreferredIntrinsicsImplementation>
std::tuple<FloatType> FMax(FloatType x, FloatType y) {
  return {Max(x, y)};
}

template <typename FloatType, enum PreferredIntrinsicsImplementation>
std::tuple<FloatType> FMin(FloatType x, FloatType y) {
  return {Min(x, y)};
}

template <typename FloatType,
          enum PreferredIntrinsicsImplementation kPreferredIntrinsicsImplementation>
std::tuple<FloatType> FMSub(uint8_t rm,
                            uint8_t frm,
                            FloatType arg1,
                            FloatType arg2,
                            FloatType arg3) {
  return intrinsics::ExecuteFloatOperation<FloatType>(
      rm,
      frm,
      [](auto x, auto y, auto z) {
        return std::get<0>(FMSubDyn<decltype(x), kPreferredIntrinsicsImplementation>(x, y, z));
      },
      arg1,
      arg2,
      arg3);
}

template <typename FloatType, enum PreferredIntrinsicsImplementation>
std::tuple<FloatType> FMSubDyn(FloatType arg1, FloatType arg2, FloatType arg3) {
  return {intrinsics::MulAdd(arg1, arg2, intrinsics::Negative(arg3))};
}

template <typename FloatType,
          enum PreferredIntrinsicsImplementation kPreferredIntrinsicsImplementation>
std::tuple<FloatType> FNMAdd(uint8_t rm,
                             uint8_t frm,
                             FloatType arg1,
                             FloatType arg2,
                             FloatType arg3) {
  return intrinsics::ExecuteFloatOperation<FloatType>(
      rm,
      frm,
      [](auto x, auto y, auto z) {
        return std::get<0>(FNMAddDyn<decltype(x), kPreferredIntrinsicsImplementation>(x, y, z));
      },
      arg1,
      arg2,
      arg3);
}

template <typename FloatType, enum PreferredIntrinsicsImplementation>
std::tuple<FloatType> FNMAddDyn(FloatType arg1, FloatType arg2, FloatType arg3) {
  return {intrinsics::MulAdd(intrinsics::Negative(arg1), arg2, arg3)};
}

template <typename FloatType,
          enum PreferredIntrinsicsImplementation kPreferredIntrinsicsImplementation>
std::tuple<FloatType> FNMSub(uint8_t rm,
                             uint8_t frm,
                             FloatType arg1,
                             FloatType arg2,
                             FloatType arg3) {
  return intrinsics::ExecuteFloatOperation<FloatType>(
      rm,
      frm,
      [](auto x, auto y, auto z) {
        return std::get<0>(FNMSubDyn<decltype(x), kPreferredIntrinsicsImplementation>(x, y, z));
      },
      arg1,
      arg2,
      arg3);
}

template <typename FloatType, enum PreferredIntrinsicsImplementation>
std::tuple<FloatType> FNMSubDyn(FloatType arg1, FloatType arg2, FloatType arg3) {
  return {intrinsics::MulAdd(intrinsics::Negative(arg1), arg2, intrinsics::Negative(arg3))};
}

template <typename FloatType, enum PreferredIntrinsicsImplementation>
std::tuple<FloatType> FSgnj(FloatType x, FloatType y) {
  using Int = typename TypeTraits<FloatType>::Int;
  using UInt = std::make_unsigned_t<Int>;
  constexpr UInt sign_bit = std::numeric_limits<Int>::min();
  constexpr UInt non_sign_bit = std::numeric_limits<Int>::max();
  return {bit_cast<FloatType>((bit_cast<UInt>(x) & non_sign_bit) | (bit_cast<UInt>(y) & sign_bit))};
}

template <typename FloatType, enum PreferredIntrinsicsImplementation>
std::tuple<FloatType> FSgnjn(FloatType x, FloatType y) {
  return FSgnj(x, Negative(y));
}

template <typename FloatType, enum PreferredIntrinsicsImplementation>
std::tuple<FloatType> FSgnjx(FloatType x, FloatType y) {
  using Int = typename TypeTraits<FloatType>::Int;
  using UInt = std::make_unsigned_t<Int>;
  constexpr UInt sign_bit = std::numeric_limits<Int>::min();
  return {bit_cast<FloatType>(bit_cast<UInt>(x) ^ (bit_cast<UInt>(y) & sign_bit))};
}

}  // namespace intrinsics

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_COMMON_INTRINSICS_H_
