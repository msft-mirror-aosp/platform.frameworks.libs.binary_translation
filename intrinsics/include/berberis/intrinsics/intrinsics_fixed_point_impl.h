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

#ifndef BERBERIS_INTRINSICS_INTRINSICS_FIXED_POINT_IMPL_H_
#define BERBERIS_INTRINSICS_INTRINSICS_FIXED_POINT_IMPL_H_

#include <cstdint>
#include <tuple>

#include "berberis/base/bit_util.h"
#include "berberis/intrinsics/guest_cpu_flags.h"

namespace berberis::intrinsics {

// Averaging Add and Subtract use bit-based tricks to ensure that there would be no overflow.
//
// Technically it's possible to implement them very easily if we would use WideType and Roundoff
// to avoid overflow: Narrow(Roundoff(Widen(x) ± Widen(y), 1)).
//
// But calculations without expansion are marginally faster, and, more importantly, this C++
// implementation may be used as a template for efficient SIMD-based implementation.
//
// The scheme works roughly in the following way:
//   1. “(a + β)/2 = a/2 + β/2” (note: this only true with rationals, not integers!)
//   2. “(2×A + B)/2 = A + B/2” (here we just replace a with 2A)
//   3. “X + Y = 2×A + B” where “A = X & Y” and “B = X ^ Y”
//      This gives us enough information to deal with Averaging Add, but for Subtract we need:
//   4. “-Y = ~Y + 1” (that's how two's complement works)
//   5. “X ^ ~Y = ~(X ^ Y)”
//      This gives us enough for the basic handling of Averaging Add and Subtract.
//      We also need to handle rouding but this is simplified by the fact that only B/2 may
//      produce half-bit (“2A/2 = A” is integer).
//   Averaging Add:           Averaging Subtract:
//     A = X & Y                A = X & ~Y
//     B = X ^ Y                B = X ^ Y
//     C = B >> 1               C = B >> 1
//     D = A + C                D = A + ~C + 1 = A - C
//  This gives us basic RDN calculation for Averaging Add and RNU for Averaging Subtract.
//  All other versions need adjustment but only in the last bit (since we are using bitwise-not
//  toavoid overflow our “D” base for negation is “half-bit greater” than for addition.
//  Thus the rounding “changes direction”).
//  • For RNU: for Averaging Add “+ (B & 1)”
//  • For RNE: for Averaging Add “+ (D & B & 1)”, for Averaging Subtract “- (D & B & 1)”
//  • For RDN: for Averaging Subtract “- (B & 1)”
//  • For ROD in case of addition we may “jam” bit from B with bitwise or (it's even mentioned in
//    the manual), but for subtraction we need to use “- (~D & B & 1)” (as in manual)
//  All calculations never produce overflow, except for RNU or RNE rounding when the smallest
//  value is subtracted from the largest one (and we are producing correct answer with ignored
//  overflow bit as described in manual).
template <typename ElementType, enum PreferredIntrinsicsImplementation>
std::tuple<ElementType> Aadd(int8_t vxrm, ElementType unwrapped_x, ElementType unwrapped_y) {
  using WrappingType = Wrapping<ElementType>;
  WrappingType x{unwrapped_x};
  WrappingType y{unwrapped_y};
  WrappingType a = x & y;
  WrappingType b = x ^ y;
  WrappingType c = b >> WrappingType{1};
  WrappingType d = a + c;
  switch (vxrm) {
    case VXRMFlags::RNU:
      d += (b & WrappingType{1});
      break;
    case VXRMFlags::RNE:
      d += (d & b & WrappingType{1});
      break;
    case VXRMFlags::RDN:
      break;
    case VXRMFlags::ROD:
      d |= (b & WrappingType{1});
      break;
  }
  return static_cast<ElementType>(d);
}

template <typename ElementType, enum PreferredIntrinsicsImplementation>
std::tuple<ElementType> Asub(int8_t vxrm, ElementType unwrapped_x, ElementType unwrapped_y) {
  using WrappingType = Wrapping<ElementType>;
  WrappingType x{unwrapped_x};
  WrappingType y{unwrapped_y};
  WrappingType a = x & ~y;
  WrappingType b = x ^ y;
  WrappingType c = b >> WrappingType{1};
  WrappingType d = a - c;
  switch (vxrm) {
    case VXRMFlags::RNU:
      break;
    case VXRMFlags::RNE:
      d -= d & b & WrappingType{1};
      break;
    case VXRMFlags::RDN:
      d -= b & WrappingType{1};
      break;
    case VXRMFlags::ROD:
      d -= ~d & b & WrappingType{1};
      break;
  }
  return static_cast<ElementType>(d);
}

// Function that rounds off a fixed-point value.
//
// The rounding mode vxrm is an integer that specifies the rounding mode.
// The following values are supported:
//
//   RNU(0): Round to nearest up
//   RNE(1): Round to nearest even
//   RND(2): Truncate
//   ROD(3): Round to nearest odd
//
template <typename ElementType, enum PreferredIntrinsicsImplementation>
std::tuple<ElementType> Roundoff(int8_t vxrm, ElementType v, ElementType premasked_d) {
  static_assert(std::is_integral_v<ElementType>, "Roundoff: ElementType must be integral");
  uint8_t d = premasked_d & ((1 << BitUtilLog2(sizeof(ElementType) * 8)) - 1);
  ElementType result = v >> d;
  if (d == 0) [[unlikely]] {
    return {result};
  }
  switch (vxrm) {
    case VXRMFlags::RNU:
      result += (v >> (d - 1)) & 1;
      break;
    case VXRMFlags::RNE:
      result += ((v >> (d - 1)) & 1) & (((v & ((1 << (d - 1)) - 1)) != 0) | ((v >> d) & 1));
      break;
    case VXRMFlags::RDN:
      break;
    case VXRMFlags::ROD:
      result |= (v & ((1 << d) - 1)) != 0;
      break;
    default:
      LOG_ALWAYS_FATAL("Roundoff: Invalid rounding mode");
  }

  return {result};
}

}  // namespace berberis::intrinsics

#endif  // BERBERIS_INTRINSICS_INTRINSICS_FIXED_POINT_IMPL_H_
