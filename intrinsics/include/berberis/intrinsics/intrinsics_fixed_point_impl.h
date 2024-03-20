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
template <typename ElementType>
std::tuple<ElementType> Roundoff(uint8_t vxrm, ElementType v, ElementType premasked_d) {
  static_assert(std::is_integral_v<ElementType>, "Roundoff: ElementType must be integral");
  uint8_t d = premasked_d & ((1 << BitUtilLog2(sizeof(ElementType) * 8)) - 1);
  if (d == 0) return {v};
  int r;
  switch (vxrm) {
    case VXRMFlags::RNU:
      r = (v >> (d - 1)) & 1;
      break;
    case VXRMFlags::RNE:
      r = ((v >> (d - 1)) & 1) & (((v & ((1 << (d - 1)) - 1)) != 0) | ((v >> d) & 1));
      break;
    case VXRMFlags::RDN:
      r = 0;
      break;
    case VXRMFlags::ROD:
      r = !((v >> d) & 1) & ((v & ((1 << d) - 1)) != 0);
      break;
    default:
      LOG_ALWAYS_FATAL("Roundoff: Invalid rounding mode");
  }

  return {(v >> d) + r};
}

}  // namespace berberis::intrinsics

#endif  // BERBERIS_INTRINSICS_INTRINSICS_FIXED_POINT_IMPL_H_
