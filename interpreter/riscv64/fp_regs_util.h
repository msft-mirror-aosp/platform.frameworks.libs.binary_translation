/*
 * Copyright (C) 2023 The Android Open Source Project
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

#ifndef BERBERIS_FP_REGS_TEST_H_
#define BERBERIS_FP_REGS_TEST_H_

#include <cstdint>

#include "berberis/base/bit_util.h"

namespace berberis {

inline constexpr class FPValueToFPReg {
 public:
  uint64_t operator()(uint64_t value) const { return value; }
  uint64_t operator()(float value) const {
    return bit_cast<uint32_t>(value) | 0xffff'ffff'0000'0000;
  }
  uint64_t operator()(double value) const { return bit_cast<uint64_t>(value); }
} kFPValueToFPReg;

}  // namespace berberis

#endif  // BERBERIS_FP_REGS_TEST_H_
