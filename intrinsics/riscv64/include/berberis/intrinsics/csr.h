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

#ifndef BERBERIS_INTRINSICS_GUEST_RISCV64_CSR_H_
#define BERBERIS_INTRINSICS_GUEST_RISCV64_CSR_H_

namespace berberis {

enum class CsrName {
  kFFlags = 0b00'00'0000'0001,
  kFrm = 0b00'00'0000'0010,
  kFCsr = 0b00'00'0000'0011,
  kMaxValue = 0b11'11'1111'1111,
};

}  // namespace berberis

#endif  // BERBERIS_INTRINSICS_GUEST_RISCV64_CSR_H_
