/*
 * Copyright (C) 2020 The Android Open Source Project
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

#ifndef RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_IMPL_H_
#define RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_IMPL_H_

#include "berberis/base/bit_util.h"
#include "berberis/intrinsics/macro_assembler.h"

namespace berberis {

namespace x86 {

// Constant suitable for NaNBoxing of RISC-V 32bit float with PXor.
extern const int32_t kNanBoxFloat32;

}  // namespace x86

}  // namespace berberis

#endif  // RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_IMPL_H_
