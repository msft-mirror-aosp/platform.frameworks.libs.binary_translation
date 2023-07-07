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

#include <inttypes.h>
#include <sys/mman.h>

#include "berberis/base/bit_util.h"
#include "berberis/base/mmap.h"
#include "berberis/base/struct_check.h"

#include "berberis/intrinsics/macro_assembler.h"

namespace berberis::constants_pool {

// All constants we refer in macroinstructions are collected in MacroAssemblerConstants.
// This allows us:
//   1. To save memory (they are not duplicated when reused).
//   2. Make it possible to access in text assembler without complex dance with hash-tables.
//   3. Allocate below-2GB-copy in x86_64 mode easily.
struct MacroAssemblerConstants {
  alignas(16) const uint32_t kNanBoxFloat32[4] = {0x00000000, 0xffffffff, 0x00000000, 0xffffffff};
  alignas(16) const uint32_t kNanBoxedNaNsFloat32[4] = {0x7fc00000,
                                                        0x0ffffffff,
                                                        0x7fc00000,
                                                        0x0ffffffff};
};

// Make sure Layout is the same in 32-bit mode and 64-bit mode.
CHECK_STRUCT_LAYOUT(MacroAssemblerConstants, 256, 128);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kNanBoxFloat32, 0, 128);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kNanBoxedNaNsFloat32, 128, 128);

// Note: because we have aligned fields and thus padding in that data structure
// value-initialization is both slower and larger than copy-initialization for
// that structure.
//
// Also assembler intrinsics for interpreter use kMacroAssemblerConstants
// directly (they couldn't use consts below because these addresses are only
// known during runtime)
extern const MacroAssemblerConstants kBerberisMacroAssemblerConstants
    __attribute__((visibility("hidden")));
const MacroAssemblerConstants kBerberisMacroAssemblerConstants;

namespace {

int32_t GetConstants() {
  static const MacroAssemblerConstants* Constants =
      new (mmap(nullptr,
                AlignUpPageSize(sizeof(MacroAssemblerConstants)),
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT,
                -1,
                0)) MacroAssemblerConstants(kBerberisMacroAssemblerConstants);
  // Note that we are returning only 32-bit address here, but it's guaranteed to
  // be enough since struct is in low memory because of MAP_32BIT flag.
  return bit_cast<intptr_t>(Constants);
}

}  // namespace

extern const int32_t kBerberisMacroAssemblerConstantsRelocated;
const int32_t kBerberisMacroAssemblerConstantsRelocated = GetConstants();
const int32_t kNanBoxFloat32 = GetConstants() + offsetof(MacroAssemblerConstants, kNanBoxFloat32);
const int32_t kNanBoxedNaNsFloat32 =
    GetConstants() + offsetof(MacroAssemblerConstants, kNanBoxedNaNsFloat32);

}  // namespace berberis::constant_pool
