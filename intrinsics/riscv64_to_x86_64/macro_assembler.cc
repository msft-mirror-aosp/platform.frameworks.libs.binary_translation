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
  alignas(16) const uint64_t kNanBoxFloat32[2] = {0xffff'ffff'0000'0000, 0xffff'ffff'0000'0000};
  alignas(16) const uint64_t kNanBoxedNansFloat32[2] = {0xffff'ffff'7fc0'0000,
                                                        0xffff'ffff'7fc0'0000};
  alignas(16) const uint32_t kCanonicalNansFloat32[4] = {0x7fc'00000,
                                                         0x7fc'00000,
                                                         0x7fc'00000,
                                                         0x7fc'00000};
  alignas(16) const uint64_t kCanonicalNansFloat64[2] = {0x7ff8'0000'0000'0000,
                                                         0x7ff8'0000'0000'0000};
  int64_t kBsrToClzInt64 = 127;
  int64_t kWidthInBits64 = 64;
  int32_t kBsrToClzInt32 = 63;
  int32_t kWidthInBits32 = 32;
  // 64 bit constants for use with arithmetic operations.
  // Used because only 32 bit immediates are supported on x86-64.
  int64_t k0x8000_0000_0000_00ff = 0x8000'0000'0000'00ff;
  alignas(16) const int8_t kRiscVToX87Exceptions[32] = {
      0x00, 0x20, 0x10, 0x30, 0x08, 0x28, 0x18, 0x38,
      0x04, 0x24, 0x14, 0x34, 0x0c, 0x2c, 0x1c, 0x3c,
      0x01, 0x21, 0x11, 0x31, 0x09, 0x29, 0x19, 0x39,
      0x05, 0x25, 0x15, 0x35, 0x0d, 0x2d, 0x1d, 0x3d};
  alignas(16) const int8_t kX87ToRiscVExceptions[64] = {
      0x00, 0x10, 0x00, 0x10, 0x08, 0x18, 0x08, 0x18,
      0x04, 0x14, 0x04, 0x14, 0x0c, 0x1c, 0x0c, 0x1c,
      0x02, 0x12, 0x02, 0x12, 0x0a, 0x1a, 0x0a, 0x1a,
      0x06, 0x16, 0x06, 0x16, 0x0e, 0x1e, 0x0e, 0x1e,
      0x01, 0x11, 0x01, 0x11, 0x09, 0x19, 0x09, 0x19,
      0x05, 0x15, 0x05, 0x15, 0x0d, 0x1d, 0x0d, 0x1d,
      0x03, 0x13, 0x03, 0x13, 0x0b, 0x1b, 0x0b, 0x1b,
      0x07, 0x17, 0x07, 0x17, 0x0f, 0x1f, 0x0f, 0x1f};
  // This table represents exactly what you see: 𝟏𝟐𝟖 + 𝐍 unset bits and then 𝟏𝟐𝟖 - 𝐍 set bits for 𝐍
  // in range from 𝟎 to 𝟕.  The last 𝟏𝟐𝟖 bits from line 𝐍 then it's mask for 𝐯𝐥 equal to 𝐍 and if
  // you shift start address down by 𝐌  bytes then you get mask for 𝟖 * 𝐌 + 𝐍 bits.
  // Using this approach we may load appropriate bitmask from memory with one load instruction and
  // the table itself takes 𝟐𝟓𝟔 bytes.
  // Since valid values for 𝐯𝐥 are from 𝟎 to 𝐯𝐥 𝟎 𝟏𝟐𝟖 (including 𝟏𝟐𝟖), 𝐌 may be from 𝟎 to 𝟏𝟔,
  // that's why we need 16 zero bytes in that table.
  // On AMD CPUs with misalignsse feature we may access data from that table without using movups,
  // and on all CPUs we may use 2KiB table instead.
  // But for now we are using movups and small table as probably-adequate compromise.
  alignas(16) const uint64_t kBitMaskTable[8][4] = {
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'fffe, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'fffc, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'fff8, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'fff0, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffe0, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffc0, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000, 0xffff'ffff'ffff'ff80, 0xffff'ffff'ffff'ffff},
  };
};

// Make sure Layout is the same in 32-bit mode and 64-bit mode.
CHECK_STRUCT_LAYOUT(MacroAssemblerConstants, 3584, 128);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kNanBoxFloat32, 0, 128);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kNanBoxedNansFloat32, 128, 128);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kCanonicalNansFloat32, 256, 128);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kCanonicalNansFloat64, 384, 128);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kBsrToClzInt64, 512, 64);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kWidthInBits64, 576, 64);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kBsrToClzInt32, 640, 32);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kWidthInBits32, 672, 32);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, k0x8000_0000_0000_00ff, 704, 64);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kRiscVToX87Exceptions, 768, 256);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kX87ToRiscVExceptions, 1024, 512);
CHECK_FIELD_LAYOUT(MacroAssemblerConstants, kBitMaskTable, 1536, 2048);

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
template <>
const int32_t kVectorConst<uint64_t{0x0000'0000'0000'0000}> =
    GetConstants() + offsetof(MacroAssemblerConstants, kBitMaskTable);
template <>
const int32_t kVectorConst<uint64_t{0xffff'ffff'ffff'ffff}> =
    GetConstants() + offsetof(MacroAssemblerConstants, kBitMaskTable) + 16;
template <>
const int32_t kVectorConst<uint64_t{0xffff'ffff'0000'0000}> =
    GetConstants() + offsetof(MacroAssemblerConstants, kNanBoxFloat32);
template <>
const int32_t kVectorConst<uint64_t{0xffff'ffff'7fc0'0000}> =
    GetConstants() + offsetof(MacroAssemblerConstants, kNanBoxedNansFloat32);
template <>
const int32_t kVectorConst<uint64_t{0x7fc0'0000'7fc0'0000}> =
    GetConstants() + offsetof(MacroAssemblerConstants, kCanonicalNansFloat32);
template <>
const int32_t kVectorConst<uint64_t{0x7ff8'0000'0000'0000}> =
    GetConstants() + offsetof(MacroAssemblerConstants, kCanonicalNansFloat64);
template <>
const int32_t kConst<uint64_t{127}> =
    GetConstants() + offsetof(MacroAssemblerConstants, kBsrToClzInt64);
template <>
const int32_t kConst<uint64_t{64}> =
    GetConstants() + offsetof(MacroAssemblerConstants, kWidthInBits64);
template <>
const int32_t kConst<uint32_t{63}> =
    GetConstants() + offsetof(MacroAssemblerConstants, kBsrToClzInt32);
template <>
const int32_t kConst<uint32_t{32}> =
    GetConstants() + offsetof(MacroAssemblerConstants, kWidthInBits32);
template <>
const int32_t kConst<uint64_t{0x8000'0000'0000'00ff}> =
    GetConstants() + offsetof(MacroAssemblerConstants, k0x8000_0000_0000_00ff);
const int32_t kRiscVToX87Exceptions =
    GetConstants() + offsetof(MacroAssemblerConstants, kRiscVToX87Exceptions);
const int32_t kX87ToRiscVExceptions =
    GetConstants() + offsetof(MacroAssemblerConstants, kX87ToRiscVExceptions);
const int32_t kBitMaskTable = GetConstants() + offsetof(MacroAssemblerConstants, kBitMaskTable);

}  // namespace berberis::constant_pool
