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

#include "gtest/gtest.h"

#include <unistd.h>

#include <cstdint>

#include "berberis/base/bit_util.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/riscv64/interpreter.h"
#include "berberis/intrinsics/guest_fp_flags.h"        // GuestModeFromHostRounding
#include "berberis/intrinsics/guest_rounding_modes.h"  // ScopedRoundingMode
#include "berberis/intrinsics/simd_register.h"
#include "berberis/intrinsics/vector_intrinsics.h"
#include "berberis/runtime_primitives/memory_region_reservation.h"

namespace berberis {

namespace {

//  Interpreter decodes the size itself, but we need to accept this template parameter to share
//  tests with translators.
template <uint8_t kInsnSize = 4>
bool RunOneInstruction(ThreadState* state, GuestAddr stop_pc) {
  InterpretInsn(state);
  return state->cpu.insn_addr == stop_pc;
}

class Riscv64InterpreterTest : public ::testing::Test {
 public:
  // Non-Compressed Instructions.
  Riscv64InterpreterTest()
      : state_{
            .cpu = {.vtype = uint64_t{1} << 63, .frm = intrinsics::GuestModeFromHostRounding()}} {}

  void InterpretFence(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    InterpretInsn(&state_);
  }

  void TestAtomicLoad(uint32_t insn_bytes,
                      const uint64_t* const data_to_load,
                      uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(data_to_load));
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(GetXReg<2>(state_.cpu), expected_result);
    EXPECT_EQ(state_.cpu.reservation_address, ToGuestAddr(data_to_load));
    // We always reserve the full 64-bit range of the reservation address.
    EXPECT_EQ(state_.cpu.reservation_value, *data_to_load);
  }

  template <typename T>
  void TestAtomicStore(uint32_t insn_bytes, T expected_result) {
    store_area_ = ~uint64_t{0};
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    SetXReg<2>(state_.cpu, kDataToStore);
    SetXReg<3>(state_.cpu, 0xdeadbeef);
    state_.cpu.reservation_address = ToGuestAddr(&store_area_);
    state_.cpu.reservation_value = store_area_;
    MemoryRegionReservation::SetOwner(ToGuestAddr(&store_area_), &state_.cpu);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(static_cast<T>(store_area_), expected_result);
    EXPECT_EQ(GetXReg<3>(state_.cpu), 0u);
  }

  void TestAtomicStoreNoLoadFailure(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    SetXReg<2>(state_.cpu, kDataToStore);
    SetXReg<3>(state_.cpu, 0xdeadbeef);
    store_area_ = 0;
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(store_area_, 0u);
    EXPECT_EQ(GetXReg<3>(state_.cpu), 1u);
  }

  void TestAtomicStoreDifferentLoadFailure(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    SetXReg<2>(state_.cpu, kDataToStore);
    SetXReg<3>(state_.cpu, 0xdeadbeef);
    state_.cpu.reservation_address = ToGuestAddr(&kDataToStore);
    state_.cpu.reservation_value = 0;
    MemoryRegionReservation::SetOwner(ToGuestAddr(&kDataToStore), &state_.cpu);
    store_area_ = 0;
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(store_area_, 0u);
    EXPECT_EQ(GetXReg<3>(state_.cpu), 1u);
  }

  // Vector instructions.

  void TestVectorInstruction(uint32_t insn_bytes,
                             const __v16qu (&expected_result_int8)[8],
                             const __v8hu (&expected_result_int16)[8],
                             const __v4su (&expected_result_int32)[8],
                             const __v2du (&expected_result_int64)[8],
                             const __v2du (&source)[16]) {
    auto Verify = [this, &source](uint32_t insn_bytes,
                                  uint8_t vsew,
                                  uint8_t vlmul_max,
                                  const auto& expected_result,
                                  auto mask) {
      // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
      // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (size_t index = 0; index < std::size(source); ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }
      // Set x1 for vx instructions.
      SetXReg<1>(state_.cpu, 0xaaaa'aaaa'aaaa'aaaa);
      for (uint8_t vlmul = 0; vlmul < vlmul_max; ++vlmul) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          for (uint8_t vma = 0; vma < 2; ++vma) {
            auto [vlmax, vtype] =
                intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
            // Incompatible vsew and vlmax. Skip it.
            if (vlmax == 0) {
              continue;
            }

            // To make tests quick enough we don't test vstart and vl change with small register
            // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of first
            // register and half of last register.
            // Don't use vlmul == 3 because that one may not be supported if instruction widens the
            // result.
            if (vlmul == 2) {
              state_.cpu.vstart = vlmax / 8;
              state_.cpu.vl = (vlmax * 7) / 8;
            } else {
              state_.cpu.vstart = 0;
              state_.cpu.vl = vlmax;
            }
            state_.cpu.vtype = vtype;

            // Set expected_result vector registers into 0b01010101… pattern.
            for (size_t index = 0; index < 8; ++index) {
              state_.cpu.v[8 + index] = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
            }

            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

            if (vlmul < 4) {
              for (size_t index = 0; index < 1 << vlmul; ++index) {
                if (index == 0 && vlmul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            SIMD128Register{
                                (kUndisturbedResult & kFractionMaskInt8[3]) |
                                (expected_result[index] & mask[index] & ~kFractionMaskInt8[3]) |
                                ((vma ? kAgnosticResult : kUndisturbedResult) & ~mask[index] &
                                 ~kFractionMaskInt8[3])}
                                .Get<__uint128_t>());
                } else if (index == 3 && vlmul == 2) {
                  EXPECT_EQ(
                      state_.cpu.v[8 + index],
                      SIMD128Register{
                          (expected_result[index] & mask[index] & kFractionMaskInt8[3]) |
                          ((vma ? kAgnosticResult : kUndisturbedResult) & ~mask[index] &
                           kFractionMaskInt8[3]) |
                          ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[3])}
                          .Get<__uint128_t>());
                } else {
                  EXPECT_EQ(
                      state_.cpu.v[8 + index],
                      SIMD128Register{(expected_result[index] & mask[index]) |
                                      ((vma ? kAgnosticResult : kUndisturbedResult) & ~mask[index])}
                          .Get<__uint128_t>());
                }
              }
            } else {
              EXPECT_EQ(
                  state_.cpu.v[8],
                  SIMD128Register{(expected_result[0] & mask[0] & kFractionMaskInt8[vlmul - 4]) |
                                  ((vma ? kAgnosticResult : kUndisturbedResult) & ~mask[0] &
                                   kFractionMaskInt8[vlmul - 4]) |
                                  ((vta ? kAgnosticResult : kUndisturbedResult) &
                                   ~kFractionMaskInt8[vlmul - 4])}
                      .Get<__uint128_t>());
            }

            if (vlmul == 2) {
              // Every vector instruction must set vstart to 0, but shouldn't touch vl.
              EXPECT_EQ(state_.cpu.vstart, 0);
              EXPECT_EQ(state_.cpu.vl, (vlmax * 7) / 8);
            }
          }
        }
      }
    };

    // Some instructions don't support use of mask register, but in these instructions bit
    // #25 is set.  Test it and skip masking tests if so.
    if ((insn_bytes & (1 << 25)) == 0) {
      Verify(insn_bytes, 0, 8, expected_result_int8, kMaskInt8);
      Verify(insn_bytes, 1, 8, expected_result_int16, kMaskInt16);
      Verify(insn_bytes, 2, 8, expected_result_int32, kMaskInt32);
      Verify(insn_bytes, 3, 8, expected_result_int64, kMaskInt64);
      Verify(insn_bytes | (1 << 25), 0, 8, expected_result_int8, kNoMask);
      Verify(insn_bytes | (1 << 25), 1, 8, expected_result_int16, kNoMask);
      Verify(insn_bytes | (1 << 25), 2, 8, expected_result_int32, kNoMask);
      Verify(insn_bytes | (1 << 25), 3, 8, expected_result_int64, kNoMask);
    } else {
      Verify(insn_bytes, 0, 1, expected_result_int8, kNoMask);
      Verify(insn_bytes, 1, 1, expected_result_int16, kNoMask);
      Verify(insn_bytes, 2, 1, expected_result_int32, kNoMask);
      Verify(insn_bytes, 3, 1, expected_result_int64, kNoMask);
    }
  }

 protected:
  static constexpr __v2du kVectorCalculationsSource[16] = {
      {0x0706'0504'0302'0100, 0x0f0e'0d0c'0b0a'0908},
      {0x1716'1514'1312'1110, 0x1f1e'1d1c'1b1a'1918},
      {0x2726'2524'2322'2120, 0x2f2e'2d2c'2b2a'2928},
      {0x3736'3534'3332'3130, 0x3f3e'3d3c'3b3a'3938},
      {0x4746'4544'4342'4140, 0x4f4e'4d4c'4b4a'4948},
      {0x5756'5554'5352'5150, 0x5f5e'5d5c'5b5a'5958},
      {0x6766'6564'6362'6160, 0x6f6e'6d6c'6b6a'6968},
      {0x7776'7574'7372'7170, 0x7f7e'7d7c'7b7a'7978},

      {0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
      {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
      {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
      {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
      {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
      {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
      {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
      {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}};

  static constexpr __v2du kVectorComparisonSource[16] = {
      {0xfff5'fff5'fff5'fff5, 0xfff5'fff5'fff5'fff5},
      {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
      {0xbbbb'bbbb'bbbb'bbbb, 0xaaaa'aaaa'aaaa'aaaa},
      {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
      {0xfff4'fff4'fff4'fff4, 0xfff6'fff6'fff6'fff6},
      {0xfff8'fff8'fff4'fff4, 0xfff5'fff5'fff5'fff5},
      {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
      {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9},

      {0xfff5'fff5'fff5'fff5, 0xfff5'fff5'fff5'fff5},
      {0x1111'1111'1111'1111, 0x1111'1111'1111'1111},
      {0xfff1'fff1'fff1'fff1, 0xfff1'fff1'fff1'fff1},
      {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
      {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
      {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
      {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
      {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}};

  // Right shift tests should use inputs with 1s in the most significant bit to differentiate
  // between logical and arithmetic right shifts.
  static constexpr __v2du kVectorRightShiftSource[16] = {
      {0xfff5'fff5'fff5'fff5, 0xfff5'fff5'fff5'fff5},
      {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
      {0xbbbb'bbbb'bbbb'bbbb, 0xaaaa'aaaa'aaaa'aaaa},
      {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
      {0xfff4'fff4'fff4'fff4, 0xfff6'fff6'fff6'fff6},
      {0xfff8'fff8'fff4'fff4, 0xfff5'fff5'fff5'fff5},
      {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
      {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9},

      {0xfff5'fff5'fff5'fff5, 0xfff5'fff5'fff5'fff5},
      {0x1111'1111'1111'1111, 0x1111'1111'1111'1111},
      {0xfff1'fff1'fff1'fff1, 0xfff1'fff1'fff1'fff1},
      {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
      {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
      {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
      {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
      {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}};

  // Mask in form suitable for storing in v0 and use in v0.t form.
  static constexpr __v2du kMask = {0xd5ad'd6b5'ad6b'b5ad, 0x6af7'57bb'deed'7bb5};
  // Mask used with vsew = 0 (8bit) elements.
  static constexpr __v16qu kMaskInt8[8] = {
      {255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255},
      {255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255},
      {255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255},
      {255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 0, 255, 255},
      {255, 0, 255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0},
      {255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 255},
      {255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0},
      {255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 0}};
  // Mask used with vsew = 1 (16bit) elements.
  static constexpr __v8hu kMaskInt16[8] = {
      {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
      {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff},
      {0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000},
      {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
      {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff},
      {0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff},
      {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
      {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff}};
  // Mask used with vsew = 2 (32bit) elements.
  static constexpr __v4su kMaskInt32[8] = {{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
                                           {0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
                                           {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0x0000'0000},
                                           {0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
                                           {0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
                                           {0x0000'0000, 0xffff'ffff, 0xffff'ffff, 0x0000'0000},
                                           {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
                                           {0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff}};
  // Mask used with vsew = 3 (64bit) elements.
  static constexpr __v2du kMaskInt64[8] = {
      {0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000},
      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff},
      {0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000},
      {0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000},
      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff},
  };
  // To verify operations without masking.
  static constexpr __v16qu kNoMask[8] = {
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}};
  // Half of sub-register lmul.
  static constexpr __v16qu kFractionMaskInt8[4] = {
      {255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},        // Half of ⅛ reg = ¹⁄₁₆
      {255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},      // Half of ¼ reg = ⅛
      {255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // Half of ½ reg = ¼
      {255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0}};  // Half of full reg = ½
  // Agnostic result is -1 on RISC-V, not 0.
  static constexpr __m128i kAgnosticResult = {-1, -1};
  // Undisturbed result is put in registers v8, v9, …, v15 and is expected to get read back.
  static constexpr __m128i kUndisturbedResult = {0x5555'5555'5555'5555, 0x5555'5555'5555'5555};

  static constexpr uint64_t kDataToLoad{0xffffeeeeddddccccULL};
  static constexpr uint64_t kDataToStore = kDataToLoad;
  uint64_t store_area_;
  ThreadState state_;
};

#define TESTSUITE Riscv64InterpretInsnTest
#define TESTING_INTERPRETER

#include "berberis/test_utils/insn_tests_riscv64-inl.h"

#undef TESTING_INTERPRETER
#undef TESTSUITE

// Tests for Non-Compressed Instructions.

TEST_F(Riscv64InterpreterTest, FenceInstructions) {
  // Fence
  InterpretFence(0x0ff0000f);
  // FenceTso
  InterpretFence(0x8330000f);
  // FenceI
  InterpretFence(0x0000100f);
}

TEST_F(Riscv64InterpreterTest, SyscallWrite) {
  const char message[] = "Hello";
  // Prepare a pipe to write to.
  int pipefd[2];
  ASSERT_EQ(0, pipe(pipefd));

  // SYS_write
  SetXReg<17>(state_.cpu, 0x40);
  // File descriptor
  SetXReg<10>(state_.cpu, pipefd[1]);
  // String
  SetXReg<11>(state_.cpu, bit_cast<uint64_t>(&message[0]));
  // Size
  SetXReg<12>(state_.cpu, sizeof(message));

  uint32_t insn_bytes = 0x00000073;
  state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
  InterpretInsn(&state_);

  // Check number of bytes written.
  EXPECT_EQ(GetXReg<10>(state_.cpu), sizeof(message));

  // Check the message was written to the pipe.
  char buf[sizeof(message)] = {};
  ssize_t read_size = read(pipefd[0], &buf, sizeof(buf));
  EXPECT_NE(read_size, -1);
  EXPECT_EQ(0, strcmp(message, buf));
  close(pipefd[0]);
  close(pipefd[1]);
}

TEST_F(Riscv64InterpreterTest, AtomicLoadInstructions) {
  // Validate sign-extension of returned value.
  const uint64_t kNegative32BitValue = 0x0000'0000'8000'0000ULL;
  const uint64_t kSignExtendedNegative = 0xffff'ffff'8000'0000ULL;
  const uint64_t kPositive32BitValue = 0xffff'ffff'0000'0000ULL;
  const uint64_t kSignExtendedPositive = 0ULL;
  static_assert(static_cast<int32_t>(kSignExtendedPositive) >= 0);
  static_assert(static_cast<int32_t>(kSignExtendedNegative) < 0);

  // Lrw - sign extends from 32 to 64.
  TestAtomicLoad(0x1000a12f, &kPositive32BitValue, kSignExtendedPositive);
  TestAtomicLoad(0x1000a12f, &kNegative32BitValue, kSignExtendedNegative);

  // Lrd
  TestAtomicLoad(0x1000b12f, &kDataToLoad, kDataToLoad);
}

TEST_F(Riscv64InterpreterTest, AtomicStoreInstructions) {
  // Scw
  TestAtomicStore(0x1820a1af, static_cast<uint32_t>(kDataToStore));

  // Scd
  TestAtomicStore(0x1820b1af, kDataToStore);
}

TEST_F(Riscv64InterpreterTest, AtomicStoreInstructionNoLoadFailure) {
  // Scw
  TestAtomicStoreNoLoadFailure(0x1820a1af);

  // Scd
  TestAtomicStoreNoLoadFailure(0x1820b1af);
}

TEST_F(Riscv64InterpreterTest, AtomicStoreInstructionDifferentLoadFailure) {
  // Scw
  TestAtomicStoreDifferentLoadFailure(0x1820a1af);

  // Scd
  TestAtomicStoreDifferentLoadFailure(0x1820b1af);
}

TEST_F(Riscv64InterpreterTest, TestVadd) {
  TestVectorInstruction(
      0x10c0457,  // Vadd.vv v8, v16, v24, v0.t
      {{0, 3, 6, 9, 13, 15, 18, 21, 25, 27, 30, 33, 36, 39, 42, 45},
       {48, 51, 54, 57, 61, 63, 66, 69, 73, 75, 78, 81, 84, 87, 90, 93},
       {96, 99, 102, 105, 109, 111, 114, 117, 121, 123, 126, 129, 132, 135, 138, 141},
       {144, 147, 150, 153, 157, 159, 162, 165, 169, 171, 174, 177, 180, 183, 186, 189},
       {192, 195, 198, 201, 205, 207, 210, 213, 217, 219, 222, 225, 228, 231, 234, 237},
       {240, 243, 246, 249, 253, 255, 2, 5, 9, 11, 14, 17, 20, 23, 26, 29},
       {32, 35, 38, 41, 45, 47, 50, 53, 57, 59, 62, 65, 68, 71, 74, 77},
       {80, 83, 86, 89, 93, 95, 98, 101, 105, 107, 110, 113, 116, 119, 122, 125}},
      {{0x0300, 0x0906, 0x0f0d, 0x1512, 0x1b19, 0x211e, 0x2724, 0x2d2a},
       {0x3330, 0x3936, 0x3f3d, 0x4542, 0x4b49, 0x514e, 0x5754, 0x5d5a},
       {0x6360, 0x6966, 0x6f6d, 0x7572, 0x7b79, 0x817e, 0x8784, 0x8d8a},
       {0x9390, 0x9996, 0x9f9d, 0xa5a2, 0xaba9, 0xb1ae, 0xb7b4, 0xbdba},
       {0xc3c0, 0xc9c6, 0xcfcd, 0xd5d2, 0xdbd9, 0xe1de, 0xe7e4, 0xedea},
       {0xf3f0, 0xf9f6, 0xfffd, 0x0602, 0x0c09, 0x120e, 0x1814, 0x1e1a},
       {0x2420, 0x2a26, 0x302d, 0x3632, 0x3c39, 0x423e, 0x4844, 0x4e4a},
       {0x5450, 0x5a56, 0x605d, 0x6662, 0x6c69, 0x726e, 0x7874, 0x7e7a}},
      {{0x0906'0300, 0x1512'0f0d, 0x211e'1b19, 0x2d2a'2724},
       {0x3936'3330, 0x4542'3f3d, 0x514e'4b49, 0x5d5a'5754},
       {0x6966'6360, 0x7572'6f6d, 0x817e'7b79, 0x8d8a'8784},
       {0x9996'9390, 0xa5a2'9f9d, 0xb1ae'aba9, 0xbdba'b7b4},
       {0xc9c6'c3c0, 0xd5d2'cfcd, 0xe1de'dbd9, 0xedea'e7e4},
       {0xf9f6'f3f0, 0x0602'fffd, 0x120f'0c09, 0x1e1b'1814},
       {0x2a27'2420, 0x3633'302d, 0x423f'3c39, 0x4e4b'4844},
       {0x5a57'5450, 0x6663'605d, 0x726f'6c69, 0x7e7b'7874}},
      {{0x1512'0f0d'0906'0300, 0x2d2a'2724'211e'1b19},
       {0x4542'3f3d'3936'3330, 0x5d5a'5754'514e'4b49},
       {0x7572'6f6d'6966'6360, 0x8d8a'8784'817e'7b79},
       {0xa5a2'9f9d'9996'9390, 0xbdba'b7b4'b1ae'aba9},
       {0xd5d2'cfcd'c9c6'c3c0, 0xedea'e7e4'e1de'dbd9},
       {0x0602'fffd'f9f6'f3f0, 0x1e1b'1815'120f'0c09},
       {0x3633'302e'2a27'2420, 0x4e4b'4845'423f'3c39},
       {0x6663'605e'5a57'5450, 0x7e7b'7875'726f'6c69}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x100c457,  // Vadd.vx v8, v16, x1, v0.t
      {{170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185},
       {186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201},
       {202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217},
       {218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233},
       {234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249},
       {250, 251, 252, 253, 254, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
       {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
       {26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41}},
      {{0xabaa, 0xadac, 0xafae, 0xb1b0, 0xb3b2, 0xb5b4, 0xb7b6, 0xb9b8},
       {0xbbba, 0xbdbc, 0xbfbe, 0xc1c0, 0xc3c2, 0xc5c4, 0xc7c6, 0xc9c8},
       {0xcbca, 0xcdcc, 0xcfce, 0xd1d0, 0xd3d2, 0xd5d4, 0xd7d6, 0xd9d8},
       {0xdbda, 0xdddc, 0xdfde, 0xe1e0, 0xe3e2, 0xe5e4, 0xe7e6, 0xe9e8},
       {0xebea, 0xedec, 0xefee, 0xf1f0, 0xf3f2, 0xf5f4, 0xf7f6, 0xf9f8},
       {0xfbfa, 0xfdfc, 0xfffe, 0x0200, 0x0402, 0x0604, 0x0806, 0x0a08},
       {0x0c0a, 0x0e0c, 0x100e, 0x1210, 0x1412, 0x1614, 0x1816, 0x1a18},
       {0x1c1a, 0x1e1c, 0x201e, 0x2220, 0x2422, 0x2624, 0x2826, 0x2a28}},
      {{0xadac'abaa, 0xb1b0'afae, 0xb5b4'b3b2, 0xb9b8'b7b6},
       {0xbdbc'bbba, 0xc1c0'bfbe, 0xc5c4'c3c2, 0xc9c8'c7c6},
       {0xcdcc'cbca, 0xd1d0'cfce, 0xd5d4'd3d2, 0xd9d8'd7d6},
       {0xdddc'dbda, 0xe1e0'dfde, 0xe5e4'e3e2, 0xe9e8'e7e6},
       {0xedec'ebea, 0xf1f0'efee, 0xf5f4'f3f2, 0xf9f8'f7f6},
       {0xfdfc'fbfa, 0x0200'fffe, 0x0605'0402, 0x0a09'0806},
       {0x0e0d'0c0a, 0x1211'100e, 0x1615'1412, 0x1a19'1816},
       {0x1e1d'1c1a, 0x2221'201e, 0x2625'2422, 0x2a29'2826}},
      {{0xb1b0'afae'adac'abaa, 0xb9b8'b7b6'b5b4'b3b2},
       {0xc1c0'bfbe'bdbc'bbba, 0xc9c8'c7c6'c5c4'c3c2},
       {0xd1d0'cfce'cdcc'cbca, 0xd9d8'd7d6'd5d4'd3d2},
       {0xe1e0'dfde'dddc'dbda, 0xe9e8'e7e6'e5e4'e3e2},
       {0xf1f0'efee'edec'ebea, 0xf9f8'f7f6'f5f4'f3f2},
       {0x0200'fffe'fdfc'fbfa, 0x0a09'0807'0605'0402},
       {0x1211'100f'0e0d'0c0a, 0x1a19'1817'1615'1412},
       {0x2221'201f'1e1d'1c1a, 0x2a29'2827'2625'2422}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x10ab457,  // Vadd.vi v8, v16, -0xb, v0.t
      {{245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 0, 1, 2, 3, 4},
       {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
       {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
       {37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52},
       {53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68},
       {69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84},
       {85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100},
       {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116}},
      {{0x00f5, 0x02f7, 0x04f9, 0x06fb, 0x08fd, 0x0aff, 0x0d01, 0x0f03},
       {0x1105, 0x1307, 0x1509, 0x170b, 0x190d, 0x1b0f, 0x1d11, 0x1f13},
       {0x2115, 0x2317, 0x2519, 0x271b, 0x291d, 0x2b1f, 0x2d21, 0x2f23},
       {0x3125, 0x3327, 0x3529, 0x372b, 0x392d, 0x3b2f, 0x3d31, 0x3f33},
       {0x4135, 0x4337, 0x4539, 0x473b, 0x493d, 0x4b3f, 0x4d41, 0x4f43},
       {0x5145, 0x5347, 0x5549, 0x574b, 0x594d, 0x5b4f, 0x5d51, 0x5f53},
       {0x6155, 0x6357, 0x6559, 0x675b, 0x695d, 0x6b5f, 0x6d61, 0x6f63},
       {0x7165, 0x7367, 0x7569, 0x776b, 0x796d, 0x7b6f, 0x7d71, 0x7f73}},
      {{0x0302'00f5, 0x0706'04f9, 0x0b0a'08fd, 0x0f0e'0d01},
       {0x1312'1105, 0x1716'1509, 0x1b1a'190d, 0x1f1e'1d11},
       {0x2322'2115, 0x2726'2519, 0x2b2a'291d, 0x2f2e'2d21},
       {0x3332'3125, 0x3736'3529, 0x3b3a'392d, 0x3f3e'3d31},
       {0x4342'4135, 0x4746'4539, 0x4b4a'493d, 0x4f4e'4d41},
       {0x5352'5145, 0x5756'5549, 0x5b5a'594d, 0x5f5e'5d51},
       {0x6362'6155, 0x6766'6559, 0x6b6a'695d, 0x6f6e'6d61},
       {0x7372'7165, 0x7776'7569, 0x7b7a'796d, 0x7f7e'7d71}},
      {{0x0706'0504'0302'00f5, 0x0f0e'0d0c'0b0a'08fd},
       {0x1716'1514'1312'1105, 0x1f1e'1d1c'1b1a'190d},
       {0x2726'2524'2322'2115, 0x2f2e'2d2c'2b2a'291d},
       {0x3736'3534'3332'3125, 0x3f3e'3d3c'3b3a'392d},
       {0x4746'4544'4342'4135, 0x4f4e'4d4c'4b4a'493d},
       {0x5756'5554'5352'5145, 0x5f5e'5d5c'5b5a'594d},
       {0x6766'6564'6362'6155, 0x6f6e'6d6c'6b6a'695d},
       {0x7776'7574'7372'7165, 0x7f7e'7d7c'7b7a'796d}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVrsub) {
  TestVectorInstruction(
      0xd00c457,  // Vrsub.vi v8, v16, x1, v0.t
      {{170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155},
       {154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139},
       {138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123},
       {122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107},
       {106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91},
       {90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75},
       {74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59},
       {58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43}},
      {{0xa9aa, 0xa7a8, 0xa5a6, 0xa3a4, 0xa1a2, 0x9fa0, 0x9d9e, 0x9b9c},
       {0x999a, 0x9798, 0x9596, 0x9394, 0x9192, 0x8f90, 0x8d8e, 0x8b8c},
       {0x898a, 0x8788, 0x8586, 0x8384, 0x8182, 0x7f80, 0x7d7e, 0x7b7c},
       {0x797a, 0x7778, 0x7576, 0x7374, 0x7172, 0x6f70, 0x6d6e, 0x6b6c},
       {0x696a, 0x6768, 0x6566, 0x6364, 0x6162, 0x5f60, 0x5d5e, 0x5b5c},
       {0x595a, 0x5758, 0x5556, 0x5354, 0x5152, 0x4f50, 0x4d4e, 0x4b4c},
       {0x494a, 0x4748, 0x4546, 0x4344, 0x4142, 0x3f40, 0x3d3e, 0x3b3c},
       {0x393a, 0x3738, 0x3536, 0x3334, 0x3132, 0x2f30, 0x2d2e, 0x2b2c}},
      {{0xa7a8'a9aa, 0xa3a4'a5a6, 0x9fa0'a1a2, 0x9b9c'9d9e},
       {0x9798'999a, 0x9394'9596, 0x8f90'9192, 0x8b8c'8d8e},
       {0x8788'898a, 0x8384'8586, 0x7f80'8182, 0x7b7c'7d7e},
       {0x7778'797a, 0x7374'7576, 0x6f70'7172, 0x6b6c'6d6e},
       {0x6768'696a, 0x6364'6566, 0x5f60'6162, 0x5b5c'5d5e},
       {0x5758'595a, 0x5354'5556, 0x4f50'5152, 0x4b4c'4d4e},
       {0x4748'494a, 0x4344'4546, 0x3f40'4142, 0x3b3c'3d3e},
       {0x3738'393a, 0x3334'3536, 0x2f30'3132, 0x2b2c'2d2e}},
      {{0xa3a4'a5a6'a7a8'a9aa, 0x9b9c'9d9e'9fa0'a1a2},
       {0x9394'9596'9798'999a, 0x8b8c'8d8e'8f90'9192},
       {0x8384'8586'8788'898a, 0x7b7c'7d7e'7f80'8182},
       {0x7374'7576'7778'797a, 0x6b6c'6d6e'6f70'7172},
       {0x6364'6566'6768'696a, 0x5b5c'5d5e'5f60'6162},
       {0x5354'5556'5758'595a, 0x4b4c'4d4e'4f50'5152},
       {0x4344'4546'4748'494a, 0x3b3c'3d3e'3f40'4142},
       {0x3334'3536'3738'393a, 0x2b2c'2d2e'2f30'3132}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0xd0ab457,  // Vrsub.vi v8, v16, -0xb, v0.t
      {{245, 244, 243, 242, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230},
       {229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216, 215, 214},
       {213, 212, 211, 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 199, 198},
       {197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182},
       {181, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166},
       {165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150},
       {149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134},
       {133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118}},
      {{0xfef5, 0xfcf3, 0xfaf1, 0xf8ef, 0xf6ed, 0xf4eb, 0xf2e9, 0xf0e7},
       {0xeee5, 0xece3, 0xeae1, 0xe8df, 0xe6dd, 0xe4db, 0xe2d9, 0xe0d7},
       {0xded5, 0xdcd3, 0xdad1, 0xd8cf, 0xd6cd, 0xd4cb, 0xd2c9, 0xd0c7},
       {0xcec5, 0xccc3, 0xcac1, 0xc8bf, 0xc6bd, 0xc4bb, 0xc2b9, 0xc0b7},
       {0xbeb5, 0xbcb3, 0xbab1, 0xb8af, 0xb6ad, 0xb4ab, 0xb2a9, 0xb0a7},
       {0xaea5, 0xaca3, 0xaaa1, 0xa89f, 0xa69d, 0xa49b, 0xa299, 0xa097},
       {0x9e95, 0x9c93, 0x9a91, 0x988f, 0x968d, 0x948b, 0x9289, 0x9087},
       {0x8e85, 0x8c83, 0x8a81, 0x887f, 0x867d, 0x847b, 0x8279, 0x8077}},
      {{0xfcfd'fef5, 0xf8f9'faf1, 0xf4f5'f6ed, 0xf0f1'f2e9},
       {0xeced'eee5, 0xe8e9'eae1, 0xe4e5'e6dd, 0xe0e1'e2d9},
       {0xdcdd'ded5, 0xd8d9'dad1, 0xd4d5'd6cd, 0xd0d1'd2c9},
       {0xcccd'cec5, 0xc8c9'cac1, 0xc4c5'c6bd, 0xc0c1'c2b9},
       {0xbcbd'beb5, 0xb8b9'bab1, 0xb4b5'b6ad, 0xb0b1'b2a9},
       {0xacad'aea5, 0xa8a9'aaa1, 0xa4a5'a69d, 0xa0a1'a299},
       {0x9c9d'9e95, 0x9899'9a91, 0x9495'968d, 0x9091'9289},
       {0x8c8d'8e85, 0x8889'8a81, 0x8485'867d, 0x8081'8279}},
      {{0xf8f9'fafb'fcfd'fef5, 0xf0f1'f2f3'f4f5'f6ed},
       {0xe8e9'eaeb'eced'eee5, 0xe0e1'e2e3'e4e5'e6dd},
       {0xd8d9'dadb'dcdd'ded5, 0xd0d1'd2d3'd4d5'd6cd},
       {0xc8c9'cacb'cccd'cec5, 0xc0c1'c2c3'c4c5'c6bd},
       {0xb8b9'babb'bcbd'beb5, 0xb0b1'b2b3'b4b5'b6ad},
       {0xa8a9'aaab'acad'aea5, 0xa0a1'a2a3'a4a5'a69d},
       {0x9899'9a9b'9c9d'9e95, 0x9091'9293'9495'968d},
       {0x8889'8a8b'8c8d'8e85, 0x8081'8283'8485'867d}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVsub) {
  TestVectorInstruction(
      0x90c0457,  // Vsub.vv v8, v16, v24, v0.t
      {{0, 255, 254, 253, 251, 251, 250, 249, 247, 247, 246, 245, 244, 243, 242, 241},
       {240, 239, 238, 237, 235, 235, 234, 233, 231, 231, 230, 229, 228, 227, 226, 225},
       {224, 223, 222, 221, 219, 219, 218, 217, 215, 215, 214, 213, 212, 211, 210, 209},
       {208, 207, 206, 205, 203, 203, 202, 201, 199, 199, 198, 197, 196, 195, 194, 193},
       {192, 191, 190, 189, 187, 187, 186, 185, 183, 183, 182, 181, 180, 179, 178, 177},
       {176, 175, 174, 173, 171, 171, 170, 169, 167, 167, 166, 165, 164, 163, 162, 161},
       {160, 159, 158, 157, 155, 155, 154, 153, 151, 151, 150, 149, 148, 147, 146, 145},
       {144, 143, 142, 141, 139, 139, 138, 137, 135, 135, 134, 133, 132, 131, 130, 129}},
      {{0xff00, 0xfcfe, 0xfafb, 0xf8fa, 0xf6f7, 0xf4f6, 0xf2f4, 0xf0f2},
       {0xeef0, 0xecee, 0xeaeb, 0xe8ea, 0xe6e7, 0xe4e6, 0xe2e4, 0xe0e2},
       {0xdee0, 0xdcde, 0xdadb, 0xd8da, 0xd6d7, 0xd4d6, 0xd2d4, 0xd0d2},
       {0xced0, 0xccce, 0xcacb, 0xc8ca, 0xc6c7, 0xc4c6, 0xc2c4, 0xc0c2},
       {0xbec0, 0xbcbe, 0xbabb, 0xb8ba, 0xb6b7, 0xb4b6, 0xb2b4, 0xb0b2},
       {0xaeb0, 0xacae, 0xaaab, 0xa8aa, 0xa6a7, 0xa4a6, 0xa2a4, 0xa0a2},
       {0x9ea0, 0x9c9e, 0x9a9b, 0x989a, 0x9697, 0x9496, 0x9294, 0x9092},
       {0x8e90, 0x8c8e, 0x8a8b, 0x888a, 0x8687, 0x8486, 0x8284, 0x8082}},
      {{0xfcfd'ff00, 0xf8f9'fafb, 0xf4f5'f6f7, 0xf0f1'f2f4},
       {0xeced'eef0, 0xe8e9'eaeb, 0xe4e5'e6e7, 0xe0e1'e2e4},
       {0xdcdd'dee0, 0xd8d9'dadb, 0xd4d5'd6d7, 0xd0d1'd2d4},
       {0xcccd'ced0, 0xc8c9'cacb, 0xc4c5'c6c7, 0xc0c1'c2c4},
       {0xbcbd'bec0, 0xb8b9'babb, 0xb4b5'b6b7, 0xb0b1'b2b4},
       {0xacad'aeb0, 0xa8a9'aaab, 0xa4a5'a6a7, 0xa0a1'a2a4},
       {0x9c9d'9ea0, 0x9899'9a9b, 0x9495'9697, 0x9091'9294},
       {0x8c8d'8e90, 0x8889'8a8b, 0x8485'8687, 0x8081'8284}},
      {{0xf8f9'fafa'fcfd'ff00, 0xf0f1'f2f3'f4f5'f6f7},
       {0xe8e9'eaea'eced'eef0, 0xe0e1'e2e3'e4e5'e6e7},
       {0xd8d9'dada'dcdd'dee0, 0xd0d1'd2d3'd4d5'd6d7},
       {0xc8c9'caca'cccd'ced0, 0xc0c1'c2c3'c4c5'c6c7},
       {0xb8b9'baba'bcbd'bec0, 0xb0b1'b2b3'b4b5'b6b7},
       {0xa8a9'aaaa'acad'aeb0, 0xa0a1'a2a3'a4a5'a6a7},
       {0x9899'9a9a'9c9d'9ea0, 0x9091'9293'9495'9697},
       {0x8889'8a8a'8c8d'8e90, 0x8081'8283'8485'8687}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x900c457,  // Vsub.vx v8, v16, x1, v0.t
      {{86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101},
       {102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117},
       {118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133},
       {134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149},
       {150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165},
       {166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181},
       {182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197},
       {198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213}},
      {{0x5656, 0x5858, 0x5a5a, 0x5c5c, 0x5e5e, 0x6060, 0x6262, 0x6464},
       {0x6666, 0x6868, 0x6a6a, 0x6c6c, 0x6e6e, 0x7070, 0x7272, 0x7474},
       {0x7676, 0x7878, 0x7a7a, 0x7c7c, 0x7e7e, 0x8080, 0x8282, 0x8484},
       {0x8686, 0x8888, 0x8a8a, 0x8c8c, 0x8e8e, 0x9090, 0x9292, 0x9494},
       {0x9696, 0x9898, 0x9a9a, 0x9c9c, 0x9e9e, 0xa0a0, 0xa2a2, 0xa4a4},
       {0xa6a6, 0xa8a8, 0xaaaa, 0xacac, 0xaeae, 0xb0b0, 0xb2b2, 0xb4b4},
       {0xb6b6, 0xb8b8, 0xbaba, 0xbcbc, 0xbebe, 0xc0c0, 0xc2c2, 0xc4c4},
       {0xc6c6, 0xc8c8, 0xcaca, 0xcccc, 0xcece, 0xd0d0, 0xd2d2, 0xd4d4}},
      {{0x5857'5656, 0x5c5b'5a5a, 0x605f'5e5e, 0x6463'6262},
       {0x6867'6666, 0x6c6b'6a6a, 0x706f'6e6e, 0x7473'7272},
       {0x7877'7676, 0x7c7b'7a7a, 0x807f'7e7e, 0x8483'8282},
       {0x8887'8686, 0x8c8b'8a8a, 0x908f'8e8e, 0x9493'9292},
       {0x9897'9696, 0x9c9b'9a9a, 0xa09f'9e9e, 0xa4a3'a2a2},
       {0xa8a7'a6a6, 0xacab'aaaa, 0xb0af'aeae, 0xb4b3'b2b2},
       {0xb8b7'b6b6, 0xbcbb'baba, 0xc0bf'bebe, 0xc4c3'c2c2},
       {0xc8c7'c6c6, 0xcccb'caca, 0xd0cf'cece, 0xd4d3'd2d2}},
      {{0x5c5b'5a59'5857'5656, 0x6463'6261'605f'5e5e},
       {0x6c6b'6a69'6867'6666, 0x7473'7271'706f'6e6e},
       {0x7c7b'7a79'7877'7676, 0x8483'8281'807f'7e7e},
       {0x8c8b'8a89'8887'8686, 0x9493'9291'908f'8e8e},
       {0x9c9b'9a99'9897'9696, 0xa4a3'a2a1'a09f'9e9e},
       {0xacab'aaa9'a8a7'a6a6, 0xb4b3'b2b1'b0af'aeae},
       {0xbcbb'bab9'b8b7'b6b6, 0xc4c3'c2c1'c0bf'bebe},
       {0xcccb'cac9'c8c7'c6c6, 0xd4d3'd2d1'd0cf'cece}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVand) {
  TestVectorInstruction(
      0x250c0457,  // Vand.vv v8, v16, v24, v0.t
      {{0, 0, 0, 2, 0, 0, 4, 6, 0, 0, 0, 2, 8, 8, 12, 14},
       {0, 0, 0, 2, 0, 0, 4, 6, 16, 16, 16, 18, 24, 24, 28, 30},
       {0, 0, 0, 2, 0, 0, 4, 6, 0, 0, 0, 2, 8, 8, 12, 14},
       {32, 32, 32, 34, 32, 32, 36, 38, 48, 48, 48, 50, 56, 56, 60, 62},
       {0, 0, 0, 2, 0, 0, 4, 6, 0, 0, 0, 2, 8, 8, 12, 14},
       {0, 0, 0, 2, 0, 0, 4, 6, 16, 16, 16, 18, 24, 24, 28, 30},
       {64, 64, 64, 66, 64, 64, 68, 70, 64, 64, 64, 66, 72, 72, 76, 78},
       {96, 96, 96, 98, 96, 96, 100, 102, 112, 112, 112, 114, 120, 120, 124, 126}},
      {{0x0000, 0x0200, 0x0000, 0x0604, 0x0000, 0x0200, 0x0808, 0x0e0c},
       {0x0000, 0x0200, 0x0000, 0x0604, 0x1010, 0x1210, 0x1818, 0x1e1c},
       {0x0000, 0x0200, 0x0000, 0x0604, 0x0000, 0x0200, 0x0808, 0x0e0c},
       {0x2020, 0x2220, 0x2020, 0x2624, 0x3030, 0x3230, 0x3838, 0x3e3c},
       {0x0000, 0x0200, 0x0000, 0x0604, 0x0000, 0x0200, 0x0808, 0x0e0c},
       {0x0000, 0x0200, 0x0000, 0x0604, 0x1010, 0x1210, 0x1818, 0x1e1c},
       {0x4040, 0x4240, 0x4040, 0x4644, 0x4040, 0x4240, 0x4848, 0x4e4c},
       {0x6060, 0x6260, 0x6060, 0x6664, 0x7070, 0x7270, 0x7878, 0x7e7c}},
      {{0x0200'0000, 0x0604'0000, 0x0200'0000, 0x0e0c'0808},
       {0x0200'0000, 0x0604'0000, 0x1210'1010, 0x1e1c'1818},
       {0x0200'0000, 0x0604'0000, 0x0200'0000, 0x0e0c'0808},
       {0x2220'2020, 0x2624'2020, 0x3230'3030, 0x3e3c'3838},
       {0x0200'0000, 0x0604'0000, 0x0200'0000, 0x0e0c'0808},
       {0x0200'0000, 0x0604'0000, 0x1210'1010, 0x1e1c'1818},
       {0x4240'4040, 0x4644'4040, 0x4240'4040, 0x4e4c'4848},
       {0x6260'6060, 0x6664'6060, 0x7270'7070, 0x7e7c'7878}},
      {{0x0604'0000'0200'0000, 0x0e0c'0808'0200'0000},
       {0x0604'0000'0200'0000, 0x1e1c'1818'1210'1010},
       {0x0604'0000'0200'0000, 0x0e0c'0808'0200'0000},
       {0x2624'2020'2220'2020, 0x3e3c'3838'3230'3030},
       {0x0604'0000'0200'0000, 0x0e0c'0808'0200'0000},
       {0x0604'0000'0200'0000, 0x1e1c'1818'1210'1010},
       {0x4644'4040'4240'4040, 0x4e4c'4848'4240'4040},
       {0x6664'6060'6260'6060, 0x7e7c'7878'7270'7070}},
      kVectorCalculationsSource);
  TestVectorInstruction(0x2500c457,  // Vand.vx v8, v16, x1, v0.t
                        {{0, 0, 2, 2, 0, 0, 2, 2, 8, 8, 10, 10, 8, 8, 10, 10},
                         {0, 0, 2, 2, 0, 0, 2, 2, 8, 8, 10, 10, 8, 8, 10, 10},
                         {32, 32, 34, 34, 32, 32, 34, 34, 40, 40, 42, 42, 40, 40, 42, 42},
                         {32, 32, 34, 34, 32, 32, 34, 34, 40, 40, 42, 42, 40, 40, 42, 42},
                         {0, 0, 2, 2, 0, 0, 2, 2, 8, 8, 10, 10, 8, 8, 10, 10},
                         {0, 0, 2, 2, 0, 0, 2, 2, 8, 8, 10, 10, 8, 8, 10, 10},
                         {32, 32, 34, 34, 32, 32, 34, 34, 40, 40, 42, 42, 40, 40, 42, 42},
                         {32, 32, 34, 34, 32, 32, 34, 34, 40, 40, 42, 42, 40, 40, 42, 42}},
                        {{0x0000, 0x0202, 0x0000, 0x0202, 0x0808, 0x0a0a, 0x0808, 0x0a0a},
                         {0x0000, 0x0202, 0x0000, 0x0202, 0x0808, 0x0a0a, 0x0808, 0x0a0a},
                         {0x2020, 0x2222, 0x2020, 0x2222, 0x2828, 0x2a2a, 0x2828, 0x2a2a},
                         {0x2020, 0x2222, 0x2020, 0x2222, 0x2828, 0x2a2a, 0x2828, 0x2a2a},
                         {0x0000, 0x0202, 0x0000, 0x0202, 0x0808, 0x0a0a, 0x0808, 0x0a0a},
                         {0x0000, 0x0202, 0x0000, 0x0202, 0x0808, 0x0a0a, 0x0808, 0x0a0a},
                         {0x2020, 0x2222, 0x2020, 0x2222, 0x2828, 0x2a2a, 0x2828, 0x2a2a},
                         {0x2020, 0x2222, 0x2020, 0x2222, 0x2828, 0x2a2a, 0x2828, 0x2a2a}},
                        {{0x0202'0000, 0x0202'0000, 0x0a0a'0808, 0x0a0a'0808},
                         {0x0202'0000, 0x0202'0000, 0x0a0a'0808, 0x0a0a'0808},
                         {0x2222'2020, 0x2222'2020, 0x2a2a'2828, 0x2a2a'2828},
                         {0x2222'2020, 0x2222'2020, 0x2a2a'2828, 0x2a2a'2828},
                         {0x0202'0000, 0x0202'0000, 0x0a0a'0808, 0x0a0a'0808},
                         {0x0202'0000, 0x0202'0000, 0x0a0a'0808, 0x0a0a'0808},
                         {0x2222'2020, 0x2222'2020, 0x2a2a'2828, 0x2a2a'2828},
                         {0x2222'2020, 0x2222'2020, 0x2a2a'2828, 0x2a2a'2828}},
                        {{0x0202'0000'0202'0000, 0x0a0a'0808'0a0a'0808},
                         {0x0202'0000'0202'0000, 0x0a0a'0808'0a0a'0808},
                         {0x2222'2020'2222'2020, 0x2a2a'2828'2a2a'2828},
                         {0x2222'2020'2222'2020, 0x2a2a'2828'2a2a'2828},
                         {0x0202'0000'0202'0000, 0x0a0a'0808'0a0a'0808},
                         {0x0202'0000'0202'0000, 0x0a0a'0808'0a0a'0808},
                         {0x2222'2020'2222'2020, 0x2a2a'2828'2a2a'2828},
                         {0x2222'2020'2222'2020, 0x2a2a'2828'2a2a'2828}},
                        kVectorCalculationsSource);
  TestVectorInstruction(
      0x250ab457,  // Vand.vi v8, v16, -0xb, v0.t
      {{0, 1, 0, 1, 4, 5, 4, 5, 0, 1, 0, 1, 4, 5, 4, 5},
       {16, 17, 16, 17, 20, 21, 20, 21, 16, 17, 16, 17, 20, 21, 20, 21},
       {32, 33, 32, 33, 36, 37, 36, 37, 32, 33, 32, 33, 36, 37, 36, 37},
       {48, 49, 48, 49, 52, 53, 52, 53, 48, 49, 48, 49, 52, 53, 52, 53},
       {64, 65, 64, 65, 68, 69, 68, 69, 64, 65, 64, 65, 68, 69, 68, 69},
       {80, 81, 80, 81, 84, 85, 84, 85, 80, 81, 80, 81, 84, 85, 84, 85},
       {96, 97, 96, 97, 100, 101, 100, 101, 96, 97, 96, 97, 100, 101, 100, 101},
       {112, 113, 112, 113, 116, 117, 116, 117, 112, 113, 112, 113, 116, 117, 116, 117}},
      {{0x0100, 0x0300, 0x0504, 0x0704, 0x0900, 0x0b00, 0x0d04, 0x0f04},
       {0x1110, 0x1310, 0x1514, 0x1714, 0x1910, 0x1b10, 0x1d14, 0x1f14},
       {0x2120, 0x2320, 0x2524, 0x2724, 0x2920, 0x2b20, 0x2d24, 0x2f24},
       {0x3130, 0x3330, 0x3534, 0x3734, 0x3930, 0x3b30, 0x3d34, 0x3f34},
       {0x4140, 0x4340, 0x4544, 0x4744, 0x4940, 0x4b40, 0x4d44, 0x4f44},
       {0x5150, 0x5350, 0x5554, 0x5754, 0x5950, 0x5b50, 0x5d54, 0x5f54},
       {0x6160, 0x6360, 0x6564, 0x6764, 0x6960, 0x6b60, 0x6d64, 0x6f64},
       {0x7170, 0x7370, 0x7574, 0x7774, 0x7970, 0x7b70, 0x7d74, 0x7f74}},
      {{0x0302'0100, 0x0706'0504, 0x0b0a'0900, 0x0f0e'0d04},
       {0x1312'1110, 0x1716'1514, 0x1b1a'1910, 0x1f1e'1d14},
       {0x2322'2120, 0x2726'2524, 0x2b2a'2920, 0x2f2e'2d24},
       {0x3332'3130, 0x3736'3534, 0x3b3a'3930, 0x3f3e'3d34},
       {0x4342'4140, 0x4746'4544, 0x4b4a'4940, 0x4f4e'4d44},
       {0x5352'5150, 0x5756'5554, 0x5b5a'5950, 0x5f5e'5d54},
       {0x6362'6160, 0x6766'6564, 0x6b6a'6960, 0x6f6e'6d64},
       {0x7372'7170, 0x7776'7574, 0x7b7a'7970, 0x7f7e'7d74}},
      {{0x0706'0504'0302'0100, 0x0f0e'0d0c'0b0a'0900},
       {0x1716'1514'1312'1110, 0x1f1e'1d1c'1b1a'1910},
       {0x2726'2524'2322'2120, 0x2f2e'2d2c'2b2a'2920},
       {0x3736'3534'3332'3130, 0x3f3e'3d3c'3b3a'3930},
       {0x4746'4544'4342'4140, 0x4f4e'4d4c'4b4a'4940},
       {0x5756'5554'5352'5150, 0x5f5e'5d5c'5b5a'5950},
       {0x6766'6564'6362'6160, 0x6f6e'6d6c'6b6a'6960},
       {0x7776'7574'7372'7170, 0x7f7e'7d7c'7b7a'7970}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVor) {
  TestVectorInstruction(
      0x290c0457,  // Vor.vv v8, v16, v24, v0.t
      {{0, 3, 6, 7, 13, 15, 14, 15, 25, 27, 30, 31, 28, 31, 30, 31},
       {48, 51, 54, 55, 61, 63, 62, 63, 57, 59, 62, 63, 60, 63, 62, 63},
       {96, 99, 102, 103, 109, 111, 110, 111, 121, 123, 126, 127, 124, 127, 126, 127},
       {112, 115, 118, 119, 125, 127, 126, 127, 121, 123, 126, 127, 124, 127, 126, 127},
       {192, 195, 198, 199, 205, 207, 206, 207, 217, 219, 222, 223, 220, 223, 222, 223},
       {240, 243, 246, 247, 253, 255, 254, 255, 249, 251, 254, 255, 252, 255, 254, 255},
       {224, 227, 230, 231, 237, 239, 238, 239, 249, 251, 254, 255, 252, 255, 254, 255},
       {240, 243, 246, 247, 253, 255, 254, 255, 249, 251, 254, 255, 252, 255, 254, 255}},
      {{0x0300, 0x0706, 0x0f0d, 0x0f0e, 0x1b19, 0x1f1e, 0x1f1c, 0x1f1e},
       {0x3330, 0x3736, 0x3f3d, 0x3f3e, 0x3b39, 0x3f3e, 0x3f3c, 0x3f3e},
       {0x6360, 0x6766, 0x6f6d, 0x6f6e, 0x7b79, 0x7f7e, 0x7f7c, 0x7f7e},
       {0x7370, 0x7776, 0x7f7d, 0x7f7e, 0x7b79, 0x7f7e, 0x7f7c, 0x7f7e},
       {0xc3c0, 0xc7c6, 0xcfcd, 0xcfce, 0xdbd9, 0xdfde, 0xdfdc, 0xdfde},
       {0xf3f0, 0xf7f6, 0xfffd, 0xfffe, 0xfbf9, 0xfffe, 0xfffc, 0xfffe},
       {0xe3e0, 0xe7e6, 0xefed, 0xefee, 0xfbf9, 0xfffe, 0xfffc, 0xfffe},
       {0xf3f0, 0xf7f6, 0xfffd, 0xfffe, 0xfbf9, 0xfffe, 0xfffc, 0xfffe}},
      {{0x0706'0300, 0x0f0e'0f0d, 0x1f1e'1b19, 0x1f1e'1f1c},
       {0x3736'3330, 0x3f3e'3f3d, 0x3f3e'3b39, 0x3f3e'3f3c},
       {0x6766'6360, 0x6f6e'6f6d, 0x7f7e'7b79, 0x7f7e'7f7c},
       {0x7776'7370, 0x7f7e'7f7d, 0x7f7e'7b79, 0x7f7e'7f7c},
       {0xc7c6'c3c0, 0xcfce'cfcd, 0xdfde'dbd9, 0xdfde'dfdc},
       {0xf7f6'f3f0, 0xfffe'fffd, 0xfffe'fbf9, 0xfffe'fffc},
       {0xe7e6'e3e0, 0xefee'efed, 0xfffe'fbf9, 0xfffe'fffc},
       {0xf7f6'f3f0, 0xfffe'fffd, 0xfffe'fbf9, 0xfffe'fffc}},
      {{0x0f0e'0f0d'0706'0300, 0x1f1e'1f1c'1f1e'1b19},
       {0x3f3e'3f3d'3736'3330, 0x3f3e'3f3c'3f3e'3b39},
       {0x6f6e'6f6d'6766'6360, 0x7f7e'7f7c'7f7e'7b79},
       {0x7f7e'7f7d'7776'7370, 0x7f7e'7f7c'7f7e'7b79},
       {0xcfce'cfcd'c7c6'c3c0, 0xdfde'dfdc'dfde'dbd9},
       {0xfffe'fffd'f7f6'f3f0, 0xfffe'fffc'fffe'fbf9},
       {0xefee'efed'e7e6'e3e0, 0xfffe'fffc'fffe'fbf9},
       {0xfffe'fffd'f7f6'f3f0, 0xfffe'fffc'fffe'fbf9}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x2900c457,  // Vor.vx v8, v16, x1, v0.t
      {{170, 171, 170, 171, 174, 175, 174, 175, 170, 171, 170, 171, 174, 175, 174, 175},
       {186, 187, 186, 187, 190, 191, 190, 191, 186, 187, 186, 187, 190, 191, 190, 191},
       {170, 171, 170, 171, 174, 175, 174, 175, 170, 171, 170, 171, 174, 175, 174, 175},
       {186, 187, 186, 187, 190, 191, 190, 191, 186, 187, 186, 187, 190, 191, 190, 191},
       {234, 235, 234, 235, 238, 239, 238, 239, 234, 235, 234, 235, 238, 239, 238, 239},
       {250, 251, 250, 251, 254, 255, 254, 255, 250, 251, 250, 251, 254, 255, 254, 255},
       {234, 235, 234, 235, 238, 239, 238, 239, 234, 235, 234, 235, 238, 239, 238, 239},
       {250, 251, 250, 251, 254, 255, 254, 255, 250, 251, 250, 251, 254, 255, 254, 255}},
      {{0xabaa, 0xabaa, 0xafae, 0xafae, 0xabaa, 0xabaa, 0xafae, 0xafae},
       {0xbbba, 0xbbba, 0xbfbe, 0xbfbe, 0xbbba, 0xbbba, 0xbfbe, 0xbfbe},
       {0xabaa, 0xabaa, 0xafae, 0xafae, 0xabaa, 0xabaa, 0xafae, 0xafae},
       {0xbbba, 0xbbba, 0xbfbe, 0xbfbe, 0xbbba, 0xbbba, 0xbfbe, 0xbfbe},
       {0xebea, 0xebea, 0xefee, 0xefee, 0xebea, 0xebea, 0xefee, 0xefee},
       {0xfbfa, 0xfbfa, 0xfffe, 0xfffe, 0xfbfa, 0xfbfa, 0xfffe, 0xfffe},
       {0xebea, 0xebea, 0xefee, 0xefee, 0xebea, 0xebea, 0xefee, 0xefee},
       {0xfbfa, 0xfbfa, 0xfffe, 0xfffe, 0xfbfa, 0xfbfa, 0xfffe, 0xfffe}},
      {{0xabaa'abaa, 0xafae'afae, 0xabaa'abaa, 0xafae'afae},
       {0xbbba'bbba, 0xbfbe'bfbe, 0xbbba'bbba, 0xbfbe'bfbe},
       {0xabaa'abaa, 0xafae'afae, 0xabaa'abaa, 0xafae'afae},
       {0xbbba'bbba, 0xbfbe'bfbe, 0xbbba'bbba, 0xbfbe'bfbe},
       {0xebea'ebea, 0xefee'efee, 0xebea'ebea, 0xefee'efee},
       {0xfbfa'fbfa, 0xfffe'fffe, 0xfbfa'fbfa, 0xfffe'fffe},
       {0xebea'ebea, 0xefee'efee, 0xebea'ebea, 0xefee'efee},
       {0xfbfa'fbfa, 0xfffe'fffe, 0xfbfa'fbfa, 0xfffe'fffe}},
      {{0xafae'afae'abaa'abaa, 0xafae'afae'abaa'abaa},
       {0xbfbe'bfbe'bbba'bbba, 0xbfbe'bfbe'bbba'bbba},
       {0xafae'afae'abaa'abaa, 0xafae'afae'abaa'abaa},
       {0xbfbe'bfbe'bbba'bbba, 0xbfbe'bfbe'bbba'bbba},
       {0xefee'efee'ebea'ebea, 0xefee'efee'ebea'ebea},
       {0xfffe'fffe'fbfa'fbfa, 0xfffe'fffe'fbfa'fbfa},
       {0xefee'efee'ebea'ebea, 0xefee'efee'ebea'ebea},
       {0xfffe'fffe'fbfa'fbfa, 0xfffe'fffe'fbfa'fbfa}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x290ab457,  // Vor.vi v8, v16, -0xb, v0.t
      {{245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255}},
      {{0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff}},
      {{0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd}},
      {{0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVxor) {
  TestVectorInstruction(
      0x2d0c0457,  // Vxor.vv v8, v16, v24, v0.t
      {{0, 3, 6, 5, 13, 15, 10, 9, 25, 27, 30, 29, 20, 23, 18, 17},
       {48, 51, 54, 53, 61, 63, 58, 57, 41, 43, 46, 45, 36, 39, 34, 33},
       {96, 99, 102, 101, 109, 111, 106, 105, 121, 123, 126, 125, 116, 119, 114, 113},
       {80, 83, 86, 85, 93, 95, 90, 89, 73, 75, 78, 77, 68, 71, 66, 65},
       {192, 195, 198, 197, 205, 207, 202, 201, 217, 219, 222, 221, 212, 215, 210, 209},
       {240, 243, 246, 245, 253, 255, 250, 249, 233, 235, 238, 237, 228, 231, 226, 225},
       {160, 163, 166, 165, 173, 175, 170, 169, 185, 187, 190, 189, 180, 183, 178, 177},
       {144, 147, 150, 149, 157, 159, 154, 153, 137, 139, 142, 141, 132, 135, 130, 129}},
      {{0x0300, 0x0506, 0x0f0d, 0x090a, 0x1b19, 0x1d1e, 0x1714, 0x1112},
       {0x3330, 0x3536, 0x3f3d, 0x393a, 0x2b29, 0x2d2e, 0x2724, 0x2122},
       {0x6360, 0x6566, 0x6f6d, 0x696a, 0x7b79, 0x7d7e, 0x7774, 0x7172},
       {0x5350, 0x5556, 0x5f5d, 0x595a, 0x4b49, 0x4d4e, 0x4744, 0x4142},
       {0xc3c0, 0xc5c6, 0xcfcd, 0xc9ca, 0xdbd9, 0xddde, 0xd7d4, 0xd1d2},
       {0xf3f0, 0xf5f6, 0xfffd, 0xf9fa, 0xebe9, 0xedee, 0xe7e4, 0xe1e2},
       {0xa3a0, 0xa5a6, 0xafad, 0xa9aa, 0xbbb9, 0xbdbe, 0xb7b4, 0xb1b2},
       {0x9390, 0x9596, 0x9f9d, 0x999a, 0x8b89, 0x8d8e, 0x8784, 0x8182}},
      {{0x0506'0300, 0x090a'0f0d, 0x1d1e'1b19, 0x1112'1714},
       {0x3536'3330, 0x393a'3f3d, 0x2d2e'2b29, 0x2122'2724},
       {0x6566'6360, 0x696a'6f6d, 0x7d7e'7b79, 0x7172'7774},
       {0x5556'5350, 0x595a'5f5d, 0x4d4e'4b49, 0x4142'4744},
       {0xc5c6'c3c0, 0xc9ca'cfcd, 0xddde'dbd9, 0xd1d2'd7d4},
       {0xf5f6'f3f0, 0xf9fa'fffd, 0xedee'ebe9, 0xe1e2'e7e4},
       {0xa5a6'a3a0, 0xa9aa'afad, 0xbdbe'bbb9, 0xb1b2'b7b4},
       {0x9596'9390, 0x999a'9f9d, 0x8d8e'8b89, 0x8182'8784}},
      {{0x090a'0f0d'0506'0300, 0x1112'1714'1d1e'1b19},
       {0x393a'3f3d'3536'3330, 0x2122'2724'2d2e'2b29},
       {0x696a'6f6d'6566'6360, 0x7172'7774'7d7e'7b79},
       {0x595a'5f5d'5556'5350, 0x4142'4744'4d4e'4b49},
       {0xc9ca'cfcd'c5c6'c3c0, 0xd1d2'd7d4'ddde'dbd9},
       {0xf9fa'fffd'f5f6'f3f0, 0xe1e2'e7e4'edee'ebe9},
       {0xa9aa'afad'a5a6'a3a0, 0xb1b2'b7b4'bdbe'bbb9},
       {0x999a'9f9d'9596'9390, 0x8182'8784'8d8e'8b89}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x2d00c457,  // Vxor.vx v8, v16, x1, v0.t
      {{170, 171, 168, 169, 174, 175, 172, 173, 162, 163, 160, 161, 166, 167, 164, 165},
       {186, 187, 184, 185, 190, 191, 188, 189, 178, 179, 176, 177, 182, 183, 180, 181},
       {138, 139, 136, 137, 142, 143, 140, 141, 130, 131, 128, 129, 134, 135, 132, 133},
       {154, 155, 152, 153, 158, 159, 156, 157, 146, 147, 144, 145, 150, 151, 148, 149},
       {234, 235, 232, 233, 238, 239, 236, 237, 226, 227, 224, 225, 230, 231, 228, 229},
       {250, 251, 248, 249, 254, 255, 252, 253, 242, 243, 240, 241, 246, 247, 244, 245},
       {202, 203, 200, 201, 206, 207, 204, 205, 194, 195, 192, 193, 198, 199, 196, 197},
       {218, 219, 216, 217, 222, 223, 220, 221, 210, 211, 208, 209, 214, 215, 212, 213}},
      {{0xabaa, 0xa9a8, 0xafae, 0xadac, 0xa3a2, 0xa1a0, 0xa7a6, 0xa5a4},
       {0xbbba, 0xb9b8, 0xbfbe, 0xbdbc, 0xb3b2, 0xb1b0, 0xb7b6, 0xb5b4},
       {0x8b8a, 0x8988, 0x8f8e, 0x8d8c, 0x8382, 0x8180, 0x8786, 0x8584},
       {0x9b9a, 0x9998, 0x9f9e, 0x9d9c, 0x9392, 0x9190, 0x9796, 0x9594},
       {0xebea, 0xe9e8, 0xefee, 0xedec, 0xe3e2, 0xe1e0, 0xe7e6, 0xe5e4},
       {0xfbfa, 0xf9f8, 0xfffe, 0xfdfc, 0xf3f2, 0xf1f0, 0xf7f6, 0xf5f4},
       {0xcbca, 0xc9c8, 0xcfce, 0xcdcc, 0xc3c2, 0xc1c0, 0xc7c6, 0xc5c4},
       {0xdbda, 0xd9d8, 0xdfde, 0xdddc, 0xd3d2, 0xd1d0, 0xd7d6, 0xd5d4}},
      {{0xa9a8'abaa, 0xadac'afae, 0xa1a0'a3a2, 0xa5a4'a7a6},
       {0xb9b8'bbba, 0xbdbc'bfbe, 0xb1b0'b3b2, 0xb5b4'b7b6},
       {0x8988'8b8a, 0x8d8c'8f8e, 0x8180'8382, 0x8584'8786},
       {0x9998'9b9a, 0x9d9c'9f9e, 0x9190'9392, 0x9594'9796},
       {0xe9e8'ebea, 0xedec'efee, 0xe1e0'e3e2, 0xe5e4'e7e6},
       {0xf9f8'fbfa, 0xfdfc'fffe, 0xf1f0'f3f2, 0xf5f4'f7f6},
       {0xc9c8'cbca, 0xcdcc'cfce, 0xc1c0'c3c2, 0xc5c4'c7c6},
       {0xd9d8'dbda, 0xdddc'dfde, 0xd1d0'd3d2, 0xd5d4'd7d6}},
      {{0xadac'afae'a9a8'abaa, 0xa5a4'a7a6'a1a0'a3a2},
       {0xbdbc'bfbe'b9b8'bbba, 0xb5b4'b7b6'b1b0'b3b2},
       {0x8d8c'8f8e'8988'8b8a, 0x8584'8786'8180'8382},
       {0x9d9c'9f9e'9998'9b9a, 0x9594'9796'9190'9392},
       {0xedec'efee'e9e8'ebea, 0xe5e4'e7e6'e1e0'e3e2},
       {0xfdfc'fffe'f9f8'fbfa, 0xf5f4'f7f6'f1f0'f3f2},
       {0xcdcc'cfce'c9c8'cbca, 0xc5c4'c7c6'c1c0'c3c2},
       {0xdddc'dfde'd9d8'dbda, 0xd5d4'd7d6'd1d0'd3d2}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x2d0ab457,  // Vxor.vi v8, v16, -0xb, v0.t
      {{245, 244, 247, 246, 241, 240, 243, 242, 253, 252, 255, 254, 249, 248, 251, 250},
       {229, 228, 231, 230, 225, 224, 227, 226, 237, 236, 239, 238, 233, 232, 235, 234},
       {213, 212, 215, 214, 209, 208, 211, 210, 221, 220, 223, 222, 217, 216, 219, 218},
       {197, 196, 199, 198, 193, 192, 195, 194, 205, 204, 207, 206, 201, 200, 203, 202},
       {181, 180, 183, 182, 177, 176, 179, 178, 189, 188, 191, 190, 185, 184, 187, 186},
       {165, 164, 167, 166, 161, 160, 163, 162, 173, 172, 175, 174, 169, 168, 171, 170},
       {149, 148, 151, 150, 145, 144, 147, 146, 157, 156, 159, 158, 153, 152, 155, 154},
       {133, 132, 135, 134, 129, 128, 131, 130, 141, 140, 143, 142, 137, 136, 139, 138}},
      {{0xfef5, 0xfcf7, 0xfaf1, 0xf8f3, 0xf6fd, 0xf4ff, 0xf2f9, 0xf0fb},
       {0xeee5, 0xece7, 0xeae1, 0xe8e3, 0xe6ed, 0xe4ef, 0xe2e9, 0xe0eb},
       {0xded5, 0xdcd7, 0xdad1, 0xd8d3, 0xd6dd, 0xd4df, 0xd2d9, 0xd0db},
       {0xcec5, 0xccc7, 0xcac1, 0xc8c3, 0xc6cd, 0xc4cf, 0xc2c9, 0xc0cb},
       {0xbeb5, 0xbcb7, 0xbab1, 0xb8b3, 0xb6bd, 0xb4bf, 0xb2b9, 0xb0bb},
       {0xaea5, 0xaca7, 0xaaa1, 0xa8a3, 0xa6ad, 0xa4af, 0xa2a9, 0xa0ab},
       {0x9e95, 0x9c97, 0x9a91, 0x9893, 0x969d, 0x949f, 0x9299, 0x909b},
       {0x8e85, 0x8c87, 0x8a81, 0x8883, 0x868d, 0x848f, 0x8289, 0x808b}},
      {{0xfcfd'fef5, 0xf8f9'faf1, 0xf4f5'f6fd, 0xf0f1'f2f9},
       {0xeced'eee5, 0xe8e9'eae1, 0xe4e5'e6ed, 0xe0e1'e2e9},
       {0xdcdd'ded5, 0xd8d9'dad1, 0xd4d5'd6dd, 0xd0d1'd2d9},
       {0xcccd'cec5, 0xc8c9'cac1, 0xc4c5'c6cd, 0xc0c1'c2c9},
       {0xbcbd'beb5, 0xb8b9'bab1, 0xb4b5'b6bd, 0xb0b1'b2b9},
       {0xacad'aea5, 0xa8a9'aaa1, 0xa4a5'a6ad, 0xa0a1'a2a9},
       {0x9c9d'9e95, 0x9899'9a91, 0x9495'969d, 0x9091'9299},
       {0x8c8d'8e85, 0x8889'8a81, 0x8485'868d, 0x8081'8289}},
      {{0xf8f9'fafb'fcfd'fef5, 0xf0f1'f2f3'f4f5'f6fd},
       {0xe8e9'eaeb'eced'eee5, 0xe0e1'e2e3'e4e5'e6ed},
       {0xd8d9'dadb'dcdd'ded5, 0xd0d1'd2d3'd4d5'd6dd},
       {0xc8c9'cacb'cccd'cec5, 0xc0c1'c2c3'c4c5'c6cd},
       {0xb8b9'babb'bcbd'beb5, 0xb0b1'b2b3'b4b5'b6bd},
       {0xa8a9'aaab'acad'aea5, 0xa0a1'a2a3'a4a5'a6ad},
       {0x9899'9a9b'9c9d'9e95, 0x9091'9293'9495'969d},
       {0x8889'8a8b'8c8d'8e85, 0x8081'8283'8485'868d}},
      kVectorCalculationsSource);
}
TEST_F(Riscv64InterpreterTest, TestVmseq) {
  TestVectorInstruction(0x610c0457,  // Vmseq.vv v8, v16, v24, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x6100c457,  // Vmseq.vx v8, v16, x1, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x610ab457,  // Vmseq.vi  v8, v16, -0xb, v0.t
                        {{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsne) {
  TestVectorInstruction(0x650c0457,  // Vmsne.vv v8, v16, v24, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x6500c457,  // Vmsne.vx v8, v16, x1, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x650ab457,  // Vmsne.vi  v8, v16, -0xb, v0.t
                        {{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsltu) {
  TestVectorInstruction(0x690c0457,  // Vmsltu.vv v8, v16, v24, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x6900c457,  // Vmsltu.vx v8, v16, x1, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
}
TEST_F(Riscv64InterpreterTest, TestVmslt) {
  TestVectorInstruction(0x6d0c0457,  // vmslt.vv v8, v16, v24, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x6d00c457,  // Vmslt.vx v8, v16, x1, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
}
TEST_F(Riscv64InterpreterTest, TestVmsleu) {
  TestVectorInstruction(0x710c0457,  // Vmsleu.vv v8, v16, v24, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x7100c457,  // Vmsleu.vx v8, v16, x1, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x710ab457,  // Vmsleu.vi  v8, v16, -0xb, v0.t
                        {{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
}
TEST_F(Riscv64InterpreterTest, TestVmsle) {
  TestVectorInstruction(0x750c0457,  // Vmsle.vv v8, v16, v24, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x7500c457,  // Vmsle.vx v8, v16, x1, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x750ab457,  // Vmsle.vi  v8, v16, -0xb, v0.t
                        {{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
}
TEST_F(Riscv64InterpreterTest, TestVmsgtu) {
  TestVectorInstruction(0x7900c457,  // Vmsgtu.vx v8, v16, x1, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x790ab457,  // Vmsgtu.vi  v8, v16, -0xb, v0.t
                        {{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
}
TEST_F(Riscv64InterpreterTest, TestVmsgt) {
  TestVectorInstruction(0x7d00c457,  // Vmsgt.vx v8, v16, x1, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x7d0ab457,  // Vmsgt.vi  v8, v16, -0xb, v0.t
                        {{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVsll) {
  TestVectorInstruction(
      0x950c0457,  // Vsll.vv v8, v16, v24, v0.t
      {{0, 4, 32, 192, 8, 20, 96, 192, 16, 36, 160, 192, 12, 52, 224, 192},
       {16, 68, 32, 192, 40, 84, 96, 192, 48, 100, 160, 192, 28, 116, 224, 192},
       {32, 132, 32, 192, 72, 148, 96, 192, 80, 164, 160, 192, 44, 180, 224, 192},
       {48, 196, 32, 192, 104, 212, 96, 192, 112, 228, 160, 192, 60, 244, 224, 192},
       {64, 4, 32, 192, 136, 20, 96, 192, 144, 36, 160, 192, 76, 52, 224, 192},
       {80, 68, 32, 192, 168, 84, 96, 192, 176, 100, 160, 192, 92, 116, 224, 192},
       {96, 132, 32, 192, 200, 148, 96, 192, 208, 164, 160, 192, 108, 180, 224, 192},
       {112, 196, 32, 192, 232, 212, 96, 192, 240, 228, 160, 192, 124, 244, 224, 192}},
      {{0x0100, 0x3020, 0x0800, 0x6000, 0x1210, 0xb0a0, 0x0c00, 0xe000},
       {0x1110, 0x3120, 0x2800, 0x6000, 0x3230, 0xb1a0, 0x1c00, 0xe000},
       {0x2120, 0x3220, 0x4800, 0x6000, 0x5250, 0xb2a0, 0x2c00, 0xe000},
       {0x3130, 0x3320, 0x6800, 0x6000, 0x7270, 0xb3a0, 0x3c00, 0xe000},
       {0x4140, 0x3420, 0x8800, 0x6000, 0x9290, 0xb4a0, 0x4c00, 0xe000},
       {0x5150, 0x3520, 0xa800, 0x6000, 0xb2b0, 0xb5a0, 0x5c00, 0xe000},
       {0x6160, 0x3620, 0xc800, 0x6000, 0xd2d0, 0xb6a0, 0x6c00, 0xe000},
       {0x7170, 0x3720, 0xe800, 0x6000, 0xf2f0, 0xb7a0, 0x7c00, 0xe000}},
      {{0x0302'0100, 0x0c0a'0800, 0x1210'0000, 0x0c00'0000},
       {0x1312'1110, 0x2c2a'2800, 0x3230'0000, 0x1c00'0000},
       {0x2322'2120, 0x4c4a'4800, 0x5250'0000, 0x2c00'0000},
       {0x3332'3130, 0x6c6a'6800, 0x7270'0000, 0x3c00'0000},
       {0x4342'4140, 0x8c8a'8800, 0x9290'0000, 0x4c00'0000},
       {0x5352'5150, 0xacaa'a800, 0xb2b0'0000, 0x5c00'0000},
       {0x6362'6160, 0xccca'c800, 0xd2d0'0000, 0x6c00'0000},
       {0x7372'7170, 0xecea'e800, 0xf2f0'0000, 0x7c00'0000}},
      {{0x0706'0504'0302'0100, 0x1a18'1614'1210'0000},
       {0x1312'1110'0000'0000, 0x3230'0000'0000'0000},
       {0x2726'2524'2322'2120, 0x5a58'5654'5250'0000},
       {0x3332'3130'0000'0000, 0x7270'0000'0000'0000},
       {0x4746'4544'4342'4140, 0x9a98'9694'9290'0000},
       {0x5352'5150'0000'0000, 0xb2b0'0000'0000'0000},
       {0x6766'6564'6362'6160, 0xdad8'd6d4'd2d0'0000},
       {0x7372'7170'0000'0000, 0xf2f0'0000'0000'0000}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x9500c457,  // Vsll.vx v8, v16, x1, v0.t
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188},
       {192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252},
       {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188},
       {192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252}},
      {{0x0000, 0x0800, 0x1000, 0x1800, 0x2000, 0x2800, 0x3000, 0x3800},
       {0x4000, 0x4800, 0x5000, 0x5800, 0x6000, 0x6800, 0x7000, 0x7800},
       {0x8000, 0x8800, 0x9000, 0x9800, 0xa000, 0xa800, 0xb000, 0xb800},
       {0xc000, 0xc800, 0xd000, 0xd800, 0xe000, 0xe800, 0xf000, 0xf800},
       {0x0000, 0x0800, 0x1000, 0x1800, 0x2000, 0x2800, 0x3000, 0x3800},
       {0x4000, 0x4800, 0x5000, 0x5800, 0x6000, 0x6800, 0x7000, 0x7800},
       {0x8000, 0x8800, 0x9000, 0x9800, 0xa000, 0xa800, 0xb000, 0xb800},
       {0xc000, 0xc800, 0xd000, 0xd800, 0xe000, 0xe800, 0xf000, 0xf800}},
      {{0x0804'0000, 0x1814'1000, 0x2824'2000, 0x3834'3000},
       {0x4844'4000, 0x5854'5000, 0x6864'6000, 0x7874'7000},
       {0x8884'8000, 0x9894'9000, 0xa8a4'a000, 0xb8b4'b000},
       {0xc8c4'c000, 0xd8d4'd000, 0xe8e4'e000, 0xf8f4'f000},
       {0x0905'0000, 0x1915'1000, 0x2925'2000, 0x3935'3000},
       {0x4945'4000, 0x5955'5000, 0x6965'6000, 0x7975'7000},
       {0x8985'8000, 0x9995'9000, 0xa9a5'a000, 0xb9b5'b000},
       {0xc9c5'c000, 0xd9d5'd000, 0xe9e5'e000, 0xf9f5'f000}},
      {{0x0804'0000'0000'0000, 0x2824'2000'0000'0000},
       {0x4844'4000'0000'0000, 0x6864'6000'0000'0000},
       {0x8884'8000'0000'0000, 0xa8a4'a000'0000'0000},
       {0xc8c4'c000'0000'0000, 0xe8e4'e000'0000'0000},
       {0x0905'0000'0000'0000, 0x2925'2000'0000'0000},
       {0x4945'4000'0000'0000, 0x6965'6000'0000'0000},
       {0x8985'8000'0000'0000, 0xa9a5'a000'0000'0000},
       {0xc9c5'c000'0000'0000, 0xe9e5'e000'0000'0000}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x9505b457,  // Vsll.vi v8, v16, 0xb, v0.t
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248},
       {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248},
       {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248},
       {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248}},
      {{0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000},
       {0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000},
       {0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000},
       {0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000}},
      {{0x1008'0000, 0x3028'2000, 0x5048'4000, 0x7068'6000},
       {0x9088'8000, 0xb0a8'a000, 0xd0c8'c000, 0xf0e8'e000},
       {0x1109'0000, 0x3129'2000, 0x5149'4000, 0x7169'6000},
       {0x9189'8000, 0xb1a9'a000, 0xd1c9'c000, 0xf1e9'e000},
       {0x120a'0000, 0x322a'2000, 0x524a'4000, 0x726a'6000},
       {0x928a'8000, 0xb2aa'a000, 0xd2ca'c000, 0xf2ea'e000},
       {0x130b'0000, 0x332b'2000, 0x534b'4000, 0x736b'6000},
       {0x938b'8000, 0xb3ab'a000, 0xd3cb'c000, 0xf3eb'e000}},
      {{0x3028'2018'1008'0000, 0x7068'6058'5048'4000},
       {0xb0a8'a098'9088'8000, 0xf0e8'e0d8'd0c8'c000},
       {0x3129'2119'1109'0000, 0x7169'6159'5149'4000},
       {0xb1a9'a199'9189'8000, 0xf1e9'e1d9'd1c9'c000},
       {0x322a'221a'120a'0000, 0x726a'625a'524a'4000},
       {0xb2aa'a29a'928a'8000, 0xf2ea'e2da'd2ca'c000},
       {0x332b'231b'130b'0000, 0x736b'635b'534b'4000},
       {0xb3ab'a39b'938b'8000, 0xf3eb'e3db'd3cb'c000}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVsrl) {
  TestVectorInstruction(0xa10c0457,  // Vsrl.vv v8, v16, v24, v0.t
                        {{7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1},
                         {85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85},
                         {93, 1, 93, 1, 93, 1, 93, 1, 85, 1, 85, 1, 85, 1, 85, 1},
                         {170, 42, 10, 2, 85, 42, 10, 2, 8, 4, 1, 0, 17, 4, 1, 0},
                         {244, 63, 15, 3, 122, 63, 15, 3, 123, 63, 15, 3, 246, 63, 15, 3},
                         {244, 63, 15, 3, 124, 63, 15, 3, 122, 63, 15, 3, 245, 63, 15, 3},
                         {187, 46, 11, 2, 93, 46, 11, 2, 93, 46, 11, 2, 187, 46, 11, 2},
                         {169, 42, 10, 2, 84, 42, 10, 2, 84, 42, 10, 2, 169, 42, 10, 2}},
                        {{0x07ff, 0x07ff, 0x07ff, 0x07ff, 0x07ff, 0x07ff, 0x07ff, 0x07ff},
                         {0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555},
                         {0x5ddd, 0x5ddd, 0x5ddd, 0x5ddd, 0x5555, 0x5555, 0x5555, 0x5555},
                         {0xaaaa, 0x0aaa, 0x0055, 0x000a, 0x0888, 0x0111, 0x0011, 0x0001},
                         {0xfff4, 0x0fff, 0x007f, 0x000f, 0x7ffb, 0x0fff, 0x00ff, 0x000f},
                         {0xfff4, 0x0fff, 0x007f, 0x000f, 0x7ffa, 0x0fff, 0x00ff, 0x000f},
                         {0xbbbb, 0x0a9b, 0x005d, 0x000a, 0x5ddd, 0x0a9b, 0x00bb, 0x000a},
                         {0xa9a9, 0x0a9a, 0x0054, 0x000a, 0x54d4, 0x0a9a, 0x00a9, 0x000a}},
                        {{0x0000'07ff, 0x0000'07ff, 0x0000'07ff, 0x0000'07ff},
                         {0x0000'5555, 0x0000'5555, 0x0000'5555, 0x0000'5555},
                         {0x0000'5ddd, 0x0000'5ddd, 0x0000'5555, 0x0000'5555},
                         {0xaaaa'aaaa, 0x0055'5555, 0x0000'0888, 0x0000'0011},
                         {0xfff4'fff4, 0x007f'fa7f, 0x0000'7ffb, 0x0000'00ff},
                         {0xfff4'fff4, 0x007f'fc7f, 0x0000'7ffa, 0x0000'00ff},
                         {0xa9bb'bbbb, 0x0054'dddd, 0x0000'54dd, 0x0000'00a9},
                         {0xa9a9'a9a9, 0x0054'd4d4, 0x0000'54d4, 0x0000'00a9}},
                        {{0x0000'0000'0000'07ff, 0x0000'0000'0000'07ff},
                         {0x0000'5555'5555'5555, 0x0000'5555'5555'5555},
                         {0x0000'0000'0000'5ddd, 0x0000'0000'0000'5555},
                         {0x0000'0000'aaaa'aaaa, 0x0000'0000'0000'0888},
                         {0xfff4'fff4'fff4'fff4, 0x0000'7ffb'7ffb'7ffb},
                         {0x0000'0000'fff8'fff8, 0x0000'0000'0000'7ffa},
                         {0xa9bb'bbbb'a9bb'bbbb, 0x0000'54dd'dddd'd4dd},
                         {0x0000'0000'a9a9'a9a9, 0x0000'0000'0000'54d4}},
                        kVectorRightShiftSource);
  TestVectorInstruction(0xa100c457,  // Vsrl.vx v8, v16, x1, v0.t
                        {{61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63},
                         {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                         {46, 46, 46, 46, 46, 46, 46, 46, 42, 42, 42, 42, 42, 42, 42, 42},
                         {42, 42, 42, 42, 42, 42, 42, 42, 4, 4, 4, 4, 4, 4, 4, 4},
                         {61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63},
                         {61, 63, 61, 63, 62, 63, 62, 63, 61, 63, 61, 63, 61, 63, 61, 63},
                         {46, 46, 46, 42, 46, 46, 46, 42, 46, 46, 46, 42, 46, 46, 46, 42},
                         {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42}},
                        {{0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f},
                         {0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a},
                         {0x002e, 0x002e, 0x002e, 0x002e, 0x002a, 0x002a, 0x002a, 0x002a},
                         {0x002a, 0x002a, 0x002a, 0x002a, 0x0004, 0x0004, 0x0004, 0x0004},
                         {0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f},
                         {0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f},
                         {0x002e, 0x002a, 0x002e, 0x002a, 0x002e, 0x002a, 0x002e, 0x002a},
                         {0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a}},
                        {{0x003f'fd7f, 0x003f'fd7f, 0x003f'fd7f, 0x003f'fd7f},
                         {0x002a'aaaa, 0x002a'aaaa, 0x002a'aaaa, 0x002a'aaaa},
                         {0x002e'eeee, 0x002e'eeee, 0x002a'aaaa, 0x002a'aaaa},
                         {0x002a'aaaa, 0x002a'aaaa, 0x0004'4444, 0x0004'4444},
                         {0x003f'fd3f, 0x003f'fd3f, 0x003f'fdbf, 0x003f'fdbf},
                         {0x003f'fd3f, 0x003f'fe3f, 0x003f'fd7f, 0x003f'fd7f},
                         {0x002a'6eee, 0x002a'6eee, 0x002a'6eee, 0x002a'6eee},
                         {0x002a'6a6a, 0x002a'6a6a, 0x002a'6a6a, 0x002a'6a6a}},
                        {{0x0000'0000'003f'fd7f, 0x0000'0000'003f'fd7f},
                         {0x0000'0000'002a'aaaa, 0x0000'0000'002a'aaaa},
                         {0x0000'0000'002e'eeee, 0x0000'0000'002a'aaaa},
                         {0x0000'0000'002a'aaaa, 0x0000'0000'0004'4444},
                         {0x0000'0000'003f'fd3f, 0x0000'0000'003f'fdbf},
                         {0x0000'0000'003f'fe3f, 0x0000'0000'003f'fd7f},
                         {0x0000'0000'002a'6eee, 0x0000'0000'002a'6eee},
                         {0x0000'0000'002a'6a6a, 0x0000'0000'002a'6a6a}},
                        kVectorRightShiftSource);
  TestVectorInstruction(0xa101b457,  // Vsrl.vi v8, v16, 0x3, v0.t
                        {{30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31},
                         {21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21},
                         {23, 23, 23, 23, 23, 23, 23, 23, 21, 21, 21, 21, 21, 21, 21, 21},
                         {21, 21, 21, 21, 21, 21, 21, 21, 2, 2, 2, 2, 2, 2, 2, 2},
                         {30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31},
                         {30, 31, 30, 31, 31, 31, 31, 31, 30, 31, 30, 31, 30, 31, 30, 31},
                         {23, 23, 23, 21, 23, 23, 23, 21, 23, 23, 23, 21, 23, 23, 23, 21},
                         {21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21}},
                        {{0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe},
                         {0x1555, 0x1555, 0x1555, 0x1555, 0x1555, 0x1555, 0x1555, 0x1555},
                         {0x1777, 0x1777, 0x1777, 0x1777, 0x1555, 0x1555, 0x1555, 0x1555},
                         {0x1555, 0x1555, 0x1555, 0x1555, 0x0222, 0x0222, 0x0222, 0x0222},
                         {0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe},
                         {0x1ffe, 0x1ffe, 0x1fff, 0x1fff, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe},
                         {0x1777, 0x1537, 0x1777, 0x1537, 0x1777, 0x1537, 0x1777, 0x1537},
                         {0x1535, 0x1535, 0x1535, 0x1535, 0x1535, 0x1535, 0x1535, 0x1535}},
                        {{0x1ffe'bffe, 0x1ffe'bffe, 0x1ffe'bffe, 0x1ffe'bffe},
                         {0x1555'5555, 0x1555'5555, 0x1555'5555, 0x1555'5555},
                         {0x1777'7777, 0x1777'7777, 0x1555'5555, 0x1555'5555},
                         {0x1555'5555, 0x1555'5555, 0x0222'2222, 0x0222'2222},
                         {0x1ffe'9ffe, 0x1ffe'9ffe, 0x1ffe'dffe, 0x1ffe'dffe},
                         {0x1ffe'9ffe, 0x1fff'1fff, 0x1ffe'bffe, 0x1ffe'bffe},
                         {0x1537'7777, 0x1537'7777, 0x1537'7777, 0x1537'7777},
                         {0x1535'3535, 0x1535'3535, 0x1535'3535, 0x1535'3535}},
                        {{0x1ffe'bffe'bffe'bffe, 0x1ffe'bffe'bffe'bffe},
                         {0x1555'5555'5555'5555, 0x1555'5555'5555'5555},
                         {0x1777'7777'7777'7777, 0x1555'5555'5555'5555},
                         {0x1555'5555'5555'5555, 0x0222'2222'2222'2222},
                         {0x1ffe'9ffe'9ffe'9ffe, 0x1ffe'dffe'dffe'dffe},
                         {0x1fff'1fff'1ffe'9ffe, 0x1ffe'bffe'bffe'bffe},
                         {0x1537'7777'7537'7777, 0x1537'7777'7537'7777},
                         {0x1535'3535'3535'3535, 0x1535'3535'3535'3535}},
                        kVectorRightShiftSource);
}

TEST_F(Riscv64InterpreterTest, TestVsra) {
  TestVectorInstruction(
      0xa50c0457,  // Vsra.vv v8, v16, v24, v0.t
      {{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213},
       {221, 255, 221, 255, 221, 255, 221, 255, 213, 255, 213, 255, 213, 255, 213, 255},
       {170, 234, 250, 254, 213, 234, 250, 254, 8, 4, 1, 0, 17, 4, 1, 0},
       {244, 255, 255, 255, 250, 255, 255, 255, 251, 255, 255, 255, 246, 255, 255, 255},
       {244, 255, 255, 255, 252, 255, 255, 255, 250, 255, 255, 255, 245, 255, 255, 255},
       {187, 238, 251, 254, 221, 238, 251, 254, 221, 238, 251, 254, 187, 238, 251, 254},
       {169, 234, 250, 254, 212, 234, 250, 254, 212, 234, 250, 254, 169, 234, 250, 254}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xd555, 0xd555, 0xd555, 0xd555, 0xd555, 0xd555, 0xd555, 0xd555},
       {0xdddd, 0xdddd, 0xdddd, 0xdddd, 0xd555, 0xd555, 0xd555, 0xd555},
       {0xaaaa, 0xfaaa, 0xffd5, 0xfffa, 0x0888, 0x0111, 0x0011, 0x0001},
       {0xfff4, 0xffff, 0xffff, 0xffff, 0xfffb, 0xffff, 0xffff, 0xffff},
       {0xfff4, 0xffff, 0xffff, 0xffff, 0xfffa, 0xffff, 0xffff, 0xffff},
       {0xbbbb, 0xfa9b, 0xffdd, 0xfffa, 0xdddd, 0xfa9b, 0xffbb, 0xfffa},
       {0xa9a9, 0xfa9a, 0xffd4, 0xfffa, 0xd4d4, 0xfa9a, 0xffa9, 0xfffa}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'd555, 0xffff'd555, 0xffff'd555, 0xffff'd555},
       {0xffff'dddd, 0xffff'dddd, 0xffff'd555, 0xffff'd555},
       {0xaaaa'aaaa, 0xffd5'5555, 0x0000'0888, 0x0000'0011},
       {0xfff4'fff4, 0xffff'fa7f, 0xffff'fffb, 0xffff'ffff},
       {0xfff4'fff4, 0xffff'fc7f, 0xffff'fffa, 0xffff'ffff},
       {0xa9bb'bbbb, 0xffd4'dddd, 0xffff'd4dd, 0xffff'ffa9},
       {0xa9a9'a9a9, 0xffd4'd4d4, 0xffff'd4d4, 0xffff'ffa9}},
      {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'd555'5555'5555, 0xffff'd555'5555'5555},
       {0xffff'ffff'ffff'dddd, 0xffff'ffff'ffff'd555},
       {0xffff'ffff'aaaa'aaaa, 0x0000'0000'0000'0888},
       {0xfff4'fff4'fff4'fff4, 0xffff'fffb'7ffb'7ffb},
       {0xffff'ffff'fff8'fff8, 0xffff'ffff'ffff'fffa},
       {0xa9bb'bbbb'a9bb'bbbb, 0xffff'd4dd'dddd'd4dd},
       {0xffff'ffff'a9a9'a9a9, 0xffff'ffff'ffff'd4d4}},
      kVectorRightShiftSource);
  TestVectorInstruction(
      0xa500c457,  // Vsra.vx v8, v16, x1, v0.t
      {{253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255},
       {234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234},
       {238, 238, 238, 238, 238, 238, 238, 238, 234, 234, 234, 234, 234, 234, 234, 234},
       {234, 234, 234, 234, 234, 234, 234, 234, 4, 4, 4, 4, 4, 4, 4, 4},
       {253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255},
       {253, 255, 253, 255, 254, 255, 254, 255, 253, 255, 253, 255, 253, 255, 253, 255},
       {238, 238, 238, 234, 238, 238, 238, 234, 238, 238, 238, 234, 238, 238, 238, 234},
       {234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea},
       {0xffee, 0xffee, 0xffee, 0xffee, 0xffea, 0xffea, 0xffea, 0xffea},
       {0xffea, 0xffea, 0xffea, 0xffea, 0x0004, 0x0004, 0x0004, 0x0004},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffee, 0xffea, 0xffee, 0xffea, 0xffee, 0xffea, 0xffee, 0xffea},
       {0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea}},
      {{0xffff'fd7f, 0xffff'fd7f, 0xffff'fd7f, 0xffff'fd7f},
       {0xffea'aaaa, 0xffea'aaaa, 0xffea'aaaa, 0xffea'aaaa},
       {0xffee'eeee, 0xffee'eeee, 0xffea'aaaa, 0xffea'aaaa},
       {0xffea'aaaa, 0xffea'aaaa, 0x0004'4444, 0x0004'4444},
       {0xffff'fd3f, 0xffff'fd3f, 0xffff'fdbf, 0xffff'fdbf},
       {0xffff'fd3f, 0xffff'fe3f, 0xffff'fd7f, 0xffff'fd7f},
       {0xffea'6eee, 0xffea'6eee, 0xffea'6eee, 0xffea'6eee},
       {0xffea'6a6a, 0xffea'6a6a, 0xffea'6a6a, 0xffea'6a6a}},
      {{0xffff'ffff'ffff'fd7f, 0xffff'ffff'ffff'fd7f},
       {0xffff'ffff'ffea'aaaa, 0xffff'ffff'ffea'aaaa},
       {0xffff'ffff'ffee'eeee, 0xffff'ffff'ffea'aaaa},
       {0xffff'ffff'ffea'aaaa, 0x0000'0000'0004'4444},
       {0xffff'ffff'ffff'fd3f, 0xffff'ffff'ffff'fdbf},
       {0xffff'ffff'ffff'fe3f, 0xffff'ffff'ffff'fd7f},
       {0xffff'ffff'ffea'6eee, 0xffff'ffff'ffea'6eee},
       {0xffff'ffff'ffea'6a6a, 0xffff'ffff'ffea'6a6a}},
      kVectorRightShiftSource);
  TestVectorInstruction(
      0xa501b457,  // Vsra.vi v8, v16, 0x3, v0.t
      {{254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {247, 247, 247, 247, 247, 247, 247, 247, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 2, 2, 2, 2, 2, 2, 2, 2},
       {254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255},
       {254, 255, 254, 255, 255, 255, 255, 255, 254, 255, 254, 255, 254, 255, 254, 255},
       {247, 247, 247, 245, 247, 247, 247, 245, 247, 247, 247, 245, 247, 247, 247, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245}},
      {{0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe},
       {0xf555, 0xf555, 0xf555, 0xf555, 0xf555, 0xf555, 0xf555, 0xf555},
       {0xf777, 0xf777, 0xf777, 0xf777, 0xf555, 0xf555, 0xf555, 0xf555},
       {0xf555, 0xf555, 0xf555, 0xf555, 0x0222, 0x0222, 0x0222, 0x0222},
       {0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe},
       {0xfffe, 0xfffe, 0xffff, 0xffff, 0xfffe, 0xfffe, 0xfffe, 0xfffe},
       {0xf777, 0xf537, 0xf777, 0xf537, 0xf777, 0xf537, 0xf777, 0xf537},
       {0xf535, 0xf535, 0xf535, 0xf535, 0xf535, 0xf535, 0xf535, 0xf535}},
      {{0xfffe'bffe, 0xfffe'bffe, 0xfffe'bffe, 0xfffe'bffe},
       {0xf555'5555, 0xf555'5555, 0xf555'5555, 0xf555'5555},
       {0xf777'7777, 0xf777'7777, 0xf555'5555, 0xf555'5555},
       {0xf555'5555, 0xf555'5555, 0x0222'2222, 0x0222'2222},
       {0xfffe'9ffe, 0xfffe'9ffe, 0xfffe'dffe, 0xfffe'dffe},
       {0xfffe'9ffe, 0xffff'1fff, 0xfffe'bffe, 0xfffe'bffe},
       {0xf537'7777, 0xf537'7777, 0xf537'7777, 0xf537'7777},
       {0xf535'3535, 0xf535'3535, 0xf535'3535, 0xf535'3535}},
      {{0xfffe'bffe'bffe'bffe, 0xfffe'bffe'bffe'bffe},
       {0xf555'5555'5555'5555, 0xf555'5555'5555'5555},
       {0xf777'7777'7777'7777, 0xf555'5555'5555'5555},
       {0xf555'5555'5555'5555, 0x0222'2222'2222'2222},
       {0xfffe'9ffe'9ffe'9ffe, 0xfffe'dffe'dffe'dffe},
       {0xffff'1fff'1ffe'9ffe, 0xfffe'bffe'bffe'bffe},
       {0xf537'7777'7537'7777, 0xf537'7777'7537'7777},
       {0xf535'3535'3535'3535, 0xf535'3535'3535'3535}},
      kVectorRightShiftSource);
}

TEST_F(Riscv64InterpreterTest, TestVmacc) {
  TestVectorInstruction(0xb5882457,  // vmacc.vv v8, v16, v24, v0.t
                        {{85, 87, 93, 103, 121, 135, 157, 183, 221, 247, 29, 71, 117, 167, 221, 23},
                         {85, 151, 221, 39, 137, 199, 29, 119, 237, 55, 157, 7, 117, 231, 93, 215},
                         {85, 215, 93, 231, 153, 7, 157, 55, 253, 119, 29, 199, 117, 39, 221, 151},
                         {85, 23, 221, 167, 169, 71, 29, 247, 13, 183, 157, 135, 117, 103, 93, 87},
                         {85, 87, 93, 103, 185, 135, 157, 183, 29, 247, 29, 71, 117, 167, 221, 23},
                         {85, 151, 221, 39, 201, 199, 29, 119, 45, 55, 157, 7, 117, 231, 93, 215},
                         {85, 215, 93, 231, 217, 7, 157, 55, 61, 119, 29, 199, 117, 39, 221, 151},
                         {85, 23, 221, 167, 233, 71, 29, 247, 77, 183, 157, 135, 117, 103, 93, 87}},
                        {{0x5555, 0x6d5d, 0xaa79, 0xfd9d, 0x7edd, 0x0e1d, 0xc675, 0x9edd},
                         {0x9755, 0xafdd, 0xfd89, 0x411d, 0xd2ed, 0x529d, 0x0b75, 0xe45d},
                         {0xdd55, 0xf65d, 0x5499, 0x889d, 0x2afd, 0x9b1d, 0x5475, 0x2ddd},
                         {0x2755, 0x40dd, 0xafa9, 0xd41d, 0x870d, 0xe79d, 0xa175, 0x7b5d},
                         {0x7555, 0x8f5d, 0x0eb9, 0x239d, 0xe71d, 0x381d, 0xf275, 0xccdd},
                         {0xc755, 0xe1dd, 0x71c9, 0x771d, 0x4b2d, 0x8c9d, 0x4775, 0x225d},
                         {0x1d55, 0x385d, 0xd8d9, 0xce9d, 0xb33d, 0xe51d, 0xa075, 0x7bdd},
                         {0x7755, 0x92dd, 0x43e9, 0x2a1d, 0x1f4d, 0x419d, 0xfd75, 0xd95d}},
                        {{0x5d57'5555, 0x44ed'aa79, 0x2a42'7edd, 0x0149'c675},
                         {0xe41b'9755, 0xdec3'fd89, 0xc71a'd2ed, 0x9114'0b75},
                         {0x76e7'dd55, 0x84a2'5499, 0x6ffb'2afd, 0x2ce6'5475},
                         {0x15bc'2755, 0x3688'afa9, 0x24e3'870d, 0xd4c0'a175},
                         {0xc098'7555, 0xf477'0eb9, 0xe5d3'e71d, 0x88a2'f275},
                         {0x777c'c755, 0xbe6d'71c9, 0xb2cc'4b2d, 0x488d'4775},
                         {0x3a69'1d55, 0x946b'd8d9, 0x8bcc'b33d, 0x147f'a075},
                         {0x095d'7755, 0x7672'43e9, 0x70d5'1f4d, 0xec79'fd75}},
                        {{0xc89d'7e69'5d57'5555, 0x5ace'6e38'2a42'7edd},
                         {0xebfd'5b02'e41b'9755, 0x8c3a'54d9'c71a'd2ed},
                         {0x2b75'4bac'76e7'dd55, 0xd9be'4f8b'6ffb'2afd},
                         {0x8705'5066'15bc'2755, 0x435a'5e4d'24e3'870d},
                         {0xfead'692f'c098'7555, 0xc90e'811e'e5d3'e71d},
                         {0x926d'9609'777c'c755, 0x6ada'b800'b2cc'4b2d},
                         {0x4245'd6f3'3a69'1d55, 0x28bf'02f2'8bcc'b33d},
                         {0x0e36'2bed'095d'7755, 0x02bb'61f4'70d5'1f4d}},
                        kVectorCalculationsSource);
  TestVectorInstruction(
      0xb500e457,  // vmacc.vx v8, x1, v16, v0.t
      {{85, 255, 169, 83, 253, 167, 81, 251, 165, 79, 249, 163, 77, 247, 161, 75},
       {245, 159, 73, 243, 157, 71, 241, 155, 69, 239, 153, 67, 237, 151, 65, 235},
       {149, 63, 233, 147, 61, 231, 145, 59, 229, 143, 57, 227, 141, 55, 225, 139},
       {53, 223, 137, 51, 221, 135, 49, 219, 133, 47, 217, 131, 45, 215, 129, 43},
       {213, 127, 41, 211, 125, 39, 209, 123, 37, 207, 121, 35, 205, 119, 33, 203},
       {117, 31, 201, 115, 29, 199, 113, 27, 197, 111, 25, 195, 109, 23, 193, 107},
       {21, 191, 105, 19, 189, 103, 17, 187, 101, 15, 185, 99, 13, 183, 97, 11},
       {181, 95, 9, 179, 93, 7, 177, 91, 5, 175, 89, 3, 173, 87, 1, 171}},
      {{0xff55, 0xa8a9, 0x51fd, 0xfb51, 0xa4a5, 0x4df9, 0xf74d, 0xa0a1},
       {0x49f5, 0xf349, 0x9c9d, 0x45f1, 0xef45, 0x9899, 0x41ed, 0xeb41},
       {0x9495, 0x3de9, 0xe73d, 0x9091, 0x39e5, 0xe339, 0x8c8d, 0x35e1},
       {0xdf35, 0x8889, 0x31dd, 0xdb31, 0x8485, 0x2dd9, 0xd72d, 0x8081},
       {0x29d5, 0xd329, 0x7c7d, 0x25d1, 0xcf25, 0x7879, 0x21cd, 0xcb21},
       {0x7475, 0x1dc9, 0xc71d, 0x7071, 0x19c5, 0xc319, 0x6c6d, 0x15c1},
       {0xbf15, 0x6869, 0x11bd, 0xbb11, 0x6465, 0x0db9, 0xb70d, 0x6061},
       {0x09b5, 0xb309, 0x5c5d, 0x05b1, 0xaf05, 0x5859, 0x01ad, 0xab01}},
      {{0x5353'ff55, 0xfb51'51fd, 0xa34e'a4a5, 0x4b4b'f74d},
       {0xf349'49f5, 0x9b46'9c9d, 0x4343'ef45, 0xeb41'41ed},
       {0x933e'9495, 0x3b3b'e73d, 0xe339'39e5, 0x8b36'8c8d},
       {0x3333'df35, 0xdb31'31dd, 0x832e'8485, 0x2b2b'd72d},
       {0xd329'29d5, 0x7b26'7c7d, 0x2323'cf25, 0xcb21'21cd},
       {0x731e'7475, 0x1b1b'c71d, 0xc319'19c5, 0x6b16'6c6d},
       {0x1313'bf15, 0xbb11'11bd, 0x630e'6465, 0x0b0b'b70d},
       {0xb309'09b5, 0x5b06'5c5d, 0x0303'af05, 0xab01'01ad}},
      {{0xfb51'51fd'5353'ff55, 0xa0a1'4ca2'a34e'a4a5},
       {0x45f1'4747'f349'49f5, 0xeb41'41ed'4343'ef45},
       {0x9091'3c92'933e'9495, 0x35e1'3737'e339'39e5},
       {0xdb31'31dd'3333'df35, 0x8081'2c82'832e'8485},
       {0x25d1'2727'd329'29d5, 0xcb21'21cd'2323'cf25},
       {0x7071'1c72'731e'7475, 0x15c1'1717'c319'19c5},
       {0xbb11'11bd'1313'bf15, 0x6061'0c62'630e'6465},
       {0x05b1'0707'b309'09b5, 0xab01'01ad'0303'af05}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVnmsac) {
  TestVectorInstruction(0xbd882457,  // vnmsac.vv v8, v16, v24, v0.t
                        {{85, 83, 77, 67, 49, 35, 13, 243, 205, 179, 141, 99, 53, 3, 205, 147},
                         {85, 19, 205, 131, 33, 227, 141, 51, 189, 115, 13, 163, 53, 195, 77, 211},
                         {85, 211, 77, 195, 17, 163, 13, 115, 173, 51, 141, 227, 53, 131, 205, 19},
                         {85, 147, 205, 3, 1, 99, 141, 179, 157, 243, 13, 35, 53, 67, 77, 83},
                         {85, 83, 77, 67, 241, 35, 13, 243, 141, 179, 141, 99, 53, 3, 205, 147},
                         {85, 19, 205, 131, 225, 227, 141, 51, 125, 115, 13, 163, 53, 195, 77, 211},
                         {85, 211, 77, 195, 209, 163, 13, 115, 109, 51, 141, 227, 53, 131, 205, 19},
                         {85, 147, 205, 3, 193, 99, 141, 179, 93, 243, 13, 35, 53, 67, 77, 83}},
                        {{0x5555, 0x3d4d, 0x0031, 0xad0d, 0x2bcd, 0x9c8d, 0xe435, 0x0bcd},
                         {0x1355, 0xfacd, 0xad21, 0x698d, 0xd7bd, 0x580d, 0x9f35, 0xc64d},
                         {0xcd55, 0xb44d, 0x5611, 0x220d, 0x7fad, 0x0f8d, 0x5635, 0x7ccd},
                         {0x8355, 0x69cd, 0xfb01, 0xd68d, 0x239d, 0xc30d, 0x0935, 0x2f4d},
                         {0x3555, 0x1b4d, 0x9bf1, 0x870d, 0xc38d, 0x728d, 0xb835, 0xddcd},
                         {0xe355, 0xc8cd, 0x38e1, 0x338d, 0x5f7d, 0x1e0d, 0x6335, 0x884d},
                         {0x8d55, 0x724d, 0xd1d1, 0xdc0d, 0xf76d, 0xc58d, 0x0a35, 0x2ecd},
                         {0x3355, 0x17cd, 0x66c1, 0x808d, 0x8b5d, 0x690d, 0xad35, 0xd14d}},
                        {{0x4d53'5555, 0x65bd'0031, 0x8068'2bcd, 0xa960'e435},
                         {0xc68f'1355, 0xcbe6'ad21, 0xe38f'd7bd, 0x1996'9f35},
                         {0x33c2'cd55, 0x2608'5611, 0x3aaf'7fad, 0x7dc4'5635},
                         {0x94ee'8355, 0x7421'fb01, 0x85c7'239d, 0xd5ea'0935},
                         {0xea12'3555, 0xb633'9bf1, 0xc4d6'c38d, 0x2207'b835},
                         {0x332d'e355, 0xec3d'38e1, 0xf7de'5f7d, 0x621d'6335},
                         {0x7041'8d55, 0x163e'd1d1, 0x1edd'f76d, 0x962b'0a35},
                         {0xa14d'3355, 0x3438'66c1, 0x39d5'8b5d, 0xbe30'ad35}},
                        {{0xe20d'2c41'4d53'5555, 0x4fdc'3c72'8068'2bcd},
                         {0xbead'4fa7'c68f'1355, 0x1e70'55d0'e38f'd7bd},
                         {0x7f35'5efe'33c2'cd55, 0xd0ec'5b1f'3aaf'7fad},
                         {0x23a5'5a44'94ee'8355, 0x6750'4c5d'85c7'239d},
                         {0xabfd'417a'ea12'3555, 0xe19c'298b'c4d6'c38d},
                         {0x183d'14a1'332d'e355, 0x3fcf'f2a9'f7de'5f7d},
                         {0x6864'd3b7'7041'8d55, 0x81eb'a7b8'1edd'f76d},
                         {0x9c74'7ebd'a14d'3355, 0xa7ef'48b6'39d5'8b5d}},
                        kVectorCalculationsSource);
  TestVectorInstruction(
      0xbd00e457,  // vnmsac.vx v8, x1, v16, v0.t
      {{85, 171, 1, 87, 173, 3, 89, 175, 5, 91, 177, 7, 93, 179, 9, 95},
       {181, 11, 97, 183, 13, 99, 185, 15, 101, 187, 17, 103, 189, 19, 105, 191},
       {21, 107, 193, 23, 109, 195, 25, 111, 197, 27, 113, 199, 29, 115, 201, 31},
       {117, 203, 33, 119, 205, 35, 121, 207, 37, 123, 209, 39, 125, 211, 41, 127},
       {213, 43, 129, 215, 45, 131, 217, 47, 133, 219, 49, 135, 221, 51, 137, 223},
       {53, 139, 225, 55, 141, 227, 57, 143, 229, 59, 145, 231, 61, 147, 233, 63},
       {149, 235, 65, 151, 237, 67, 153, 239, 69, 155, 241, 71, 157, 243, 73, 159},
       {245, 75, 161, 247, 77, 163, 249, 79, 165, 251, 81, 167, 253, 83, 169, 255}},
      {{0xab55, 0x0201, 0x58ad, 0xaf59, 0x0605, 0x5cb1, 0xb35d, 0x0a09},
       {0x60b5, 0xb761, 0x0e0d, 0x64b9, 0xbb65, 0x1211, 0x68bd, 0xbf69},
       {0x1615, 0x6cc1, 0xc36d, 0x1a19, 0x70c5, 0xc771, 0x1e1d, 0x74c9},
       {0xcb75, 0x2221, 0x78cd, 0xcf79, 0x2625, 0x7cd1, 0xd37d, 0x2a29},
       {0x80d5, 0xd781, 0x2e2d, 0x84d9, 0xdb85, 0x3231, 0x88dd, 0xdf89},
       {0x3635, 0x8ce1, 0xe38d, 0x3a39, 0x90e5, 0xe791, 0x3e3d, 0x94e9},
       {0xeb95, 0x4241, 0x98ed, 0xef99, 0x4645, 0x9cf1, 0xf39d, 0x4a49},
       {0xa0f5, 0xf7a1, 0x4e4d, 0xa4f9, 0xfba5, 0x5251, 0xa8fd, 0xffa9}},
      {{0x5756'ab55, 0xaf59'58ad, 0x075c'0605, 0x5f5e'b35d},
       {0xb761'60b5, 0x0f64'0e0d, 0x6766'bb65, 0xbf69'68bd},
       {0x176c'1615, 0x6f6e'c36d, 0xc771'70c5, 0x1f74'1e1d},
       {0x7776'cb75, 0xcf79'78cd, 0x277c'2625, 0x7f7e'd37d},
       {0xd781'80d5, 0x2f84'2e2d, 0x8786'db85, 0xdf89'88dd},
       {0x378c'3635, 0x8f8e'e38d, 0xe791'90e5, 0x3f94'3e3d},
       {0x9796'eb95, 0xef99'98ed, 0x479c'4645, 0x9f9e'f39d},
       {0xf7a1'a0f5, 0x4fa4'4e4d, 0xa7a6'fba5, 0xffa9'a8fd}},
      {{0xaf59'58ad'5756'ab55, 0x0a09'5e08'075c'0605},
       {0x64b9'6362'b761'60b5, 0xbf69'68bd'6766'bb65},
       {0x1a19'6e18'176c'1615, 0x74c9'7372'c771'70c5},
       {0xcf79'78cd'7776'cb75, 0x2a29'7e28'277c'2625},
       {0x84d9'8382'd781'80d5, 0xdf89'88dd'8786'db85},
       {0x3a39'8e38'378c'3635, 0x94e9'9392'e791'90e5},
       {0xef99'98ed'9796'eb95, 0x4a49'9e48'479c'4645},
       {0xa4f9'a3a2'f7a1'a0f5, 0xffa9'a8fd'a7a6'fba5}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVmadd) {
  TestVectorInstruction(
      0xa5882457,  // vmadd.vv v8, v16, v24, v0.t
      {{0, 87, 174, 5, 93, 179, 10, 97, 185, 15, 102, 189, 20, 107, 194, 25},
       {112, 199, 30, 117, 205, 35, 122, 209, 41, 127, 214, 45, 132, 219, 50, 137},
       {224, 55, 142, 229, 61, 147, 234, 65, 153, 239, 70, 157, 244, 75, 162, 249},
       {80, 167, 254, 85, 173, 3, 90, 177, 9, 95, 182, 13, 100, 187, 18, 105},
       {192, 23, 110, 197, 29, 115, 202, 33, 121, 207, 38, 125, 212, 43, 130, 217},
       {48, 135, 222, 53, 141, 227, 58, 145, 233, 63, 150, 237, 68, 155, 242, 73},
       {160, 247, 78, 165, 253, 83, 170, 1, 89, 175, 6, 93, 180, 11, 98, 185},
       {16, 103, 190, 21, 109, 195, 26, 113, 201, 31, 118, 205, 36, 123, 210, 41}},
      {{0x5700, 0xafae, 0x085d, 0x610a, 0xb9b9, 0x1266, 0x6b14, 0xc3c2},
       {0x1c70, 0x751e, 0xcdcd, 0x267a, 0x7f29, 0xd7d6, 0x3084, 0x8932},
       {0xe1e0, 0x3a8e, 0x933d, 0xebea, 0x4499, 0x9d46, 0xf5f4, 0x4ea2},
       {0xa750, 0xfffe, 0x58ad, 0xb15a, 0x0a09, 0x62b6, 0xbb64, 0x1412},
       {0x6cc0, 0xc56e, 0x1e1d, 0x76ca, 0xcf79, 0x2826, 0x80d4, 0xd982},
       {0x3230, 0x8ade, 0xe38d, 0x3c3a, 0x94e9, 0xed96, 0x4644, 0x9ef2},
       {0xf7a0, 0x504e, 0xa8fd, 0x01aa, 0x5a59, 0xb306, 0x0bb4, 0x6462},
       {0xbd10, 0x15be, 0x6e6d, 0xc71a, 0x1fc9, 0x7876, 0xd124, 0x29d2}},
      {{0x0503'5700, 0x610a'085d, 0xbd10'b9b9, 0x1917'6b14},
       {0x751e'1c70, 0xd124'cdcd, 0x2d2b'7f29, 0x8932'3084},
       {0xe538'e1e0, 0x413f'933d, 0x9d46'4499, 0xf94c'f5f4},
       {0x5553'a750, 0xb15a'58ad, 0x0d61'0a09, 0x6967'bb64},
       {0xc56e'6cc0, 0x2175'1e1d, 0x7d7b'cf79, 0xd982'80d4},
       {0x3589'3230, 0x918f'e38d, 0xed96'94e9, 0x499d'4644},
       {0xa5a3'f7a0, 0x01aa'a8fd, 0x5db1'5a59, 0xb9b8'0bb4},
       {0x15be'bd10, 0x71c5'6e6d, 0xcdcc'1fc9, 0x29d2'd124}},
      {{0x610a'085d'0503'5700, 0xc3c2'15be'bd10'b9b9},
       {0x267a'2322'751e'1c70, 0x8932'3084'2d2b'7f29},
       {0xebea'3de7'e538'e1e0, 0x4ea2'4b49'9d46'4499},
       {0xb15a'58ad'5553'a750, 0x1412'660f'0d61'0a09},
       {0x76ca'7372'c56e'6cc0, 0xd982'80d4'7d7b'cf79},
       {0x3c3a'8e38'3589'3230, 0x9ef2'9b99'ed96'94e9},
       {0x01aa'a8fd'a5a3'f7a0, 0x6462'b65f'5db1'5a59},
       {0xc71a'c3c3'15be'bd10, 0x29d2'd124'cdcc'1fc9}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0xa500e457,  // vmadd.vx v8, x1, v16, v0.t
      {{114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129},
       {130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145},
       {146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161},
       {162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177},
       {178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193},
       {194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209},
       {210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225},
       {226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241}},
      {{0x1d72, 0x1f74, 0x2176, 0x2378, 0x257a, 0x277c, 0x297e, 0x2b80},
       {0x2d82, 0x2f84, 0x3186, 0x3388, 0x358a, 0x378c, 0x398e, 0x3b90},
       {0x3d92, 0x3f94, 0x4196, 0x4398, 0x459a, 0x479c, 0x499e, 0x4ba0},
       {0x4da2, 0x4fa4, 0x51a6, 0x53a8, 0x55aa, 0x57ac, 0x59ae, 0x5bb0},
       {0x5db2, 0x5fb4, 0x61b6, 0x63b8, 0x65ba, 0x67bc, 0x69be, 0x6bc0},
       {0x6dc2, 0x6fc4, 0x71c6, 0x73c8, 0x75ca, 0x77cc, 0x79ce, 0x7bd0},
       {0x7dd2, 0x7fd4, 0x81d6, 0x83d8, 0x85da, 0x87dc, 0x89de, 0x8be0},
       {0x8de2, 0x8fe4, 0x91e6, 0x93e8, 0x95ea, 0x97ec, 0x99ee, 0x9bf0}},
      {{0x74c9'1d72, 0x78cd'2176, 0x7cd1'257a, 0x80d5'297e},
       {0x84d9'2d82, 0x88dd'3186, 0x8ce1'358a, 0x90e5'398e},
       {0x94e9'3d92, 0x98ed'4196, 0x9cf1'459a, 0xa0f5'499e},
       {0xa4f9'4da2, 0xa8fd'51a6, 0xad01'55aa, 0xb105'59ae},
       {0xb509'5db2, 0xb90d'61b6, 0xbd11'65ba, 0xc115'69be},
       {0xc519'6dc2, 0xc91d'71c6, 0xcd21'75ca, 0xd125'79ce},
       {0xd529'7dd2, 0xd92d'81d6, 0xdd31'85da, 0xe135'89de},
       {0xe539'8de2, 0xe93d'91e6, 0xed41'95ea, 0xf145'99ee}},
      {{0x2377'cc20'74c9'1d72, 0x2b7f'd428'7cd1'257a},
       {0x3387'dc30'84d9'2d82, 0x3b8f'e438'8ce1'358a},
       {0x4397'ec40'94e9'3d92, 0x4b9f'f448'9cf1'459a},
       {0x53a7'fc50'a4f9'4da2, 0x5bb0'0458'ad01'55aa},
       {0x63b8'0c60'b509'5db2, 0x6bc0'1468'bd11'65ba},
       {0x73c8'1c70'c519'6dc2, 0x7bd0'2478'cd21'75ca},
       {0x83d8'2c80'd529'7dd2, 0x8be0'3488'dd31'85da},
       {0x93e8'3c90'e539'8de2, 0x9bf0'4498'ed41'95ea}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVnmsub) {
  TestVectorInstruction(
      0xad882457,  // vnmsub.vv v8, v16, v24, v0.t
      {{0, 173, 90, 7, 181, 97, 14, 187, 105, 21, 194, 111, 28, 201, 118, 35},
       {208, 125, 42, 215, 133, 49, 222, 139, 57, 229, 146, 63, 236, 153, 70, 243},
       {160, 77, 250, 167, 85, 1, 174, 91, 9, 181, 98, 15, 188, 105, 22, 195},
       {112, 29, 202, 119, 37, 209, 126, 43, 217, 133, 50, 223, 140, 57, 230, 147},
       {64, 237, 154, 71, 245, 161, 78, 251, 169, 85, 2, 175, 92, 9, 182, 99},
       {16, 189, 106, 23, 197, 113, 30, 203, 121, 37, 210, 127, 44, 217, 134, 51},
       {224, 141, 58, 231, 149, 65, 238, 155, 73, 245, 162, 79, 252, 169, 86, 3},
       {176, 93, 10, 183, 101, 17, 190, 107, 25, 197, 114, 31, 204, 121, 38, 211}},
      {{0xad00, 0x5c5a, 0x0bb5, 0xbb0e, 0x6a69, 0x19c2, 0xc91c, 0x7876},
       {0x27d0, 0xd72a, 0x8685, 0x35de, 0xe539, 0x9492, 0x43ec, 0xf346},
       {0xa2a0, 0x51fa, 0x0155, 0xb0ae, 0x6009, 0x0f62, 0xbebc, 0x6e16},
       {0x1d70, 0xccca, 0x7c25, 0x2b7e, 0xdad9, 0x8a32, 0x398c, 0xe8e6},
       {0x9840, 0x479a, 0xf6f5, 0xa64e, 0x55a9, 0x0502, 0xb45c, 0x63b6},
       {0x1310, 0xc26a, 0x71c5, 0x211e, 0xd079, 0x7fd2, 0x2f2c, 0xde86},
       {0x8de0, 0x3d3a, 0xec95, 0x9bee, 0x4b49, 0xfaa2, 0xa9fc, 0x5956},
       {0x08b0, 0xb80a, 0x6765, 0x16be, 0xc619, 0x7572, 0x24cc, 0xd426}},
      {{0x0704'ad00, 0xbb0e'0bb5, 0x6f17'6a69, 0x2320'c91c},
       {0xd72a'27d0, 0x8b33'8685, 0x3f3c'e539, 0xf346'43ec},
       {0xa74f'a2a0, 0x5b59'0155, 0x0f62'6009, 0xc36b'bebc},
       {0x7775'1d70, 0x2b7e'7c25, 0xdf87'dad9, 0x9391'398c},
       {0x479a'9840, 0xfba3'f6f5, 0xafad'55a9, 0x63b6'b45c},
       {0x17c0'1310, 0xcbc9'71c5, 0x7fd2'd079, 0x33dc'2f2c},
       {0xe7e5'8de0, 0x9bee'ec95, 0x4ff8'4b49, 0x0401'a9fc},
       {0xb80b'08b0, 0x6c14'6765, 0x201d'c619, 0xd427'24cc}},
      {{0xbb0e'0bb5'0704'ad00, 0x7876'1e71'6f17'6a69},
       {0x35de'312f'd72a'27d0, 0xf346'43ec'3f3c'e539},
       {0xb0ae'56aa'a74f'a2a0, 0x6e16'6967'0f62'6009},
       {0x2b7e'7c25'7775'1d70, 0xe8e6'8ee1'df87'dad9},
       {0xa64e'a1a0'479a'9840, 0x63b6'b45c'afad'55a9},
       {0x211e'c71b'17c0'1310, 0xde86'd9d7'7fd2'd079},
       {0x9bee'ec95'e7e5'8de0, 0x5956'ff52'4ff8'4b49},
       {0x16bf'1210'b80b'08b0, 0xd427'24cd'201d'c619}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0xad00e457,  // vnmsub.vx v8, x1, v16, v0.t
      {{142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157},
       {158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173},
       {174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189},
       {190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205},
       {206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221},
       {222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237},
       {238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253},
       {254, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}},
      {{0xe48e, 0xe690, 0xe892, 0xea94, 0xec96, 0xee98, 0xf09a, 0xf29c},
       {0xf49e, 0xf6a0, 0xf8a2, 0xfaa4, 0xfca6, 0xfea8, 0x00aa, 0x02ac},
       {0x04ae, 0x06b0, 0x08b2, 0x0ab4, 0x0cb6, 0x0eb8, 0x10ba, 0x12bc},
       {0x14be, 0x16c0, 0x18c2, 0x1ac4, 0x1cc6, 0x1ec8, 0x20ca, 0x22cc},
       {0x24ce, 0x26d0, 0x28d2, 0x2ad4, 0x2cd6, 0x2ed8, 0x30da, 0x32dc},
       {0x34de, 0x36e0, 0x38e2, 0x3ae4, 0x3ce6, 0x3ee8, 0x40ea, 0x42ec},
       {0x44ee, 0x46f0, 0x48f2, 0x4af4, 0x4cf6, 0x4ef8, 0x50fa, 0x52fc},
       {0x54fe, 0x5700, 0x5902, 0x5b04, 0x5d06, 0x5f08, 0x610a, 0x630c}},
      {{0x913a'e48e, 0x953e'e892, 0x9942'ec96, 0x9d46'f09a},
       {0xa14a'f49e, 0xa54e'f8a2, 0xa952'fca6, 0xad57'00aa},
       {0xb15b'04ae, 0xb55f'08b2, 0xb963'0cb6, 0xbd67'10ba},
       {0xc16b'14be, 0xc56f'18c2, 0xc973'1cc6, 0xcd77'20ca},
       {0xd17b'24ce, 0xd57f'28d2, 0xd983'2cd6, 0xdd87'30da},
       {0xe18b'34de, 0xe58f'38e2, 0xe993'3ce6, 0xed97'40ea},
       {0xf19b'44ee, 0xf59f'48f2, 0xf9a3'4cf6, 0xfda7'50fa},
       {0x01ab'54fe, 0x05af'5902, 0x09b3'5d06, 0x0db7'610a}},
      {{0xea94'3de7'913a'e48e, 0xf29c'45ef'9942'ec96},
       {0xfaa4'4df7'a14a'f49e, 0x02ac'55ff'a952'fca6},
       {0x0ab4'5e07'b15b'04ae, 0x12bc'660f'b963'0cb6},
       {0x1ac4'6e17'c16b'14be, 0x22cc'761f'c973'1cc6},
       {0x2ad4'7e27'd17b'24ce, 0x32dc'862f'd983'2cd6},
       {0x3ae4'8e37'e18b'34de, 0x42ec'963f'e993'3ce6},
       {0x4af4'9e47'f19b'44ee, 0x52fc'a64f'f9a3'4cf6},
       {0x5b04'ae58'01ab'54fe, 0x630c'b660'09b3'5d06}},
      kVectorCalculationsSource);
}

}  // namespace

}  // namespace berberis
