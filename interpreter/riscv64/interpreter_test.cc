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
                             const __v2du (&expected_result_int64)[8]) {
    // Mask in form suitable for storing in v0 and use in v0.t form.
    constexpr __v2du kMask = {0xd5ad'd6b5'ad6b'b5ad, 0x6af7'57bb'deed'7bb5};
    // Mask used with vsew = 0 (8bit) elements.
    constexpr __v16qu kMaskInt8[8] = {
        {255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255},
        {255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255},
        {255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255},
        {255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 0, 255, 255},
        {255, 0, 255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0},
        {255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 255},
        {255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0},
        {255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 0}};
    // Mask used with vsew = 1 (16bit) elements.
    constexpr __v8hu kMaskInt16[8] = {
        {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
        {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff},
        {0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000},
        {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
        {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff},
        {0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff},
        {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
        {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff}};
    // Mask used with vsew = 2 (32bit) elements.
    constexpr __v4su kMaskInt32[8] = {
        {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
        {0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
        {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0x0000'0000},
        {0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
        {0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
        {0x0000'0000, 0xffff'ffff, 0xffff'ffff, 0x0000'0000},
        {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
        {0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff}};
    // Mask used with vsew = 3 (64bit) elements.
    constexpr __v2du kMaskInt64[8] = {
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
    constexpr __v16qu kNoMask[8] = {
        {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}};

    auto Verify = [this, &kMask](uint32_t insn_bytes,
                                 uint8_t vsew,
                                 uint8_t vlmul_max,
                                 const auto& expected_result,
                                 auto mask) {
      constexpr __v16qu kFractionMaskInt8[4] = {
          {255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0}};
      constexpr __m128i kAgnosticResult = {-1, -1};
      constexpr __m128i kUndisturbedResult = {0x5555'5555'5555'5555, 0x5555'5555'5555'5555};
      constexpr __v2du source[16] = {
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
  static constexpr uint64_t kDataToLoad{0xffffeeeeddddccccULL};
  static constexpr uint64_t kDataToStore = kDataToLoad;
  uint64_t store_area_;
  ThreadState state_;
};

#define TESTSUITE Riscv64InterpretInsnTest

#include "berberis/test_utils/insn_tests_riscv64-inl.h"

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
      0x10c0457, // Vadd.vv v8, v16, v24, v0.t
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
       {0x6663'605e'5a57'5450, 0x7e7b'7875'726f'6c69}});
  TestVectorInstruction(
      0x100c457, // Vadd.vv v8, v16, x1, v0.t
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
       {0x2221'201f'1e1d'1c1a, 0x2a29'2827'2625'2422}});
  TestVectorInstruction(
      0x10ab457, // Vadd.vv v8, v16, -0xb, v0.t
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
       {0x7776'7574'7372'7165, 0x7f7e'7d7c'7b7a'796d}});
}

TEST_F(Riscv64InterpreterTest, TestVrsub) {
  TestVectorInstruction(
      0xd00c457, // Vrsub.vi v8, v16, x1, v0.t
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
       {0x3334'3536'3738'393a, 0x2b2c'2d2e'2f30'3132}});
  TestVectorInstruction(
      0xd0ab457, // Vrsub.vi v8, v16, -0xb, v0.t
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
       {0x8889'8a8b'8c8d'8e85, 0x8081'8283'8485'867d}});
}

TEST_F(Riscv64InterpreterTest, TestVsub) {
  TestVectorInstruction(
      0x90c0457, // Vsub.vv v8, v16, v24, v0.t
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
       {0x8889'8a8a'8c8d'8e90, 0x8081'8283'8485'8687}});
  TestVectorInstruction(
      0x900c457, // Vsub.vx v8, v16, x1, v0.t
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
       {0xcccb'cac9'c8c7'c6c6, 0xd4d3'd2d1'd0cf'cece}});
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
       {0x6664'6060'6260'6060, 0x7e7c'7878'7270'7070}});
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
                         {0x2222'2020'2222'2020, 0x2a2a'2828'2a2a'2828}});
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
       {0x7776'7574'7372'7170, 0x7f7e'7d7c'7b7a'7970}});
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
       {0xfffe'fffd'f7f6'f3f0, 0xfffe'fffc'fffe'fbf9}});
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
       {0xfffe'fffe'fbfa'fbfa, 0xfffe'fffe'fbfa'fbfa}});
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
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd}});
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
       {0x999a'9f9d'9596'9390, 0x8182'8784'8d8e'8b89}});
  TestVectorInstruction(
      0x2d00c457,  // Vxor.vx v8, v16, x1, v0.t"
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
       {0xdddc'dfde'd9d8'dbda, 0xd5d4'd7d6'd1d0'd3d2}});
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
       {0x8889'8a8b'8c8d'8e85, 0x8081'8283'8485'868d}});
}
}  // namespace

}  // namespace berberis
