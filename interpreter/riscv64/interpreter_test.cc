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
  Riscv64InterpreterTest() : state_{.cpu = {.frm = intrinsics::GuestModeFromHostRounding()}} {}

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
                             const __v16qu (&result_int8)[8],
                             const __v8hu (&result_int16)[8],
                             const __v4su (&result_int32)[8],
                             const __v2du (&result_int64)[8]) {
    constexpr __v16qu kMaskInt8[8] = {
        {255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255},
        {255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255},
        {255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255},
        {255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 0, 255, 255},
        {255, 0, 255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0},
        {255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 255},
        {255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0},
        {255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 0}};
    constexpr __v8hu kMaskInt16[8] = {
        {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
        {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff},
        {0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000},
        {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
        {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff},
        {0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff},
        {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
        {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff}};
    constexpr __v4su kMaskInt32[8] = {
        {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
        {0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
        {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0x0000'0000},
        {0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
        {0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
        {0x0000'0000, 0xffff'ffff, 0xffff'ffff, 0x0000'0000},
        {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
        {0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff}};
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

    auto Verify = [this, &insn_bytes](uint8_t vsew, auto result, auto mask) {
      constexpr __m128i kUndisturbedResult = {0x5555'5555'5555'5555, 0x5555'5555'5555'5555};
      constexpr __v2du kMask = {0xd5ad'd6b5'ad6b'b5ad, 0x6af7'57bb'deed'7bb5};
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
      for (size_t index = 0; index < arraysize(source); ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }
      for (uint8_t vlmul = 0; vlmul < 8; ++vlmul) {
        // Use vsetvl to check whether this combination is allowed and find out VLMAX.
        uint32_t vsetvl = 0x803170d7;
        state_.cpu.insn_addr = ToGuestAddr(&vsetvl);
        SetXReg<2>(state_.cpu, ~0ULL);
        SetXReg<3>(state_.cpu, (vsew << 3) | vlmul);
        EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
        uint64_t vlmax = GetXReg<1>(state_.cpu);
        // Incompatible vsew and vlmax. Skip it.
        if (vlmax == 0) {
          continue;
        }

        if ((insn_bytes & (1 << 25)) == 0) {
          // Set result vector registers into 0b01010101… pattern.
          for (size_t index = 0; index < 8; ++index) {
            state_.cpu.v[8 + index] = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
          }

          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

          if (vlmul < 4) {
            for (size_t index = 0; index < 1 << vlmul; ++index) {
              EXPECT_EQ(state_.cpu.v[8 + index],
                        SIMD128Register{(result[index] & mask[index]) |
                                        (kUndisturbedResult & !mask[index])}
                            .Get<__uint128_t>());
            }
          }
        }

        // Set result vector registers into 0b01010101… pattern.
        for (size_t index = 0; index < 8; ++index) {
          state_.cpu.v[8 + index] = SIMD128Register{__m128i{kUndisturbedResult}}.Get<__uint128_t>();
        }

        uint32_t no_mask_insn_bytes = insn_bytes | (1 << 25);
        state_.cpu.insn_addr = ToGuestAddr(&no_mask_insn_bytes);
        EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

        if (vlmul < 4) {
          for (size_t index = 0; index < 1 << vlmul; ++index) {
            EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{result[index]}.Get<__uint128_t>());
          }
        }
      }
    };

    Verify(0, result_int8, kMaskInt8);
    Verify(1, result_int16, kMaskInt16);
    Verify(2, result_int32, kMaskInt32);
    Verify(3, result_int64, kMaskInt64);
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

TEST_F(Riscv64InterpreterTest, TestVaddvv) {
  TestVectorInstruction(
      0x10c0457,
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
}

}  // namespace

}  // namespace berberis
