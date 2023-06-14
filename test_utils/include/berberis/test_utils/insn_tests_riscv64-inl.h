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

// This inline-header is intended be included into a source file testing the correctness of
// riscv64 instructions execution by an interpreter or a jit-translator.
//
// Assumptions list:
//
// 1. Includes
//
// #include "gtest/gtest.h"
//
// #include <cstdint>
// #include <initializer_list>
// #include <tuple>
//
// #include "berberis/guest_state/guest_addr.h"
// #include "berberis/guest_state/guest_state_riscv64.h"
//
// 2. RunOneInstruction is defined and implemented
//
// 3. TESTSUITE macro is defined

#ifndef TESTSUITE
#error "TESTSUITE is undefined"
#endif

class TESTSUITE : public ::testing::Test {
 public:
  void TestOp(uint32_t insn_bytes,
              std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto [arg1, arg2, expected_result] : args) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetXReg<2>(state_.cpu, arg1);
      SetXReg<3>(state_.cpu, arg2);
      EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  void TestOpImm(uint32_t insn_bytes,
                 std::initializer_list<std::tuple<uint64_t, uint16_t, uint64_t>> args) {
    for (auto [arg1, imm, expected_result] : args) {
      CHECK_LE(imm, 63);
      uint32_t insn_bytes_with_immediate = insn_bytes | imm << 20;
      state_.cpu.insn_addr = bit_cast<GuestAddr>(&insn_bytes_with_immediate);
      SetXReg<2>(state_.cpu, arg1);
      EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  void TestAuipc(uint32_t insn_bytes, uint64_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_offset + code_start);
  }

  void TestLui(uint32_t insn_bytes, uint64_t expected_result) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

 private:
  ThreadState state_;
};

TEST_F(TESTSUITE, OpInstructions) {
  // Add
  TestOp(0x003100b3, {{19, 23, 42}});
  // Sub
  TestOp(0x403100b3, {{42, 23, 19}});
  // And
  TestOp(0x003170b3, {{0b0101, 0b0011, 0b0001}});
  // Or
  TestOp(0x003160b3, {{0b0101, 0b0011, 0b0111}});
  // Xor
  TestOp(0x003140b3, {{0b0101, 0b0011, 0b0110}});
  // Sll
  TestOp(0x003110b3, {{0b1010, 3, 0b1010'000}});
  // Srl
  TestOp(0x003150b3, {{0xf000'0000'0000'0000ULL, 12, 0x000f'0000'0000'0000ULL}});
  // Sra
  TestOp(0x403150b3, {{0xf000'0000'0000'0000ULL, 12, 0xffff'0000'0000'0000ULL}});
  // Slt
  TestOp(0x003120b3,
         {
             {19, 23, 1},
             {23, 19, 0},
             {~0ULL, 0, 1},
         });
  // Sltu
  TestOp(0x003130b3,
         {
             {19, 23, 1},
             {23, 19, 0},
             {~0ULL, 0, 0},
         });
  // Mul
  TestOp(0x023100b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0x0a3d'70a3'd70a'3d71}});
  // Mulh
  TestOp(0x23110b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0x28f5'c28f'5c28'f5c3}});
  // Mulhsu
  TestOp(0x23120b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0xc28f'5c28'f5c2'8f5c}});
  // Mulhu
  TestOp(0x23130b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0x5c28'f5c2'8f5c'28f5}});
  // Div
  TestOp(0x23140b3, {{0x9999'9999'9999'9999, 0x3333, 0xfffd'fffd'fffd'fffe}});
  // Div
  TestOp(0x23140b3, {{42, 2, 21}});
  // Divu
  TestOp(0x23150b3, {{0x9999'9999'9999'9999, 0x3333, 0x0003'0003'0003'0003}});
  // Rem
  TestOp(0x23160b3, {{0x9999'9999'9999'9999, 0x3333, 0xffff'ffff'ffff'ffff}});
  // Remu
  TestOp(0x23170b3, {{0x9999'9999'9999'9999, 0x3333, 0}});
}

TEST_F(TESTSUITE, Op32Instructions) {
  // Addw
  TestOp(0x003100bb, {{19, 23, 42}, {0x8000'0000, 0, 0xffff'ffff'8000'0000}});
  // Subw
  TestOp(0x403100bb, {{42, 23, 19}, {0x8000'0000, 0, 0xffff'ffff'8000'0000}});
  // Sllw
  TestOp(0x003110bb, {{0b1010, 3, 0b1010'000}});
  // Srlw
  TestOp(0x003150bb, {{0x0000'0000'f000'0000ULL, 12, 0x0000'0000'000f'0000ULL}});
  // Sraw
  TestOp(0x403150bb, {{0x0000'0000'f000'0000ULL, 12, 0xffff'ffff'ffff'0000ULL}});
  // Mulw
  TestOp(0x023100bb, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0xffff'ffff'd70a'3d71}});
  // Divw
  TestOp(0x23140bb, {{0x9999'9999'9999'9999, 0x3333, 0xffff'ffff'fffd'fffe}});
  // Divuw
  TestOp(0x23150bb,
         {{0x9999'9999'9999'9999, 0x3333, 0x0000'0000'0003'0003},
          {0xffff'ffff'8000'0000, 1, 0xffff'ffff'8000'0000}});
  // Remw
  TestOp(0x23160bb, {{0x9999'9999'9999'9999, 0x3333, 0xffff'ffff'ffff'ffff}});
  // Remuw
  TestOp(0x23170bb,
         {{0x9999'9999'9999'9999, 0x3333, 0},
          {0xffff'ffff'8000'0000, 0xffff'ffff'8000'0001, 0xffff'ffff'8000'0000}});
}

TEST_F(TESTSUITE, OpImmInstructions) {
  // Addi
  TestOpImm(0x00010093, {{19, 23, 42}});
  // Slti
  TestOpImm(0x00012093,
            {
                {19, 23, 1},
                {23, 19, 0},
                {~0ULL, 0, 1},
            });
  // Sltiu
  TestOpImm(0x00013093,
            {
                {19, 23, 1},
                {23, 19, 0},
                {~0ULL, 0, 0},
            });
  // Xori
  TestOpImm(0x00014093, {{0b0101, 0b0011, 0b0110}});
  // Ori
  TestOpImm(0x00016093, {{0b0101, 0b0011, 0b0111}});
  // Andi
  TestOpImm(0x00017093, {{0b0101, 0b0011, 0b0001}});
  // Slli
  TestOpImm(0x00011093, {{0b1010, 3, 0b1010'000}});
  // Srli
  TestOpImm(0x00015093, {{0xf000'0000'0000'0000ULL, 12, 0x000f'0000'0000'0000ULL}});
  // Srai
  TestOpImm(0x40015093, {{0xf000'0000'0000'0000ULL, 12, 0xffff'0000'0000'0000ULL}});
}

TEST_F(TESTSUITE, OpImm32Instructions) {
  // Addiw
  TestOpImm(0x0001009b, {{19, 23, 42}, {0x8000'0000, 0, 0xffff'ffff'8000'0000}});
  // Slliw
  TestOpImm(0x0001109b, {{0b1010, 3, 0b1010'000}});
  // Srliw
  TestOpImm(0x0001509b, {{0x0000'0000'f000'0000ULL, 12, 0x0000'0000'000f'0000ULL}});
  // Sraiw
  TestOpImm(0x4001509b, {{0x0000'0000'f000'0000ULL, 12, 0xffff'ffff'ffff'0000ULL}});
}

TEST_F(TESTSUITE, UpperImmInstructions) {
  // Auipc
  TestAuipc(0xfedcb097, 0xffff'ffff'fedc'b000);
  // Lui
  TestLui(0xfedcb0b7, 0xffff'ffff'fedc'b000);
}
