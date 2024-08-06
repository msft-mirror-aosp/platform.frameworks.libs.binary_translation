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

#include "gtest/gtest.h"

#include <cstdint>
#include <initializer_list>
#include <tuple>

#include "berberis/base/bit_util.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/riscv64/interpreter.h"

namespace berberis {

namespace {

class Riscv64ToArm64InterpreterTest : public ::testing::Test {
 public:
  template <uint8_t kInsnSize = 4>
  void RunInstruction(const uint32_t& insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    InterpretInsn(&state_);
  }

  void TestOp(uint32_t insn_bytes,
              // The tuple is [arg1, arg2, expected_result].
              std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto arg : args) {
      SetXReg<2>(state_.cpu, std::get<0>(arg));
      SetXReg<3>(state_.cpu, std::get<1>(arg));
      RunInstruction(insn_bytes);
      EXPECT_EQ(GetXReg<1>(state_.cpu), std::get<2>(arg));
    }
  }

  void TestOpImm(uint32_t insn_bytes,
                 std::initializer_list<std::tuple<uint64_t, uint16_t, uint64_t>> args) {
    for (auto [arg1, imm, expected_result] : args) {
      CHECK_LE(imm, 63);
      uint32_t insn_bytes_with_immediate = insn_bytes | imm << 20;
      SetXReg<2>(state_.cpu, arg1);
      RunInstruction(insn_bytes_with_immediate);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  void TestAuipc(uint32_t insn_bytes, uint64_t expected_offset) {
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_offset + ToGuestAddr(&insn_bytes));
  }

  void TestLui(uint32_t insn_bytes, uint64_t expected_result) {
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  void TestBranch(uint32_t insn_bytes,
                  std::initializer_list<std::tuple<uint64_t, uint64_t, int8_t>> args) {
    auto code_start = ToGuestAddr(&insn_bytes);
    for (auto [arg1, arg2, expected_offset] : args) {
      state_.cpu.insn_addr = code_start;
      SetXReg<1>(state_.cpu, arg1);
      SetXReg<2>(state_.cpu, arg2);
      InterpretInsn(&state_);
      EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    }
  }

  void TestJumpAndLink(uint32_t insn_bytes, int8_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    InterpretInsn(&state_);
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    EXPECT_EQ(GetXReg<1>(state_.cpu), code_start + 4);
  }

  void TestLoad(uint32_t insn_bytes, uint64_t expected_result) {
    // Offset is always 8.
    SetXReg<2>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&kDataToLoad) - 8));
    RunInstruction(insn_bytes);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  // kLinkRegisterOffsetIfUsed is size of instruction or 0 if instruction does not link register.
  template <uint8_t kLinkRegisterOffsetIfUsed>
  void TestJumpAndLinkRegister(uint32_t insn_bytes, uint64_t base_disp, int64_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<1>(state_.cpu, 0);
    SetXReg<2>(state_.cpu, code_start + base_disp);
    InterpretInsn(&state_);
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    if constexpr (kLinkRegisterOffsetIfUsed == 0) {
      EXPECT_EQ(GetXReg<1>(state_.cpu), 0UL);
    } else {
      EXPECT_EQ(GetXReg<1>(state_.cpu), code_start + kLinkRegisterOffsetIfUsed);
    }
  }

  void TestStore(uint32_t insn_bytes, uint64_t expected_result) {
    // Offset is always 8.
    SetXReg<1>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - 8));
    SetXReg<2>(state_.cpu, kDataToStore);
    store_area_ = 0;
    RunInstruction(insn_bytes);
    EXPECT_EQ(store_area_, expected_result);
  }

 protected:
  static constexpr uint64_t kDataToLoad{0xffffeeeeddddccccULL};
  static constexpr uint64_t kDataToStore = kDataToLoad;
  uint64_t store_area_;
  ThreadState state_;
};

TEST_F(Riscv64ToArm64InterpreterTest, OpInstructions) {
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
  // Andn
  TestOp(0x403170b3, {{0b0101, 0b0011, 0b0100}});
  // Orn
  TestOp(0x403160b3, {{0b0101, 0b0011, 0xffff'ffff'ffff'fffd}});
  // Xnor
  TestOp(0x403140b3, {{0b0101, 0b0011, 0xffff'ffff'ffff'fff9}});
}

TEST_F(Riscv64ToArm64InterpreterTest, OpImmInstructions) {
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
  // Rori
  TestOpImm(0x60015093, {{0xf000'0000'0000'000fULL, 4, 0xff00'0000'0000'0000ULL}});
}

TEST_F(Riscv64ToArm64InterpreterTest, UpperImmInstructions) {
  // Auipc
  TestAuipc(0xfedcb097, 0xffff'ffff'fedc'b000);
  // Lui
  TestLui(0xfedcb0b7, 0xffff'ffff'fedc'b000);
}

TEST_F(Riscv64ToArm64InterpreterTest, TestBranchInstructions) {
  // Beq
  TestBranch(0x00208463,
             {
                 {42, 42, 8},
                 {41, 42, 4},
                 {42, 41, 4},
             });
  // Bne
  TestBranch(0x00209463,
             {
                 {42, 42, 4},
                 {41, 42, 8},
                 {42, 41, 8},
             });
  // Bltu
  TestBranch(0x0020e463,
             {
                 {41, 42, 8},
                 {42, 42, 4},
                 {42, 41, 4},
                 {0xf000'0000'0000'0000ULL, 42, 4},
                 {42, 0xf000'0000'0000'0000ULL, 8},
             });
  // Bgeu
  TestBranch(0x0020f463,
             {
                 {42, 41, 8},
                 {42, 42, 8},
                 {41, 42, 4},
                 {0xf000'0000'0000'0000ULL, 42, 8},
                 {42, 0xf000'0000'0000'0000ULL, 4},
             });
  // Blt
  TestBranch(0x0020c463,
             {
                 {41, 42, 8},
                 {42, 42, 4},
                 {42, 41, 4},
                 {0xf000'0000'0000'0000ULL, 42, 8},
                 {42, 0xf000'0000'0000'0000ULL, 4},
             });
  // Bge
  TestBranch(0x0020d463,
             {
                 {42, 41, 8},
                 {42, 42, 8},
                 {41, 42, 4},
                 {0xf000'0000'0000'0000ULL, 42, 4},
                 {42, 0xf000'0000'0000'0000ULL, 8},
             });
  // Beq with negative offset.
  TestBranch(0xfe208ee3,
             {
                 {42, 42, -4},
             });
}

TEST_F(Riscv64ToArm64InterpreterTest, JumpAndLinkInstructions) {
  // Jal
  TestJumpAndLink(0x008000ef, 8);
  // Jal with negative offset.
  TestJumpAndLink(0xffdff0ef, -4);
}

TEST_F(Riscv64ToArm64InterpreterTest, JumpAndLinkRegisterInstructions) {
  // Jalr offset=4.
  TestJumpAndLinkRegister<4>(0x004100e7, 38, 42);
  // Jalr offset=-4.
  TestJumpAndLinkRegister<4>(0xffc100e7, 42, 38);
  // Jalr offset=5 - must properly align the target to even.
  TestJumpAndLinkRegister<4>(0x005100e7, 38, 42);
  // Jr offset=4.
  TestJumpAndLinkRegister<0>(0x00410067, 38, 42);
  // Jr offset=-4.
  TestJumpAndLinkRegister<0>(0xffc10067, 42, 38);
  // Jr offset=5 - must properly align the target to even.
  TestJumpAndLinkRegister<0>(0x00510067, 38, 42);
}

TEST_F(Riscv64ToArm64InterpreterTest, LoadInstructions) {
  // Offset is always 8.
  // Lbu
  TestLoad(0x00814083, kDataToLoad & 0xffULL);
  // Lhu
  TestLoad(0x00815083, kDataToLoad & 0xffffULL);
  // Lwu
  TestLoad(0x00816083, kDataToLoad & 0xffff'ffffULL);
  // Ldu
  TestLoad(0x00813083, kDataToLoad);
  // Lb
  TestLoad(0x00810083, int64_t{int8_t(kDataToLoad)});
  // Lh
  TestLoad(0x00811083, int64_t{int16_t(kDataToLoad)});
  // Lw
  TestLoad(0x00812083, int64_t{int32_t(kDataToLoad)});
}

TEST_F(Riscv64ToArm64InterpreterTest, StoreInstructions) {
  // Offset is always 8.
  // Sb
  TestStore(0x00208423, kDataToStore & 0xffULL);
  // Sh
  TestStore(0x00209423, kDataToStore & 0xffffULL);
  // Sw
  TestStore(0x0020a423, kDataToStore & 0xffff'ffffULL);
  // Sd
  TestStore(0x0020b423, kDataToStore);
}

// Corresponding to interpreter_test.cc

TEST_F(Riscv64ToArm64InterpreterTest, FenceInstructions) {
  // Fence
  RunInstruction(0x0ff0000f);
  // FenceTso
  RunInstruction(0x8330000f);

  // FenceI explicitly not supported.
}

}  // namespace

}  // namespace berberis
