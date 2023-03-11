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
#include <initializer_list>
#include <tuple>
#include <type_traits>

#include "berberis/base/bit_util.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_riscv64.h"
#include "berberis/interpreter/riscv64/interpreter.h"

namespace berberis {

namespace {

class Riscv64InterpreterTest : public ::testing::Test {
 public:
  void InterpretOp(uint32_t insn_bytes,
                   std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto [arg1, arg2, expected_result] : args) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetXReg<2>(state_.cpu, arg1);
      SetXReg<3>(state_.cpu, arg2);
      InterpretInsn(&state_);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  void InterpretOpImm(uint32_t insn_bytes,
                      std::initializer_list<std::tuple<uint64_t, uint16_t, uint64_t>> args) {
    for (auto [arg1, imm, expected_result] : args) {
      CHECK_LE(imm, 63);
      uint32_t insn_bytes_with_immediate = insn_bytes | imm << 20;
      state_.cpu.insn_addr = bit_cast<GuestAddr>(&insn_bytes_with_immediate);
      SetXReg<2>(state_.cpu, arg1);
      InterpretInsn(&state_);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  void InterpretLoad(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    // Offset is always 8.
    SetXReg<2>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&kDataToLoad) - 8));
    InterpretInsn(&state_);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  void InterpretStore(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    // Offset is always 8.
    SetXReg<1>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - 8));
    SetXReg<2>(state_.cpu, kDataToStore);
    store_area_ = 0;
    InterpretInsn(&state_);
    EXPECT_EQ(store_area_, expected_result);
  }

  void InterpretBranch(uint32_t insn_bytes,
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

  void InterpretJumpAndLink(uint32_t insn_bytes, int8_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    InterpretInsn(&state_);
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    EXPECT_EQ(GetXReg<1>(state_.cpu), code_start + 4);
  }

  void InterpretJumpAndLinkRegister(uint32_t insn_bytes, uint64_t base_disp,
                                    int64_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<2>(state_.cpu, code_start + base_disp);
    InterpretInsn(&state_);
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
  }

 protected:
  static constexpr uint64_t kDataToLoad{0xffffeeeeddddccccULL};
  static constexpr uint64_t kDataToStore = kDataToLoad;
  uint64_t store_area_;
  ThreadState state_;
};

TEST_F(Riscv64InterpreterTest, OpInstructions) {
  // Add
  InterpretOp(0x003100b3, {{19, 23, 42}});
  // Sub
  InterpretOp(0x403100b3, {{42, 23, 19}});
  // And
  InterpretOp(0x003170b3, {{0b0101, 0b0011, 0b0001}});
  // Or
  InterpretOp(0x003160b3, {{0b0101, 0b0011, 0b0111}});
  // Xor
  InterpretOp(0x003140b3, {{0b0101, 0b0011, 0b0110}});
  // Sll
  InterpretOp(0x003110b3, {{0b1010, 3, 0b1010'000}});
  // Srl
  InterpretOp(0x003150b3, {{0xf000'0000'0000'0000ULL, 12, 0x000f'0000'0000'0000ULL}});
  // Sra
  InterpretOp(0x403150b3, {{0xf000'0000'0000'0000ULL, 12, 0xffff'0000'0000'0000ULL}});
  // Slt
  InterpretOp(0x003120b3, {
                              {19, 23, 1},
                              {23, 19, 0},
                              {~0ULL, 0, 1},
                          });
  // Sltu
  InterpretOp(0x003130b3, {
                              {19, 23, 1},
                              {23, 19, 0},
                              {~0ULL, 0, 0},
                          });

  // Mul
  InterpretOp(0x023100b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0x0a3d'70a3'd70a'3d71}});
  // Mulh
  InterpretOp(0x23110b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0x28f5'c28f'5c28'f5c3}});
  // Mulhsu
  InterpretOp(0x23120b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0xc28f'5c28'f5c2'8f5c}});
  // Mulhu
  InterpretOp(0x23130b3, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0x5c28'f5c2'8f5c'28f5}});
  // Div
  InterpretOp(0x23140b3, {{0x9999'9999'9999'9999, 0x3333, 0xfffd'fffd'fffd'fffe}});
  // Div
  InterpretOp(0x23150b3, {{0x9999'9999'9999'9999, 0x3333, 0x0003'0003'0003'0003}});
  // Rem
  InterpretOp(0x23160b3, {{0x9999'9999'9999'9999, 0x3333, 0xffff'ffff'ffff'ffff}});
  // Remu
  InterpretOp(0x23170b3, {{0x9999'9999'9999'9999, 0x3333, 0}});
}

TEST_F(Riscv64InterpreterTest, Op32Instructions) {
  // Addw
  InterpretOp(0x003100bb, {{19, 23, 42}});
  // Subw
  InterpretOp(0x403100bb, {{42, 23, 19}});
  // Sllw
  InterpretOp(0x003110bb, {{0b1010, 3, 0b1010'000}});
  // Srlw
  InterpretOp(0x003150bb, {{0x0000'0000'f000'0000ULL, 12, 0x0000'0000'000f'0000ULL}});
  // Sraw
  InterpretOp(0x403150bb, {{0x0000'0000'f000'0000ULL, 12, 0xffff'ffff'ffff'0000ULL}});
  // Mulw
  InterpretOp(0x023100bb, {{0x9999'9999'9999'9999, 0x9999'9999'9999'9999, 0xffff'ffff'd70a'3d71}});
  // Divw
  InterpretOp(0x23140bb, {{0x9999'9999'9999'9999, 0x3333, 0xffff'ffff'fffd'fffe}});
  // Divuw
  InterpretOp(0x23150bb, {{0x9999'9999'9999'9999, 0x3333, 0x0000'0000'0003'0003}});
  // Remw
  InterpretOp(0x23160bb, {{0x9999'9999'9999'9999, 0x3333, 0xffff'ffff'ffff'ffff}});
  // Remuw
  InterpretOp(0x23170bb, {{0x9999'9999'9999'9999, 0x3333, 0}});
}

TEST_F(Riscv64InterpreterTest, OpImmInstructions) {
  // Addi
  InterpretOpImm(0x00010093, {{19, 23, 42}});
  // Slli
  InterpretOpImm(0x00011093, {{0b1010, 3, 0b1010'000}});
  // Slti
  InterpretOpImm(0x00012093, {
                                 {19, 23, 1},
                                 {23, 19, 0},
                                 {~0ULL, 0, 1},
                             });
  // Sltiu
  InterpretOpImm(0x00013093, {
                                 {19, 23, 1},
                                 {23, 19, 0},
                                 {~0ULL, 0, 0},
                             });
  // Xori
  InterpretOpImm(0x00014093, {{0b0101, 0b0011, 0b0110}});
  // Srli
  InterpretOpImm(0x00015093, {{0xf000'0000'0000'0000ULL, 12, 0x000f'0000'0000'0000ULL}});
  // Srai
  InterpretOpImm(0x40015093, {{0xf000'0000'0000'0000ULL, 12, 0xffff'0000'0000'0000ULL}});
  // Ori
  InterpretOpImm(0x00016093, {{0b0101, 0b0011, 0b0111}});
  // Andi
  InterpretOpImm(0x00017093, {{0b0101, 0b0011, 0b0001}});
}

TEST_F(Riscv64InterpreterTest, OpImm32Instructions) {
  // Addiw
  InterpretOpImm(0x0001009b, {{19, 23, 42}});
  // Slliw
  InterpretOpImm(0x0001109b, {{0b1010, 3, 0b1010'000}});
  // Srliw
  InterpretOpImm(0x0001509b, {{0x0000'0000'f000'0000ULL, 12, 0x0000'0000'000f'0000ULL}});
  // Sraiw
  InterpretOpImm(0x4001509b, {{0x0000'0000'f000'0000ULL, 12, 0xffff'ffff'ffff'0000ULL}});
}

TEST_F(Riscv64InterpreterTest, LoadInstructions) {
  // Offset is always 8.
  // Lbu
  InterpretLoad(0x00814083, kDataToLoad & 0xffULL);
  // Lhu
  InterpretLoad(0x00815083, kDataToLoad & 0xffffULL);
  // Lwu
  InterpretLoad(0x00816083, kDataToLoad & 0xffff'ffffULL);
  // Ldu
  InterpretLoad(0x00813083, kDataToLoad);
  // Lb
  InterpretLoad(0x00810083, int64_t{int8_t(kDataToLoad)});
  // Lh
  InterpretLoad(0x00811083, int64_t{int16_t(kDataToLoad)});
  // Lw
  InterpretLoad(0x00812083, int64_t{int32_t(kDataToLoad)});
}

TEST_F(Riscv64InterpreterTest, StoreInstructions) {
  // Offset is always 8.
  // Sb
  InterpretStore(0x00208423, kDataToStore & 0xffULL);
  // Sh
  InterpretStore(0x00209423, kDataToStore & 0xffffULL);
  // Sw
  InterpretStore(0x0020a423, kDataToStore & 0xffff'ffffULL);
  // Sd
  InterpretStore(0x0020b423, kDataToStore);
}

TEST_F(Riscv64InterpreterTest, BranchInstructions) {
  // Beq
  InterpretBranch(0x00208463, {
                                  {42, 42, 8},
                                  {41, 42, 4},
                                  {42, 41, 4},
                              });
  // Bne
  InterpretBranch(0x00209463, {
                                  {42, 42, 4},
                                  {41, 42, 8},
                                  {42, 41, 8},
                              });
  // Blt
  InterpretBranch(0x0020c463, {
                                  {41, 42, 8},
                                  {42, 42, 4},
                                  {42, 41, 4},
                                  {0xf000'0000'0000'0000ULL, 42, 8},
                                  {42, 0xf000'0000'0000'0000ULL, 4},
                              });
  // Bltu
  InterpretBranch(0x0020e463, {
                                  {41, 42, 8},
                                  {42, 42, 4},
                                  {42, 41, 4},
                                  {0xf000'0000'0000'0000ULL, 42, 4},
                                  {42, 0xf000'0000'0000'0000ULL, 8},
                              });
  // Bge
  InterpretBranch(0x0020d463, {
                                  {42, 41, 8},
                                  {42, 42, 8},
                                  {41, 42, 4},
                                  {0xf000'0000'0000'0000ULL, 42, 4},
                                  {42, 0xf000'0000'0000'0000ULL, 8},
                              });
  // Bgeu
  InterpretBranch(0x0020f463, {
                                  {42, 41, 8},
                                  {42, 42, 8},
                                  {41, 42, 4},
                                  {0xf000'0000'0000'0000ULL, 42, 8},
                                  {42, 0xf000'0000'0000'0000ULL, 4},
                              });
  // Beq with negative offset.
  InterpretBranch(0xfe208ee3, {
                                  {42, 42, -4},
                              });
}

TEST_F(Riscv64InterpreterTest, JumpAndLinkInstructions) {
  // Jal
  InterpretJumpAndLink(0x008000ef, 8);
  // Jal with negative offset.
  InterpretJumpAndLink(0xffdff0ef, -4);
}

TEST_F(Riscv64InterpreterTest, JumpAndLinkRegisterInstructions) {
  // Jalr offset=4.
  InterpretJumpAndLinkRegister(0x004100e7, 38, 42);
  // Jalr offset=-4.
  InterpretJumpAndLinkRegister(0xffc100e7, 42, 38);
  // Jalr offset=5 - must properly align the target to even.
  InterpretJumpAndLinkRegister(0x005100e7, 38, 42);
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
  read(pipefd[0], &buf, sizeof(buf));
  EXPECT_EQ(0, strcmp(message, buf));
  close(pipefd[0]);
  close(pipefd[1]);
}

}  // namespace

}  // namespace berberis
