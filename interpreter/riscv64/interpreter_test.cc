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

#include <initializer_list>
#include <tuple>

#include "berberis/base/bit_util.h"
#include "berberis/guest_state/guest_state_riscv64.h"
#include "berberis/interpreter/riscv64/interpreter.h"

namespace berberis {

namespace {

class Riscv64InterpreterTest : public ::testing::Test {
 public:
  void InterpretOp(uint32_t insn_bytes,
                   // The tuple is [arg1, arg2, expected_result].
                   std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto arg : args) {
      state_.cpu.insn_addr = bit_cast<GuestAddr>(&insn_bytes);
      SetXReg<2>(state_.cpu, std::get<0>(arg));
      SetXReg<3>(state_.cpu, std::get<1>(arg));
      InterpretInsn(&state_);
      EXPECT_EQ(GetXReg<1>(state_.cpu), std::get<2>(arg));
    }
  }

  void InterpretLoad(uint32_t insn_bytes,
                     uint64_t expected_result) {
    state_.cpu.insn_addr = bit_cast<GuestAddr>(&insn_bytes);
    // Offset is always 8.
    SetXReg<2>(state_.cpu, bit_cast<uint64_t>(bit_cast<uint8_t*>(&kDataToLoad) - 8));
    InterpretInsn(&state_);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

 protected:
  static constexpr uint64_t kDataToLoad{0xffffeeeeddddccccULL};
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
  // Slr
  InterpretOp(0x003150b3, {{0xf000'0000'0000'0000ULL, 12, 0x000f'0000'0000'0000ULL}});
  // Sla
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
}

TEST_F(Riscv64InterpreterTest, LoadInstructions) {
  // Offset is always 8.
  // Ld
  InterpretLoad(0x00813083, kDataToLoad);
}

}  // namespace

}  // namespace berberis
