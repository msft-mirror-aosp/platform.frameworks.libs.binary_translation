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

#include <cstdint>
#include <initializer_list>
#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_riscv64.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/test_utils/scoped_exec_region.h"
#include "berberis/test_utils/testing_run_generated_code.h"

#include "riscv64_to_x86_64/lite_translator.h"

namespace berberis {

namespace {

template <uint8_t kInsnSize = 4>
bool RunOneInstruction(ThreadState* state, GuestAddr stop_pc) {
  MachineCode machine_code;
  bool success =
      LiteTranslateRange(state->cpu.insn_addr, state->cpu.insn_addr + kInsnSize, &machine_code);

  if (!success) {
    return false;
  }

  ScopedExecRegion exec(&machine_code);

  TestingRunGeneratedCode(state, exec.get(), stop_pc);
  return true;
}

// TODO(b/277619887): Share tests with the interpreter.
class Riscv64LiteTranslateInsnTest : public ::testing::Test {
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

 private:
  ThreadState state_;
};

TEST_F(Riscv64LiteTranslateInsnTest, OpInstructions) {
  // Add
  TestOp(0x003100b3, {{19, 23, 42}});
}

}  // namespace

}  // namespace berberis
