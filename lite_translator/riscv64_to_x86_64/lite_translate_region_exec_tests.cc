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

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/test_utils/scoped_exec_region.h"
#include "berberis/test_utils/testing_run_generated_code.h"

namespace berberis {

namespace {

class Riscv64LiteTranslateRegionTest : public ::testing::Test {
 public:
  void Reset(const uint32_t code[]) { state_.cpu.insn_addr = ToGuestAddr(code); }

  template <typename T>
  bool Run(T code, GuestAddr expected_stop_addr) {
    Reset(code);
    GuestAddr code_end = ToGuestAddr(bit_cast<char*>(&code[0]) + sizeof(code));
    MachineCode machine_code;
    bool success = LiteTranslateRange(state_.cpu.insn_addr,
                                      code_end,
                                      &machine_code,
                                      LiteTranslateParams{.allow_dispatch = false});

    if (!success) {
      return false;
    }

    ScopedExecRegion exec(&machine_code);

    TestingRunGeneratedCode(&state_, exec.get(), expected_stop_addr);

    return state_.cpu.insn_addr == expected_stop_addr;
  }

 protected:
  ThreadState state_;
};

TEST_F(Riscv64LiteTranslateRegionTest, AddTwice) {
  static const uint32_t code[] = {
      0x003100b3,  // add x1, x2, x3
      0x002081b3,  // add x3, x1, x2
  };
  SetXReg<1>(state_.cpu, 0);
  SetXReg<2>(state_.cpu, 1);
  SetXReg<3>(state_.cpu, 1);
  EXPECT_TRUE(Run(code, ToGuestAddr(bit_cast<char*>(&code[0]) + sizeof(code))));
  EXPECT_EQ(GetXReg<3>(state_.cpu), 3ULL);
}

TEST_F(Riscv64LiteTranslateRegionTest, RegionEnd) {
  static const uint32_t code[] = {
      0x003100b3,  // add x1, x2, x3
      0x002081b3,  // add x3, x1, x2
      0x008000ef,  // jal x1, 8
      0x003100b3,  // add x1, x2, x3
      0x002081b3,  // add x3, x1, x2
  };
  SetXReg<1>(state_.cpu, 0);
  SetXReg<2>(state_.cpu, 1);
  SetXReg<3>(state_.cpu, 1);
  EXPECT_TRUE(Run(code, ToGuestAddr(code) + 8));
  EXPECT_EQ(GetXReg<3>(state_.cpu), 3ULL);
}

TEST_F(Riscv64LiteTranslateRegionTest, GracefulFailure) {
  static const uint32_t code[] = {
      0x003100b3,  // add x1, x2, x3
      0x00000073,  // ecall #0x0
  };
  MachineCode machine_code;
  EXPECT_FALSE(LiteTranslateRange(ToGuestAddr(code),
                                  ToGuestAddr(code) + 8,
                                  &machine_code,
                                  LiteTranslateParams{.allow_dispatch = false}));
}

}  // namespace

}  // namespace berberis