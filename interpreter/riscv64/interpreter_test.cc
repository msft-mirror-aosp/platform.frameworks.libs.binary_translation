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

}  // namespace

}  // namespace berberis
