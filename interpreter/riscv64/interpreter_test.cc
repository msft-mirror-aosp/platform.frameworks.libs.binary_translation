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
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/riscv64/interpreter.h"
#include "berberis/intrinsics/guest_rounding_modes.h"  // ScopedRoundingMode

#include "fp_regs_util.h"
#include "tuple_map.h"

namespace berberis {

namespace {

class Riscv64InterpreterTest : public ::testing::Test {
 public:
  // Compressed Instructions.

  template <RegisterType register_type, uint64_t expected_result, uint8_t kTargetReg>
  void InterpretCompressedStore(uint16_t insn_bytes, uint64_t offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    store_area_ = 0;
    SetXReg<kTargetReg>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - offset));
    SetReg<register_type, 9>(state_.cpu, kDataToLoad);
    InterpretInsn(&state_);
    EXPECT_EQ(store_area_, expected_result);
  }

  template <RegisterType register_type, uint64_t expected_result, uint8_t kSourceReg>
  void InterpretCompressedLoad(uint16_t insn_bytes, uint64_t offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<kSourceReg>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&kDataToLoad) - offset));
    InterpretInsn(&state_);
    EXPECT_EQ((GetReg<register_type, 9>(state_.cpu)), expected_result);
  }

  void InterpretCAddi16sp(uint16_t insn_bytes, uint64_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<2>(state_.cpu, 1);
    InterpretInsn(&state_);
    EXPECT_EQ(GetXReg<2>(state_.cpu), 1 + expected_offset);
  }

  void InterpretCAddi4spn(uint16_t insn_bytes, uint64_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<2>(state_.cpu, 1);
    InterpretInsn(&state_);
    EXPECT_EQ(GetXReg<9>(state_.cpu), 1 + expected_offset);
  }

  void InterpretCAddi(uint16_t insn_bytes, uint64_t expected_increment) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<2>(state_.cpu, 1);
    InterpretInsn(&state_);
    EXPECT_EQ(GetXReg<2>(state_.cpu), 1 + expected_increment);
  }

  void InterpretCBeqzBnez(uint16_t insn_bytes, uint64_t value, int16_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<9>(state_.cpu, value);
    InterpretInsn(&state_);
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
  }

  void InterpretCMiscAluImm(uint16_t insn_bytes, uint64_t value, uint64_t expected_result) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<9>(state_.cpu, value);
    InterpretInsn(&state_);
    EXPECT_EQ(GetXReg<9>(state_.cpu), expected_result);
  }

  void InterpretCMiscAlu(uint16_t insn_bytes,
                         std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto [arg1, arg2, expected_result] : args) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetXReg<8>(state_.cpu, arg1);
      SetXReg<9>(state_.cpu, arg2);
      InterpretInsn(&state_);
      EXPECT_EQ(GetXReg<8>(state_.cpu), expected_result);
    }
  }

  void InterpretCJ(uint16_t insn_bytes, int16_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    InterpretInsn(&state_);
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
  }

  void InterpretCOp(uint32_t insn_bytes,
                    std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto [arg1, arg2, expected_result] : args) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetXReg<1>(state_.cpu, arg1);
      SetXReg<2>(state_.cpu, arg2);
      InterpretInsn(&state_);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  // Non-Compressed Instructions.

  void InterpretCsr(uint32_t insn_bytes, uint8_t expected_rm) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    state_.cpu.frm = 0b001u;
    InterpretInsn(&state_);
    EXPECT_EQ(GetXReg<2>(state_.cpu), 0b001u);
    EXPECT_EQ(state_.cpu.frm, expected_rm);
  }

  template <typename... Types>
  void InterpretFma(uint32_t insn_bytes, std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg1, arg2, arg3, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetFReg<2>(state_.cpu, arg1);
      SetFReg<3>(state_.cpu, arg2);
      SetFReg<4>(state_.cpu, arg3);
      InterpretInsn(&state_);
      EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
    }
  }

  template <typename... Types>
  void InterpretOpFp(uint32_t insn_bytes, std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg1, arg2, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetFReg<2>(state_.cpu, arg1);
      SetFReg<3>(state_.cpu, arg2);
      InterpretInsn(&state_);
      EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
    }
  }

  template <typename... Types>
  void InterpretFmvFloatToInteger(uint32_t insn_bytes,
                                  std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetFReg<1>(state_.cpu, arg);
      InterpretInsn(&state_);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  template <typename... Types>
  void InterpretFmvIntegerToFloat(uint32_t insn_bytes,
                                  std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg, expected_result] : args) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetXReg<1>(state_.cpu, arg);
      InterpretInsn(&state_);
      EXPECT_EQ(GetFReg<1>(state_.cpu), kFPValueToFPReg(expected_result));
    }
  }

  template <typename... Types>
  void InterpretOpFpGpRegisterTarget(uint32_t insn_bytes,
                                     std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg1, arg2, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetFReg<2>(state_.cpu, arg1);
      SetFReg<3>(state_.cpu, arg2);
      InterpretInsn(&state_);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  template <typename... Types>
  void InterpretOpFpGpRegisterTargetSingleInput(uint32_t insn_bytes,
                                                std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetFReg<2>(state_.cpu, arg);
      InterpretInsn(&state_);
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  template <typename... Types>
  void InterpretOpFpGpRegisterSourceSingleInput(uint32_t insn_bytes,
                                                std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetXReg<2>(state_.cpu, arg);
      InterpretInsn(&state_);
      EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
    }
  }

  template <typename... Types>
  void InterpretOpFpSingleInput(uint32_t insn_bytes,
                                std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetFReg<2>(state_.cpu, arg);
      InterpretInsn(&state_);
      EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
    }
  }

  void InterpretFence(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    InterpretInsn(&state_);
  }

  void InterpretAmo(uint32_t insn_bytes, uint64_t arg1, uint64_t arg2, uint64_t expected_result,
                    uint64_t expected_memory) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    // Copy arg1 into store_area_
    store_area_ = arg1;
    SetXReg<2>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_)));
    SetXReg<3>(state_.cpu, arg2);
    InterpretInsn(&state_);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    EXPECT_EQ(store_area_, expected_memory);
  }

  void InterpretAmo(uint32_t insn_bytes32, uint32_t insn_bytes64, uint64_t expected_memory) {
    InterpretAmo(insn_bytes32, 0xffff'eeee'dddd'ccccULL, 0xaaaa'bbbb'cccc'ddddULL,
                 0xffff'ffff'dddd'ccccULL, 0xffff'eeee'0000'0000 | uint32_t(expected_memory));
    InterpretAmo(insn_bytes64, 0xffff'eeee'dddd'ccccULL, 0xaaaa'bbbb'cccc'ddddULL,
                 0xffff'eeee'dddd'ccccULL, expected_memory);
  }

  void InterpretLi(uint32_t insn_bytes, uint64_t expected_result) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    InterpretInsn(&state_);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  void InterpretLoadFp(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    // Offset is always 8.
    SetXReg<2>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&kDataToLoad) - 8));
    InterpretInsn(&state_);
    EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
  }

  void InterpretAtomicLoad(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&kDataToLoad));
    InterpretInsn(&state_);
    EXPECT_EQ(GetXReg<2>(state_.cpu), expected_result);
  }

  void InterpretStoreFp(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    // Offset is always 8.
    SetXReg<1>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - 8));
    SetFReg<2>(state_.cpu, kDataToStore);
    store_area_ = 0;
    InterpretInsn(&state_);
    EXPECT_EQ(store_area_, expected_result);
  }

  void InterpretAtomicStore(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    SetXReg<2>(state_.cpu, kDataToStore);
    SetXReg<3>(state_.cpu, 0xdeadbeef);
    store_area_ = 0;
    InterpretInsn(&state_);
    EXPECT_EQ(store_area_, expected_result);
    EXPECT_EQ(GetXReg<3>(state_.cpu), 0u);
  }

  // kLinkRegisterOffsetIfUsed is size of instruction or 0 if instruction does not link register.
  template <uint8_t kLinkRegisterOffsetIfUsed>
  void InterpretJumpAndLinkRegister(uint32_t insn_bytes,
                                    uint64_t base_disp,
                                    int64_t expected_offset) {
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

 protected:
  static constexpr uint64_t kDataToLoad{0xffffeeeeddddccccULL};
  static constexpr uint64_t kDataToStore = kDataToLoad;
  uint64_t store_area_;
  ThreadState state_;
};

bool RunOneInstruction(ThreadState* state, GuestAddr stop_pc) {
  InterpretInsn(state);
  return state->cpu.insn_addr == stop_pc;
}

#define TESTSUITE Riscv64InterpretInsnTest

#include "berberis/test_utils/insn_tests_riscv64-inl.h"

#undef TESTSUITE

// Tests for Compressed Instructions.

template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedLoadOrStore64bit(Riscv64InterpreterTest* that) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 3;
      uint8_t i3_i5 : 3;
      uint8_t i6_i7 : 2;
    } i_bits;
  };
  for (offset = int16_t{0}; offset < int16_t{256}; offset += 8) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t rd : 3;
        uint8_t i6_i7 : 2;
        uint8_t rs : 3;
        uint8_t i3_i5 : 3;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b00,
        .rd = 1,
        .i6_i7 = i_bits.i6_i7,
        .rs = 0,
        .i3_i5 = i_bits.i3_i5,
        .high_opcode = 0b000,
    };
    (that->*execute_instruction_func)(o_bits.parcel | opcode, offset);
  }
}

TEST_F(Riscv64InterpreterTest, CompressedLoadAndStores) {
  // c.Fld
  TestCompressedLoadOrStore64bit<
      0b001'000'000'00'000'00,
      &Riscv64InterpreterTest::InterpretCompressedLoad<RegisterType::kFpReg, kDataToLoad, 8>>(this);
  // c.Ld
  TestCompressedLoadOrStore64bit<
      0b011'000'000'00'000'00,
      &Riscv64InterpreterTest::InterpretCompressedLoad<RegisterType::kReg, kDataToLoad, 8>>(this);
  // c.Fsd
  TestCompressedLoadOrStore64bit<
      0b101'000'000'00'000'00,
      &Riscv64InterpreterTest::InterpretCompressedStore<RegisterType::kFpReg, kDataToLoad, 8>>(
      this);
  // c.Sd
  TestCompressedLoadOrStore64bit<
      0b111'000'000'00'000'00,
      &Riscv64InterpreterTest::InterpretCompressedStore<RegisterType::kReg, kDataToLoad, 8>>(this);
}

TEST_F(Riscv64InterpreterTest, TestCompressedStore32bitsp) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 2;
      uint8_t i2_i5 : 4;
      uint8_t i6_i7 : 2;
    } i_bits;
  };
  for (offset = uint16_t{0}; offset < uint16_t{256}; offset += 4) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t rs2 : 5;
        uint8_t i6_i7 : 2;
        uint8_t i2_i5 : 4;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b10,
        .rs2 = 9,
        .i6_i7 = i_bits.i6_i7,
        .i2_i5 = i_bits.i2_i5,
        .high_opcode = 0b110,
    };
    // c.Swsp
    InterpretCompressedStore<RegisterType::kReg,
                             static_cast<uint64_t>(static_cast<uint32_t>(kDataToStore)),
                             2>(o_bits.parcel, offset);
  }
}

template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedStore64bitsp(Riscv64InterpreterTest* that) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 3;
      uint8_t i3_i5 : 3;
      uint8_t i6_i8 : 3;
    } i_bits;
  };
  for (offset = uint16_t{0}; offset < uint16_t{512}; offset += 8) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t rs2 : 5;
        uint8_t i6_i8 : 3;
        uint8_t i3_i5 : 3;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b10,
        .rs2 = 9,
        .i6_i8 = i_bits.i6_i8,
        .i3_i5 = i_bits.i3_i5,
        .high_opcode = 0b101,
    };
    (that->*execute_instruction_func)(o_bits.parcel | opcode, offset);
  }
}

TEST_F(Riscv64InterpreterTest, TestCompressedStore64bitsp) {
  // c.Fsdsp
  TestCompressedStore64bitsp<
      0b001'000'000'00'000'00,
      &Riscv64InterpreterTest::InterpretCompressedStore<RegisterType::kFpReg, kDataToStore, 2>>(
      this);
  // c.Sdsp
  TestCompressedStore64bitsp<
      0b011'000'000'00'000'00,
      &Riscv64InterpreterTest::InterpretCompressedStore<RegisterType::kReg, kDataToStore, 2>>(this);
}

TEST_F(Riscv64InterpreterTest, TestCompressedLoad32bitsp) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 2;
      uint8_t i2_i4 : 3;
      uint8_t i5 : 1;
      uint8_t i6_i7 : 2;
    } i_bits;
  };
  for (offset = uint16_t{0}; offset < uint16_t{256}; offset += 4) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i6_i7 : 2;
        uint8_t i2_i4 : 3;
        uint8_t rd : 5;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b10,
        .i6_i7 = i_bits.i6_i7,
        .i2_i4 = i_bits.i2_i4,
        .rd = 9,
        .i5 = i_bits.i5,
        .high_opcode = 0b010,
    };
    // c.Lwsp
    InterpretCompressedLoad<RegisterType::kReg,
                            static_cast<uint64_t>(static_cast<int32_t>(kDataToLoad)),
                            2>(o_bits.parcel, offset);
  }
}

template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedLoad64bitsp(Riscv64InterpreterTest* that) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 3;
      uint8_t i3_i4 : 2;
      uint8_t i5 : 1;
      uint8_t i6_i8 : 3;
    } i_bits;
  };
  for (offset = uint16_t{0}; offset < uint16_t{512}; offset += 8) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i6_i8 : 3;
        uint8_t i3_i4 : 2;
        uint8_t rd : 5;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b10,
        .i6_i8 = i_bits.i6_i8,
        .i3_i4 = i_bits.i3_i4,
        .rd = 9,
        .i5 = i_bits.i5,
        .high_opcode = 0b001,
    };
    (that->*execute_instruction_func)(o_bits.parcel | opcode, offset);
  }
}

TEST_F(Riscv64InterpreterTest, TestCompressedLoad64bitsp) {
  // c.Fldsp
  TestCompressedLoad64bitsp<
      0b001'000'000'00'000'00,
      &Riscv64InterpreterTest::InterpretCompressedLoad<RegisterType::kFpReg, kDataToLoad, 2>>(this);
  // c.Ldsp
  TestCompressedLoad64bitsp<
      0b011'000'000'00'000'00,
      &Riscv64InterpreterTest::InterpretCompressedLoad<RegisterType::kReg, kDataToLoad, 2>>(this);
}

TEST_F(Riscv64InterpreterTest, CAddi) {
  union {
    int8_t offset;
    struct [[gnu::packed]] {
      uint8_t i4_i0 : 5;
      uint8_t i5 : 1;
    } i_bits;
  };
  for (offset = int8_t{-32}; offset < int8_t{31}; offset++) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i4_i0 : 5;
        uint8_t r : 5;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      } __attribute__((__packed__));
    } o_bits = {
        .low_opcode = 0,
        .i4_i0 = i_bits.i4_i0,
        .r = 2,
        .i5 = i_bits.i5,
        .high_opcode = 0,
    };
    // c.Addi
    InterpretCAddi(o_bits.parcel | 0b0000'0000'0000'0001, offset);
    // c.Addiw
    InterpretCAddi(o_bits.parcel | 0b0010'0000'0000'0001, offset);
  }
}

TEST_F(Riscv64InterpreterTest, CAddi16sp) {
  union {
    int16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 4;
      uint8_t i4 : 1;
      uint8_t i5 : 1;
      uint8_t i6 : 1;
      uint8_t i7 : 1;
      uint8_t i8 : 1;
      uint8_t i9 : 1;
    } i_bits;
  };
  for (offset = int16_t{-512}; offset < int16_t{512}; offset += 16) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i5 : 1;
        uint8_t i7 : 1;
        uint8_t i8 : 1;
        uint8_t i6 : 1;
        uint8_t i4 : 1;
        uint8_t rd : 5;
        uint8_t i9 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b01,
        .i5 = i_bits.i5,
        .i7 = i_bits.i7,
        .i8 = i_bits.i8,
        .i6 = i_bits.i6,
        .i4 = i_bits.i4,
        .rd = 2,
        .i9 = i_bits.i9,
        .high_opcode = 0b011,
    };
    InterpretCAddi16sp(o_bits.parcel, offset);
  }
}

TEST_F(Riscv64InterpreterTest, CLui) {
  union {
    int32_t offset;
    struct [[gnu::packed]] {
      uint8_t : 12;
      uint8_t i12_i16 : 5;
      uint8_t i17 : 1;
    } i_bits;
  };
  for (offset = int32_t{-131072}; offset < int32_t{131072}; offset += 4096) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i12_i16 : 5;
        uint8_t rd : 5;
        uint8_t i17 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b01,
        .i12_i16 = i_bits.i12_i16,
        .rd = 1,
        .i17 = i_bits.i17,
        .high_opcode = 0b011,
    };
    InterpretLi(o_bits.parcel, offset);
  }
}

TEST_F(Riscv64InterpreterTest, CLi) {
  union {
    int8_t offset;
    struct [[gnu::packed]] {
      uint8_t i0_i4 : 5;
      uint8_t i5 : 1;
    } i_bits;
  };
  for (offset = int8_t{-32}; offset < int8_t{32}; offset++) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i0_i4 : 5;
        uint8_t rd : 5;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b01,
        .i0_i4 = i_bits.i0_i4,
        .rd = 1,
        .i5 = i_bits.i5,
        .high_opcode = 0b010,
    };
    InterpretLi(o_bits.parcel, offset);
  }
}

TEST_F(Riscv64InterpreterTest, CAddi4spn) {
  union {
    int16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 2;
      uint8_t i2 : 1;
      uint8_t i3 : 1;
      uint8_t i4 : 1;
      uint8_t i5 : 1;
      uint8_t i6 : 1;
      uint8_t i7 : 1;
      uint8_t i8 : 1;
      uint8_t i9 : 1;
    } i_bits;
  };
  for (offset = int16_t{4}; offset < int16_t{1024}; offset += 4) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t rd : 3;
        uint8_t i3 : 1;
        uint8_t i2 : 1;
        uint8_t i6 : 1;
        uint8_t i7 : 1;
        uint8_t i8 : 1;
        uint8_t i9 : 1;
        uint8_t i4 : 1;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b00,
        .rd = 1,
        .i3 = i_bits.i3,
        .i2 = i_bits.i2,
        .i6 = i_bits.i6,
        .i7 = i_bits.i7,
        .i8 = i_bits.i8,
        .i9 = i_bits.i9,
        .i4 = i_bits.i4,
        .i5 = i_bits.i5,
        .high_opcode = 0b000,
    };
    InterpretCAddi4spn(o_bits.parcel, offset);
  }
}

TEST_F(Riscv64InterpreterTest, CBeqzBnez) {
  union {
    int16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 1;
      uint8_t i1 : 1;
      uint8_t i2 : 1;
      uint8_t i3 : 1;
      uint8_t i4 : 1;
      uint8_t i5 : 1;
      uint8_t i6 : 1;
      uint8_t i7 : 1;
      uint8_t i8 : 1;
    } i_bits;
  };
  for (offset = int16_t{-256}; offset < int16_t{256}; offset += 8) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i5 : 1;
        uint8_t i1 : 1;
        uint8_t i2 : 1;
        uint8_t i6 : 1;
        uint8_t i7 : 1;
        uint8_t rs : 3;
        uint8_t i3 : 1;
        uint8_t i4 : 1;
        uint8_t i8 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0,
        .i5 = i_bits.i5,
        .i1 = i_bits.i1,
        .i2 = i_bits.i2,
        .i6 = i_bits.i6,
        .i7 = i_bits.i7,
        .rs = 1,
        .i3 = i_bits.i3,
        .i4 = i_bits.i4,
        .i8 = i_bits.i8,
        .high_opcode = 0,
    };
    InterpretCBeqzBnez(o_bits.parcel | 0b1100'0000'0000'0001, 0, offset);
    InterpretCBeqzBnez(o_bits.parcel | 0b1110'0000'0000'0001, 1, offset);
  }
}

TEST_F(Riscv64InterpreterTest, CMiscAluInstructions) {
  // c.Sub
  InterpretCMiscAlu(0x8c05, {{42, 23, 19}});
  // c.Xor
  InterpretCMiscAlu(0x8c25, {{0b0101, 0b0011, 0b0110}});
  // c.Or
  InterpretCMiscAlu(0x8c45, {{0b0101, 0b0011, 0b0111}});
  // c.And
  InterpretCMiscAlu(0x8c65, {{0b0101, 0b0011, 0b0001}});
  // c.SubW
  InterpretCMiscAlu(0x9c05, {{42, 23, 19}});
  // c.AddW
  InterpretCMiscAlu(0x9c25, {{19, 23, 42}});
}

TEST_F(Riscv64InterpreterTest, CMiscAluImm) {
  union {
    uint8_t uimm;
    // Note: c.Andi uses sign-extended immediate while c.Srli/c.cSrain need zero-extended one.
    // If we store the value into uimm and read from imm compiler would do correct conversion.
    int8_t imm : 6;
    struct [[gnu::packed]] {
      uint8_t i0_i4 : 5;
      uint8_t i5 : 1;
    } i_bits;
  };
  for (uimm = uint8_t{0}; uimm < uint8_t{64}; uimm++) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i0_i4 : 5;
        uint8_t r : 3;
        uint8_t mid_opcode : 2;
        uint8_t i5 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0,
        .i0_i4 = i_bits.i0_i4,
        .r = 1,
        .mid_opcode = 0,
        .i5 = i_bits.i5,
        .high_opcode = 0,
    };
    // The o_bits.parcel here doesn't include opcodes and we are adding it in the function call.
    // c.Srli
    InterpretCMiscAluImm(o_bits.parcel | 0b1000'0000'0000'0001,
                         0x8000'0000'0000'0000ULL,
                         0x8000'0000'0000'0000ULL >> uimm);
    // c.Srai
    InterpretCMiscAluImm(o_bits.parcel | 0b1000'0100'0000'0001,
                         0x8000'0000'0000'0000LL,
                         ~0 ^ ((0x8000'0000'0000'0000 ^ ~0) >>
                               uimm));  // Avoid shifting negative numbers to avoid UB
    // c.Andi
    InterpretCMiscAluImm(o_bits.parcel | 0b1000'1000'0000'0001,
                         0xffff'ffff'ffff'ffffULL,
                         0xffff'ffff'ffff'ffffULL & imm);

    // Previous instructions use 3-bit register encoding where 0b000 is r8, 0b001 is r9, etc.
    // c.Slli uses 5-bit register encoding. Since we want it to also work with r9 in the test body
    // we add 0b01000 to register bits to mimic that shift-by-8.
    // c.Slli                                   vvvvvv adds 8 to r to handle rd' vs rd difference.
    InterpretCMiscAluImm(o_bits.parcel | 0b0000'0100'0000'0010,
                         0x0000'0000'0000'0001ULL,
                         0x0000'0000'0000'0001ULL << uimm);
  }
}

TEST_F(Riscv64InterpreterTest, CJ) {
  union {
    int16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 1;
      uint8_t i1 : 1;
      uint8_t i2 : 1;
      uint8_t i3 : 1;
      uint8_t i4 : 1;
      uint8_t i5 : 1;
      uint8_t i6 : 1;
      uint8_t i7 : 1;
      uint8_t i8 : 1;
      uint8_t i9 : 1;
      uint8_t i10 : 1;
      uint8_t i11 : 1;
    } i_bits;
  };
  for (offset = int16_t{-2048}; offset < int16_t{2048}; offset += 2) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
        uint8_t low_opcode : 2;
        uint8_t i5 : 1;
        uint8_t i1 : 1;
        uint8_t i2 : 1;
        uint8_t i3 : 1;
        uint8_t i7 : 1;
        uint8_t i6 : 1;
        uint8_t i10 : 1;
        uint8_t i8 : 1;
        uint8_t i9 : 1;
        uint8_t i4 : 1;
        uint8_t i11 : 1;
        uint8_t high_opcode : 3;
      };
    } o_bits = {
        .low_opcode = 0b01,
        .i5 = i_bits.i5,
        .i1 = i_bits.i1,
        .i2 = i_bits.i2,
        .i3 = i_bits.i3,
        .i7 = i_bits.i7,
        .i6 = i_bits.i6,
        .i10 = i_bits.i10,
        .i8 = i_bits.i8,
        .i9 = i_bits.i9,
        .i4 = i_bits.i4,
        .i11 = i_bits.i11,
        .high_opcode = 0b101,
    };
    InterpretCJ(o_bits.parcel, offset);
  }
}

TEST_F(Riscv64InterpreterTest, CJalr) {
  // C.Jr
  InterpretJumpAndLinkRegister<0>(0x8102, 42, 42);
  // C.Mv
  InterpretCOp(0x808a, {{0, 1, 1}});
  // C.Jalr
  InterpretJumpAndLinkRegister<2>(0x9102, 42, 42);
  // C.Add
  InterpretCOp(0x908a, {{1, 2, 3}});
}

// Tests for Non-Compressed Instructions.

TEST_F(Riscv64InterpreterTest, CsrInstructions) {
  ScopedRoundingMode scoped_rounding_mode;
  // Csrrw x2, frm, 2
  InterpretCsr(0x00215173, 2);
  // Csrrsi x2, frm, 2
  InterpretCsr(0x00216173, 3);
  // Csrrci x2, frm, 1
  InterpretCsr(0x0020f173, 0);
}

TEST_F(Riscv64InterpreterTest, FenceInstructions) {
  // Fence
  InterpretFence(0x0ff0000f);
  // FenceTso
  InterpretFence(0x8330000f);
  // FenceI
  InterpretFence(0x0000100f);
}

TEST_F(Riscv64InterpreterTest, FmaInstructions) {
  // Fmadd.S
  InterpretFma(0x203170c3, {std::tuple{1.0f, 2.0f, 3.0f, 5.0f}});
  // Fmadd.D
  InterpretFma(0x223170c3, {std::tuple{1.0, 2.0, 3.0, 5.0}});
  // Fmsub.S
  InterpretFma(0x203170c7, {std::tuple{1.0f, 2.0f, 3.0f, -1.0f}});
  // Fmsub.D
  InterpretFma(0x223170c7, {std::tuple{1.0, 2.0, 3.0, -1.0}});
  // Fnmsub.S
  InterpretFma(0x203170cb, {std::tuple{1.0f, 2.0f, 3.0f, 1.0f}});
  // Fnmsub.D
  InterpretFma(0x223170cb, {std::tuple{1.0, 2.0, 3.0, 1.0}});
  // Fnmadd.S
  InterpretFma(0x203170cf, {std::tuple{1.0f, 2.0f, 3.0f, -5.0f}});
  // Fnmadd.D
  InterpretFma(0x223170cf, {std::tuple{1.0, 2.0, 3.0, -5.0}});
}

TEST_F(Riscv64InterpreterTest, AmoInstructions) {
  // Verifying that all aq and rl combinations work for Amoswap, but only test relaxed one for most
  // other instructions for brevity.

  // AmoswaoW/AmoswaoD
  InterpretAmo(0x083120af, 0x083130af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoswapWAq/AmoswapDAq
  InterpretAmo(0x0c3120af, 0x0c3130af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoswapWRl/AmoswapDRl
  InterpretAmo(0x0a3120af, 0x0a3130af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoswapWAqrl/AmoswapDAqrl
  InterpretAmo(0x0e3120af, 0x0e3130af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmoaddW/AmoaddD
  InterpretAmo(0x003120af, 0x003130af, 0xaaaa'aaaa'aaaa'aaa9);

  // AmoxorW/AmoxorD
  InterpretAmo(0x203120af, 0x203130af, 0x5555'5555'1111'1111);

  // AmoandW/AmoandD
  InterpretAmo(0x603120af, 0x603130af, 0xaaaa'aaaa'cccc'cccc);

  // AmoorW/AmoorD
  InterpretAmo(0x403120af, 0x403130af, 0xffff'ffff'dddd'dddd);

  // AmominW/AmominD
  InterpretAmo(0x803120af, 0x803130af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmomaxW/AmomaxD
  InterpretAmo(0xa03120af, 0xa03130af, 0xffff'eeee'dddd'ccccULL);

  // AmominuW/AmominuD
  InterpretAmo(0xc03120af, 0xc03130af, 0xaaaa'bbbb'cccc'ddddULL);

  // AmomaxuW/AmomaxuD
  InterpretAmo(0xe03120af, 0xe03130af, 0xffff'eeee'dddd'ccccULL);
}

TEST_F(Riscv64InterpreterTest, OpFpInstructions) {
  // FAdd.S
  InterpretOpFp(0x003100d3, {std::tuple{1.0f, 2.0f, 3.0f}});
  // FAdd.D
  InterpretOpFp(0x023100d3, {std::tuple{1.0, 2.0, 3.0}});
  // FSub.S
  InterpretOpFp(0x083100d3, {std::tuple{3.0f, 2.0f, 1.0f}});
  // FSub.D
  InterpretOpFp(0x0a3100d3, {std::tuple{3.0, 2.0, 1.0}});
  // FMul.S
  InterpretOpFp(0x103100d3, {std::tuple{3.0f, 2.0f, 6.0f}});
  // FMul.D
  InterpretOpFp(0x123100d3, {std::tuple{3.0, 2.0, 6.0}});
  // FDiv.S
  InterpretOpFp(0x183100d3, {std::tuple{6.0f, 2.0f, 3.0f}});
  // FDiv.D
  InterpretOpFp(0x1a3100d3, {std::tuple{6.0, 2.0, 3.0}});

  // FSgnj.S
  InterpretOpFp(0x203100d3,
                {std::tuple{1.0f, 2.0f, 1.0f},
                 {-1.0f, 2.0f, 1.0f},
                 {1.0f, -2.0f, -1.0f},
                 {-1.0f, -2.0f, -1.0f}});
  // FSgnj.D
  InterpretOpFp(0x223100d3,
                {
                    std::tuple{1.0, 2.0, 1.0},
                    {-1.0, 2.0, 1.0},
                    {1.0, -2.0, -1.0},
                    {-1.0, -2.0, -1.0},
                });
  // FSgnjn.S
  InterpretOpFp(0x203110d3,
                {
                    std::tuple{1.0f, 2.0f, -1.0f},
                    {1.0f, 2.0f, -1.0f},
                    {1.0f, -2.0f, 1.0f},
                    {-1.0f, -2.0f, 1.0f},
                });
  // FSgnjn.D
  InterpretOpFp(0x223110d3,
                {
                    std::tuple{1.0, 2.0, -1.0},
                    {1.0, 2.0, -1.0},
                    {1.0, -2.0, 1.0},
                    {-1.0, -2.0, 1.0},
                });
  // FSgnjx.S
  InterpretOpFp(0x203120d3,
                {
                    std::tuple{1.0f, 2.0f, 1.0f},
                    {-1.0f, 2.0f, -1.0f},
                    {1.0f, -2.0f, -1.0f},
                    {-1.0f, -2.0f, 1.0f},
                });
  // FSgnjx.D
  InterpretOpFp(0x223120d3,
                {
                    std::tuple{1.0, 2.0, 1.0},
                    {-1.0, 2.0, -1.0},
                    {1.0, -2.0, -1.0},
                    {-1.0, -2.0, 1.0},
                });
  // FMin.S
  InterpretOpFp(0x283100d3,
                {std::tuple{+0.f, +0.f, +0.f},
                 {+0.f, -0.f, -0.f},
                 {-0.f, +0.f, -0.f},
                 {-0.f, -0.f, -0.f},
                 {+0.f, 1.f, +0.f},
                 {-0.f, 1.f, -0.f}});
  // FMin.D
  InterpretOpFp(0x2a3100d3,
                {std::tuple{+0.0, +0.0, +0.0},
                 {+0.0, -0.0, -0.0},
                 {-0.0, +0.0, -0.0},
                 {-0.0, -0.0, -0.0},
                 {+0.0, 1.0, +0.0},
                 {-0.0, 1.0, -0.0}});
  // FMax.S
  InterpretOpFp(0x283110d3,
                {std::tuple{+0.f, +0.f, +0.f},
                 {+0.f, -0.f, +0.f},
                 {-0.f, +0.f, +0.f},
                 {-0.f, -0.f, -0.f},
                 {+0.f, 1.f, 1.f},
                 {-0.f, 1.f, 1.f}});
  // FMax.D
  InterpretOpFp(0x2a3110d3,
                {std::tuple{+0.0, +0.0, +0.0},
                 {+0.0, -0.0, +0.0},
                 {-0.0, +0.0, +0.0},
                 {-0.0, -0.0, -0.0},
                 {+0.0, 1.0, 1.0},
                 {-0.0, 1.0, 1.0}});
}

TEST_F(Riscv64InterpreterTest, OpFpSingleInputInstructions) {
  // FSqrt.S
  InterpretOpFpSingleInput(0x580170d3, {std::tuple{4.0f, 2.0f}});
  // FSqrt.D
  InterpretOpFpSingleInput(0x5a0170d3, {std::tuple{16.0, 4.0}});
}

TEST_F(Riscv64InterpreterTest, Fmv) {
  // Fmv.X.W
  InterpretFmvFloatToInteger(0xe00080d3,
                             {std::tuple{1.0f, static_cast<uint64_t>(bit_cast<uint32_t>(1.0f))},
                              {-1.0f, static_cast<int64_t>(bit_cast<int32_t>(-1.0f))}});
  // Fmv.W.X
  InterpretFmvIntegerToFloat(
      0xf00080d3, {std::tuple{bit_cast<uint32_t>(1.0f), 1.0f}, {bit_cast<uint32_t>(-1.0f), -1.0f}});
  // Fmv.X.D
  InterpretFmvFloatToInteger(
      0xe20080d3, {std::tuple{1.0, bit_cast<uint64_t>(1.0)}, {-1.0, bit_cast<uint64_t>(-1.0)}});
  // Fmv.D.X
  InterpretFmvIntegerToFloat(
      0xf20080d3, {std::tuple{bit_cast<uint64_t>(1.0), 1.0}, {bit_cast<uint64_t>(-1.0), -1.0}});
}

TEST_F(Riscv64InterpreterTest, OpFpFcvt) {
  // Fcvt.S.D
  InterpretOpFpSingleInput(0x401170d3, {std::tuple{1.0, 1.0f}});
  // Fcvt.D.S
  InterpretOpFpSingleInput(0x420100d3, {std::tuple{2.0f, 2.0}});
  // Fcvt.W.S
  InterpretOpFpGpRegisterTargetSingleInput(0xc00170d3, {std::tuple{3.0f, 3UL}});
  // Fcvt.WU.S
  InterpretOpFpGpRegisterTargetSingleInput(0xc01170d3, {std::tuple{3.0f, 3UL}});
  // Fcvt.L.S
  InterpretOpFpGpRegisterTargetSingleInput(0xc02170d3, {std::tuple{3.0f, 3UL}});
  // Fcvt.LU.S
  InterpretOpFpGpRegisterTargetSingleInput(0xc03170d3, {std::tuple{3.0f, 3UL}});
  // Fcvt.W.D
  InterpretOpFpGpRegisterTargetSingleInput(0xc20170d3, {std::tuple{3.0, 3UL}});
  // Fcvt.WU.D
  InterpretOpFpGpRegisterTargetSingleInput(0xc21170d3, {std::tuple{3.0, 3UL}});
  // Fcvt.L.D
  InterpretOpFpGpRegisterTargetSingleInput(0xc22170d3, {std::tuple{3.0, 3UL}});
  // Fcvt.LU.D
  InterpretOpFpGpRegisterTargetSingleInput(0xc23170d3, {std::tuple{3.0, 3UL}});
  // Fcvt.S.W
  InterpretOpFpGpRegisterSourceSingleInput(0xd00170d3, {std::tuple{3UL, 3.0f}});
  // Fcvt.S.WU
  InterpretOpFpGpRegisterSourceSingleInput(0xd01170d3, {std::tuple{3UL, 3.0f}});
  // Fcvt.S.L
  InterpretOpFpGpRegisterSourceSingleInput(0xd02170d3, {std::tuple{3UL, 3.0f}});
  // Fcvt.S.LU
  InterpretOpFpGpRegisterSourceSingleInput(0xd03170d3, {std::tuple{3UL, 3.0f}});
  // Fcvt.D.W
  InterpretOpFpGpRegisterSourceSingleInput(0xd20170d3, {std::tuple{3UL, 3.0}});
  // Fcvt.D.Wu
  InterpretOpFpGpRegisterSourceSingleInput(0xd21170d3, {std::tuple{3UL, 3.0}});
  // Fcvt.D.L
  InterpretOpFpGpRegisterSourceSingleInput(0xd22170d3, {std::tuple{3UL, 3.0}});
  // Fcvt.D.LU
  InterpretOpFpGpRegisterSourceSingleInput(0xd23170d3, {std::tuple{3UL, 3.0}});
}

TEST_F(Riscv64InterpreterTest, OpFpGpRegisterTargetInstructions) {
  // Fle.S
  InterpretOpFpGpRegisterTarget(
      0xa03100d3, {std::tuple{1.0f, 2.0f, 1UL}, {2.0f, 1.0f, 0UL}, {0.0f, 0.0f, 1UL}});
  // Fle.D
  InterpretOpFpGpRegisterTarget(0xa23100d3,
                                {std::tuple{1.0, 2.0, 1UL}, {2.0, 1.0, 0UL}, {0.0, 0.0, 1UL}});
  // Flt.S
  InterpretOpFpGpRegisterTarget(
      0xa03110d3, {std::tuple{1.0f, 2.0f, 1UL}, {2.0f, 1.0f, 0UL}, {0.0f, 0.0f, 0UL}});
  // Flt.D
  InterpretOpFpGpRegisterTarget(0xa23110d3,
                                {std::tuple{1.0, 2.0, 1UL}, {2.0, 1.0, 0UL}, {0.0, 0.0, 0UL}});
  // Feq.S
  InterpretOpFpGpRegisterTarget(
      0xa03120d3, {std::tuple{1.0f, 2.0f, 0UL}, {2.0f, 1.0f, 0UL}, {0.0f, 0.0f, 1UL}});
  // Feq.D
  InterpretOpFpGpRegisterTarget(0xa23120d3,
                                {std::tuple{1.0, 2.0, 0UL}, {2.0, 1.0, 0UL}, {0.0, 0.0, 1UL}});
}

TEST_F(Riscv64InterpreterTest, InterpretOpFpGpRegisterTargetSingleInput) {
  // Fclass.S
  InterpretOpFpGpRegisterTargetSingleInput(
      0xe00110d3,
      {std::tuple{-std::numeric_limits<float>::infinity(), 0b00'0000'0001UL},
       {-1.0f, 0b00'0000'0010UL},
       {-std::numeric_limits<float>::denorm_min(), 0b00'0000'0100UL},
       {-0.0f, 0b00'0000'1000UL},
       {0.0f, 0b00'0001'0000UL},
       {std::numeric_limits<float>::denorm_min(), 0b00'0010'0000UL},
       {1.0f, 0b00'0100'0000UL},
       {std::numeric_limits<float>::infinity(), 0b00'1000'0000UL},
       {std::numeric_limits<float>::signaling_NaN(), 0b01'0000'0000UL},
       {std::numeric_limits<float>::quiet_NaN(), 0b10'0000'0000UL}});
  // Fclass.D
  InterpretOpFpGpRegisterTargetSingleInput(
      0xe20110d3,
      {std::tuple{-std::numeric_limits<double>::infinity(), 0b00'0000'0001UL},
       {-1.0, 0b00'0000'0010UL},
       {-std::numeric_limits<double>::denorm_min(), 0b00'0000'0100UL},
       {-0.0, 0b00'0000'1000UL},
       {0.0, 0b00'0001'0000UL},
       {std::numeric_limits<double>::denorm_min(), 0b00'0010'0000UL},
       {1.0, 0b00'0100'0000UL},
       {std::numeric_limits<double>::infinity(), 0b00'1000'0000UL},
       {std::numeric_limits<double>::signaling_NaN(), 0b01'0000'0000UL},
       {std::numeric_limits<double>::quiet_NaN(), 0b10'0000'0000UL}});
}

TEST_F(Riscv64InterpreterTest, RoundingModeTest) {
  // FAdd.S
  InterpretOpFp(0x003100d3,
                // Test RNE
                {std::tuple{1.0000001f, 0.000000059604645f, 1.0000002f},
                 {1.0000002f, 0.000000059604645f, 1.0000002f},
                 {1.0000004f, 0.000000059604645f, 1.0000005f},
                 {-1.0000001f, -0.000000059604645f, -1.0000002f},
                 {-1.0000002f, -0.000000059604645f, -1.0000002f},
                 {-1.0000004f, -0.000000059604645f, -1.0000005f}});
  // FAdd.S
  InterpretOpFp(0x003110d3,
                // Test RTZ
                {std::tuple{1.0000001f, 0.000000059604645f, 1.0000001f},
                 {1.0000002f, 0.000000059604645f, 1.0000002f},
                 {1.0000004f, 0.000000059604645f, 1.0000004f},
                 {-1.0000001f, -0.000000059604645f, -1.0000001f},
                 {-1.0000002f, -0.000000059604645f, -1.0000002f},
                 {-1.0000004f, -0.000000059604645f, -1.0000004f}});
  // FAdd.S
  InterpretOpFp(0x003120d3,
                // Test RDN
                {std::tuple{1.0000001f, 0.000000059604645f, 1.0000001f},
                 {1.0000002f, 0.000000059604645f, 1.0000002f},
                 {1.0000004f, 0.000000059604645f, 1.0000004f},
                 {-1.0000001f, -0.000000059604645f, -1.0000002f},
                 {-1.0000002f, -0.000000059604645f, -1.0000004f},
                 {-1.0000004f, -0.000000059604645f, -1.0000005f}});
  // FAdd.S
  InterpretOpFp(0x003130d3,
                // Test RUP
                {std::tuple{1.0000001f, 0.000000059604645f, 1.0000002f},
                 {1.0000002f, 0.000000059604645f, 1.0000004f},
                 {1.0000004f, 0.000000059604645f, 1.0000005f},
                 {-1.0000001f, -0.000000059604645f, -1.0000001f},
                 {-1.0000002f, -0.000000059604645f, -1.0000002f},
                 {-1.0000004f, -0.000000059604645f, -1.0000004f}});
  // FAdd.S
  InterpretOpFp(0x003140d3,
                // Test RMM
                {std::tuple{1.0000001f, 0.000000059604645f, 1.0000002f},
                 {1.0000002f, 0.000000059604645f, 1.0000004f},
                 {1.0000004f, 0.000000059604645f, 1.0000005f},
                 {-1.0000001f, -0.000000059604645f, -1.0000002f},
                 {-1.0000002f, -0.000000059604645f, -1.0000004f},
                 {-1.0000004f, -0.000000059604645f, -1.0000005f}});

  // FAdd.D
  InterpretOpFp(
      0x023100d3,
      // Test RNE
      {std::tuple{1.0000000000000002, 0.00000000000000011102230246251565, 1.0000000000000004},
       {1.0000000000000004, 0.00000000000000011102230246251565, 1.0000000000000004},
       {1.0000000000000007, 0.00000000000000011102230246251565, 1.0000000000000009},
       {-1.0000000000000002, -0.00000000000000011102230246251565, -1.0000000000000004},
       {-1.0000000000000004, -0.00000000000000011102230246251565, -1.0000000000000004},
       {-1.0000000000000007, -0.00000000000000011102230246251565, -1.0000000000000009}});
  // FAdd.D
  InterpretOpFp(
      0x023110d3,
      // Test RTZ
      {std::tuple{1.0000000000000002, 0.00000000000000011102230246251565, 1.0000000000000002},
       {1.0000000000000004, 0.00000000000000011102230246251565, 1.0000000000000004},
       {1.0000000000000007, 0.00000000000000011102230246251565, 1.0000000000000007},
       {-1.0000000000000002, -0.00000000000000011102230246251565, -1.0000000000000002},
       {-1.0000000000000004, -0.00000000000000011102230246251565, -1.0000000000000004},
       {-1.0000000000000007, -0.00000000000000011102230246251565, -1.0000000000000007}});
  // FAdd.D
  InterpretOpFp(
      0x023120d3,
      // Test RDN
      {std::tuple{1.0000000000000002, 0.00000000000000011102230246251565, 1.0000000000000002},
       {1.0000000000000004, 0.00000000000000011102230246251565, 1.0000000000000004},
       {1.0000000000000007, 0.00000000000000011102230246251565, 1.0000000000000007},
       {-1.0000000000000002, -0.00000000000000011102230246251565, -1.0000000000000004},
       {-1.0000000000000004, -0.00000000000000011102230246251565, -1.0000000000000007},
       {-1.0000000000000007, -0.00000000000000011102230246251565, -1.0000000000000009}});
  // FAdd.D
  InterpretOpFp(
      0x023130d3,
      // Test RUP
      {std::tuple{1.0000000000000002, 0.00000000000000011102230246251565, 1.0000000000000004},
       {1.0000000000000004, 0.00000000000000011102230246251565, 1.0000000000000007},
       {1.0000000000000007, 0.00000000000000011102230246251565, 1.0000000000000009},
       {-1.0000000000000002, -0.00000000000000011102230246251565, -1.0000000000000002},
       {-1.0000000000000004, -0.00000000000000011102230246251565, -1.0000000000000004},
       {-1.0000000000000007, -0.00000000000000011102230246251565, -1.0000000000000007}});
  // FAdd.D
  InterpretOpFp(
      0x023140d3,
      // Test RMM
      {std::tuple{1.0000000000000002, 0.00000000000000011102230246251565, 1.0000000000000004},
       {1.0000000000000004, 0.00000000000000011102230246251565, 1.0000000000000007},
       {1.0000000000000007, 0.00000000000000011102230246251565, 1.0000000000000009},
       {-1.0000000000000002, -0.00000000000000011102230246251565, -1.0000000000000004},
       {-1.0000000000000004, -0.00000000000000011102230246251565, -1.0000000000000007},
       {-1.0000000000000007, -0.00000000000000011102230246251565, -1.0000000000000009}});
}

TEST_F(Riscv64InterpreterTest, LoadFpInstructions) {
  // Offset is always 8.
  // Flw
  InterpretLoadFp(0x00812087, kDataToLoad | 0xffffffff00000000ULL);
  // Fld
  InterpretLoadFp(0x00813087, kDataToLoad);
}

TEST_F(Riscv64InterpreterTest, AtomicLoadInstructions) {
  // Lrw
  InterpretAtomicLoad(0x1000a12f, int64_t{int32_t(kDataToLoad)});
  // Lrd
  InterpretAtomicLoad(0x1000b12f, kDataToLoad);
}

TEST_F(Riscv64InterpreterTest, StoreFpInstructions) {
  // Offset is always 8.
  // Fsw
  InterpretStoreFp(0x0020a427, kDataToStore & 0xffff'ffffULL);
  // Fsd
  InterpretStoreFp(0x0020b427, kDataToStore);
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
