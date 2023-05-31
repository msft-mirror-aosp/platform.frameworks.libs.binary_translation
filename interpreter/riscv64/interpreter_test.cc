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
#include "berberis/intrinsics/guest_rounding_modes.h"  // ScopedRoundingMode

#include "fp_regs_util.h"
#include "tuple_map.h"

namespace berberis {

namespace {

class Riscv64InterpreterTest : public ::testing::Test {
 public:
  // Compressed Instructions.

  template <RegisterType register_type, uint64_t expected_result>
  void InterpretCompressedStore(uint16_t insn_bytes, uint64_t offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    store_area_ = 0;
    SetXReg<8>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - offset));
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

  void InterpretAuipc(uint32_t insn_bytes, uint64_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    InterpretInsn(&state_);
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_offset + code_start);
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

  void InterpretLoadFp(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    // Offset is always 8.
    SetXReg<2>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&kDataToLoad) - 8));
    InterpretInsn(&state_);
    EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
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

  void InterpretStoreFp(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    // Offset is always 8.
    SetXReg<1>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - 8));
    SetFReg<2>(state_.cpu, kDataToStore);
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

// Tests for Compressed Instructions.

template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedLoadOrStore32bit(Riscv64InterpreterTest* that) {
  union {
    uint16_t offset;
    struct {
      uint8_t : 2;
      uint8_t i2 : 1;
      uint8_t i3_i5 : 3;
      uint8_t i6 : 1;
    } i_bits;
  };
  for (offset = uint8_t{0}; offset < uint8_t{128}; offset += 4) {
    union {
      int16_t parcel;
      struct {
        uint8_t low_opcode : 2;
        uint8_t rd : 3;
        uint8_t i6 : 1;
        uint8_t i2 : 1;
        uint8_t rs : 3;
        uint8_t i3_i5 : 3;
        uint8_t high_opcode : 3;
      } __attribute__((__packed__));
    } o_bits = {
        .low_opcode = 0b00,
        .rd = 1,
        .i6 = i_bits.i6,
        .i2 = i_bits.i2,
        .rs = 0,
        .i3_i5 = i_bits.i3_i5,
        .high_opcode = 0b000,
    };
    (that->*execute_instruction_func)(o_bits.parcel | opcode, offset);
  }
}

TEST_F(Riscv64InterpreterTest, CompressedLoadAndStores32bit) {
  // c.Lw
  TestCompressedLoadOrStore32bit<0b010'000'000'00'000'00,
                                 &Riscv64InterpreterTest::InterpretCompressedLoad<
                                     RegisterType::kReg,
                                     static_cast<uint64_t>(static_cast<int32_t>(kDataToLoad)),
                                     8>>(this);
  // c.Sw
  TestCompressedLoadOrStore32bit<0b110'000'000'00'000'00,
                                 &Riscv64InterpreterTest::InterpretCompressedStore<
                                     RegisterType::kReg,
                                     static_cast<uint64_t>(static_cast<uint32_t>(kDataToLoad))>>(
      this);
}

template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedLoadOrStore64bit(Riscv64InterpreterTest* that) {
  union {
    uint16_t offset;
    struct {
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
      &Riscv64InterpreterTest::InterpretCompressedStore<RegisterType::kFpReg, kDataToLoad>>(this);
  // c.Sd
  TestCompressedLoadOrStore64bit<
      0b111'000'000'00'000'00,
      &Riscv64InterpreterTest::InterpretCompressedStore<RegisterType::kReg, kDataToLoad>>(this);
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
    struct {
      uint8_t i4_i0 : 5;
      uint8_t i5 : 1;
    } i_bits;
  };
  for (offset = int8_t{-32}; offset < int8_t{31}; offset++) {
    union {
      int16_t parcel;
      struct {
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
    struct {
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
    struct {
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
      struct {
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
    struct {
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
    struct {
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
    struct {
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
      struct {
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

TEST_F(Riscv64InterpreterTest, CsrInstrctuion) {
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
  InterpretOp(0x23150bb,
              {{0x9999'9999'9999'9999, 0x3333, 0x0000'0000'0003'0003},
               {0xffff'ffff'8000'0000, 1, 0xffff'ffff'8000'0000}});
  // Remw
  InterpretOp(0x23160bb, {{0x9999'9999'9999'9999, 0x3333, 0xffff'ffff'ffff'ffff}});
  // Remuw
  InterpretOp(0x23170bb,
              {{0x9999'9999'9999'9999, 0x3333, 0},
               {0xffff'ffff'8000'0000, 0xffff'ffff'8000'0001, 0xffff'ffff'8000'0000}});
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

TEST_F(Riscv64InterpreterTest, UpperImmArgs) {
  // Lui
  InterpretLi(0xfedcb0b7, 0xffff'ffff'fedc'b000);
  // Auipc
  InterpretAuipc(0xfedcb097, 0xffff'ffff'fedc'b000);
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

TEST_F(Riscv64InterpreterTest, LoadFpInstructions) {
  // Offset is always 8.
  // Flw
  InterpretLoadFp(0x00812087, kDataToLoad | 0xffffffff00000000ULL);
  // Fld
  InterpretLoadFp(0x00813087, kDataToLoad);
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

TEST_F(Riscv64InterpreterTest, StoreFpInstructions) {
  // Offset is always 8.
  // Fsw
  InterpretStoreFp(0x0020a427, kDataToStore & 0xffff'ffffULL);
  // Fsd
  InterpretStoreFp(0x0020b427, kDataToStore);
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
  InterpretJumpAndLinkRegister<4>(0x004100e7, 38, 42);
  // Jalr offset=-4.
  InterpretJumpAndLinkRegister<4>(0xffc100e7, 42, 38);
  // Jalr offset=5 - must properly align the target to even.
  InterpretJumpAndLinkRegister<4>(0x005100e7, 38, 42);
  // Jr offset=4.
  InterpretJumpAndLinkRegister<0>(0x00410067, 38, 42);
  // Jr offset=-4.
  InterpretJumpAndLinkRegister<0>(0xffc10067, 42, 38);
  // Jr offset=5 - must properly align the target to even.
  InterpretJumpAndLinkRegister<0>(0x00510067, 38, 42);
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
