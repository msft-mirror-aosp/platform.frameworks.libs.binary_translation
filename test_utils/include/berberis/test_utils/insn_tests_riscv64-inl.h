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
// #include <vector>
//
// #include "berberis/base/bit_util.h"
// #include "berberis/guest_state/guest_addr.h"
// #include "berberis/guest_state/guest_state_riscv64.h"
//
// 2. RunOneInstruction is defined and implemented
//
// 3. TESTSUITE macro is defined

#ifndef TESTSUITE
#error "TESTSUITE is undefined"
#endif

// TODO(b/276787675): remove these files from interpreter when they are no longer needed there.
// Maybe extract FPvalueToFPReg and TupleMap to a separate header?
inline constexpr class FPValueToFPReg {
 public:
  uint64_t operator()(uint64_t value) const { return value; }
  uint64_t operator()(float value) const {
    return bit_cast<uint32_t>(value) | 0xffff'ffff'0000'0000;
  }
  uint64_t operator()(double value) const { return bit_cast<uint64_t>(value); }
} kFPValueToFPReg;

// Helper function for the unit tests. Can be used to normalize values before processing.
//
// “container” is supposed to be container of tuples, e.g. std::initializer_list<std::tuple<…>>.
// “transformer” would be applied to the individual elements of tuples in the following loop:
//
//   for (auto& [value1, value2, value3] : TupleMap(container, [](auto value){ return …; })) {
//     …
//   }
//
// Returns vector of tuples where each tuple element is processed by transformer.
template <typename ContainerType, typename Transformer>
decltype(auto) TupleMap(const ContainerType& container, const Transformer& transformer) {
  using std::begin;

  auto transform_tuple_func = [&transformer](auto&&... value) {
    return std::tuple{transformer(value)...};
  };

  std::vector<decltype(std::apply(transform_tuple_func, *begin(container)))> result;

  for (const auto& tuple : container) {
    result.push_back(std::apply(transform_tuple_func, tuple));
  }

  return result;
}

class TESTSUITE : public ::testing::Test {
 public:
  // Compressed Instructions.

  template <RegisterType register_type, uint64_t expected_result, uint8_t kTargetReg>
  void TestCompressedStore(uint16_t insn_bytes, uint64_t offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    store_area_ = 0;
    SetXReg<kTargetReg>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - offset));
    SetReg<register_type, 9>(state_.cpu, kDataToLoad);
    EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + 2));
    EXPECT_EQ(store_area_, expected_result);
  }

  template <RegisterType register_type, uint64_t expected_result, uint8_t kSourceReg>
  void TestCompressedLoad(uint16_t insn_bytes, uint64_t offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<kSourceReg>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&kDataToLoad) - offset));
    EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + 2));
    EXPECT_EQ((GetReg<register_type, 9>(state_.cpu)), expected_result);
  }

  void TestCAddi(uint16_t insn_bytes, uint64_t expected_increment) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<2>(state_.cpu, 1);
    EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + 2));
    EXPECT_EQ(GetXReg<2>(state_.cpu), 1 + expected_increment);
  }

  void TestCAddi16sp(uint16_t insn_bytes, uint64_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<2>(state_.cpu, 1);
    EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + 2));
    EXPECT_EQ(GetXReg<2>(state_.cpu), 1 + expected_offset);
  }

  void TestLi(uint32_t insn_bytes, uint64_t expected_result) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + 2));
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  void TestCAddi4spn(uint16_t insn_bytes, uint64_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<2>(state_.cpu, 1);
    EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + 2));
    EXPECT_EQ(GetXReg<9>(state_.cpu), 1 + expected_offset);
  }

  void TestCBeqzBnez(uint16_t insn_bytes, uint64_t value, int16_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<9>(state_.cpu, value);
    EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + expected_offset));
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
  }

  void TestCMiscAlu(uint16_t insn_bytes,
                    std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto [arg1, arg2, expected_result] : args) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetXReg<8>(state_.cpu, arg1);
      SetXReg<9>(state_.cpu, arg2);
      EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + 2));
      EXPECT_EQ(GetXReg<8>(state_.cpu), expected_result);
    }
  }

  void TestCMiscAluImm(uint16_t insn_bytes, uint64_t value, uint64_t expected_result) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<9>(state_.cpu, value);
    EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + 2));
    EXPECT_EQ(GetXReg<9>(state_.cpu), expected_result);
  }

  void TestCJ(uint16_t insn_bytes, int16_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + expected_offset));
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
  }

  void TestCOp(uint32_t insn_bytes,
               std::initializer_list<std::tuple<uint64_t, uint64_t, uint64_t>> args) {
    for (auto [arg1, arg2, expected_result] : args) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetXReg<1>(state_.cpu, arg1);
      SetXReg<2>(state_.cpu, arg2);
      EXPECT_TRUE(RunOneInstruction<2>(&state_, state_.cpu.insn_addr + 2));
      EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
    }
  }

  // Non-Compressed Instructions.

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

  template <typename... Types>
  void TestOpFp(uint32_t insn_bytes, std::initializer_list<std::tuple<Types...>> args) {
    for (auto [arg1, arg2, expected_result] : TupleMap(args, kFPValueToFPReg)) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      SetFReg<2>(state_.cpu, arg1);
      SetFReg<3>(state_.cpu, arg2);
      EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
      EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
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

  void TestBranch(uint32_t insn_bytes,
                  std::initializer_list<std::tuple<uint64_t, uint64_t, int8_t>> args) {
    auto code_start = ToGuestAddr(&insn_bytes);
    for (auto [arg1, arg2, expected_offset] : args) {
      state_.cpu.insn_addr = code_start;
      SetXReg<1>(state_.cpu, arg1);
      SetXReg<2>(state_.cpu, arg2);
      EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + expected_offset));
      EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    }
  }

  void TestJumpAndLink(uint32_t insn_bytes, int8_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + expected_offset));
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    EXPECT_EQ(GetXReg<1>(state_.cpu), code_start + 4);
  }

  void TestLoad(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    // Offset is always 8.
    SetXReg<2>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&kDataToLoad) - 8));
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  // kLinkRegisterOffsetIfUsed is size of instruction or 0 if instruction does not link register.
  template <uint8_t kLinkRegisterOffsetIfUsed>
  void TestJumpAndLinkRegister(uint32_t insn_bytes, uint64_t base_disp, int64_t expected_offset) {
    auto code_start = ToGuestAddr(&insn_bytes);
    state_.cpu.insn_addr = code_start;
    SetXReg<1>(state_.cpu, 0);
    SetXReg<2>(state_.cpu, code_start + base_disp);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + expected_offset));
    EXPECT_EQ(state_.cpu.insn_addr, code_start + expected_offset);
    if constexpr (kLinkRegisterOffsetIfUsed == 0) {
      EXPECT_EQ(GetXReg<1>(state_.cpu), 0UL);
    } else {
      EXPECT_EQ(GetXReg<1>(state_.cpu), code_start + kLinkRegisterOffsetIfUsed);
    }
  }

  void TestStore(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    // Offset is always 8.
    SetXReg<1>(state_.cpu, ToGuestAddr(bit_cast<uint8_t*>(&store_area_) - 8));
    SetXReg<2>(state_.cpu, kDataToStore);
    store_area_ = 0;
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(store_area_, expected_result);
  }

 protected:
  static constexpr uint64_t kDataToLoad{0xffffeeeeddddccccULL};
  static constexpr uint64_t kDataToStore = kDataToLoad;
  uint64_t store_area_;

 private:
  ThreadState state_;
};

// Tests for Compressed Instructions.
template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedLoadOrStore32bit(TESTSUITE* that) {
  union {
    uint16_t offset;
    struct [[gnu::packed]] {
      uint8_t : 2;
      uint8_t i2 : 1;
      uint8_t i3_i5 : 3;
      uint8_t i6 : 1;
    } i_bits;
  };
  for (offset = uint8_t{0}; offset < uint8_t{128}; offset += 4) {
    union {
      int16_t parcel;
      struct [[gnu::packed]] {
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

TEST_F(TESTSUITE, CompressedLoadAndStores32bit) {
  // c.Lw
  TestCompressedLoadOrStore32bit<
      0b010'000'000'00'000'00,
      &TESTSUITE::TestCompressedLoad<RegisterType::kReg,
                                     static_cast<uint64_t>(static_cast<int32_t>(kDataToLoad)),
                                     8>>(this);
  // c.Sw
  TestCompressedLoadOrStore32bit<
      0b110'000'000'00'000'00,
      &TESTSUITE::TestCompressedStore<RegisterType::kReg,
                                      static_cast<uint64_t>(static_cast<uint32_t>(kDataToLoad)),
                                      8>>(this);
}

template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedLoadOrStore64bit(TESTSUITE* that) {
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

TEST_F(TESTSUITE, CompressedLoadAndStores) {
  // c.Ld
  TestCompressedLoadOrStore64bit<
      0b011'000'000'00'000'00,
      &TESTSUITE::TestCompressedLoad<RegisterType::kReg, kDataToLoad, 8>>(this);
  // c.Sd
  TestCompressedLoadOrStore64bit<
      0b111'000'000'00'000'00,
      &TESTSUITE::TestCompressedStore<RegisterType::kReg, kDataToLoad, 8>>(this);
}

TEST_F(TESTSUITE, TestCompressedStore32bitsp) {
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
    TestCompressedStore<RegisterType::kReg,
                        static_cast<uint64_t>(static_cast<uint32_t>(kDataToStore)),
                        2>(o_bits.parcel, offset);
  }
}

template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedStore64bitsp(TESTSUITE* that) {
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

TEST_F(TESTSUITE, TestCompressedStore64bitsp) {
  // c.Sdsp
  TestCompressedStore64bitsp<0b011'000'000'00'000'00,
                             &TESTSUITE::TestCompressedStore<RegisterType::kReg, kDataToStore, 2>>(
      this);
}

TEST_F(TESTSUITE, TestCompressedLoad32bitsp) {
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
    TestCompressedLoad<RegisterType::kReg,
                       static_cast<uint64_t>(static_cast<int32_t>(kDataToLoad)),
                       2>(o_bits.parcel, offset);
  }
}

template <uint16_t opcode, auto execute_instruction_func>
void TestCompressedLoad64bitsp(TESTSUITE* that) {
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

TEST_F(TESTSUITE, TestCompressedLoad64bitsp) {
  // c.Ldsp
  TestCompressedLoad64bitsp<0b011'000'000'00'000'00,
                            &TESTSUITE::TestCompressedLoad<RegisterType::kReg, kDataToLoad, 2>>(
      this);
}

TEST_F(TESTSUITE, CAddi) {
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
    TestCAddi(o_bits.parcel | 0b0000'0000'0000'0001, offset);
    // c.Addiw
    TestCAddi(o_bits.parcel | 0b0010'0000'0000'0001, offset);
  }
}

TEST_F(TESTSUITE, CAddi16sp) {
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
    TestCAddi16sp(o_bits.parcel, offset);
  }
}

TEST_F(TESTSUITE, CLui) {
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
    TestLi(o_bits.parcel, offset);
  }
}

TEST_F(TESTSUITE, CLi) {
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
    TestLi(o_bits.parcel, offset);
  }
}

TEST_F(TESTSUITE, CAddi4spn) {
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
    TestCAddi4spn(o_bits.parcel, offset);
  }
}

TEST_F(TESTSUITE, CBeqzBnez) {
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
    TestCBeqzBnez(o_bits.parcel | 0b1100'0000'0000'0001, 0, offset);
    TestCBeqzBnez(o_bits.parcel | 0b1110'0000'0000'0001, 1, offset);
  }
}

TEST_F(TESTSUITE, CMiscAluInstructions) {
  // c.Sub
  TestCMiscAlu(0x8c05, {{42, 23, 19}});
  // c.Xor
  TestCMiscAlu(0x8c25, {{0b0101, 0b0011, 0b0110}});
  // c.Or
  TestCMiscAlu(0x8c45, {{0b0101, 0b0011, 0b0111}});
  // c.And
  TestCMiscAlu(0x8c65, {{0b0101, 0b0011, 0b0001}});
  // c.SubW
  TestCMiscAlu(0x9c05, {{42, 23, 19}});
  // c.AddW
  TestCMiscAlu(0x9c25, {{19, 23, 42}});
}

TEST_F(TESTSUITE, CMiscAluImm) {
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
    TestCMiscAluImm(o_bits.parcel | 0b1000'0000'0000'0001,
                    0x8000'0000'0000'0000ULL,
                    0x8000'0000'0000'0000ULL >> uimm);
    // c.Srai
    TestCMiscAluImm(o_bits.parcel | 0b1000'0100'0000'0001,
                    0x8000'0000'0000'0000LL,
                    ~0 ^ ((0x8000'0000'0000'0000 ^ ~0) >>
                          uimm));  // Avoid shifting negative numbers to avoid UB
    // c.Andi
    TestCMiscAluImm(o_bits.parcel | 0b1000'1000'0000'0001,
                    0xffff'ffff'ffff'ffffULL,
                    0xffff'ffff'ffff'ffffULL & imm);

    // Previous instructions use 3-bit register encoding where 0b000 is r8, 0b001 is r9, etc.
    // c.Slli uses 5-bit register encoding. Since we want it to also work with r9 in the test body
    // we add 0b01000 to register bits to mimic that shift-by-8.
    // c.Slli                                   vvvvvv adds 8 to r to handle rd' vs rd difference.
    TestCMiscAluImm(o_bits.parcel | 0b0000'0100'0000'0010,
                    0x0000'0000'0000'0001ULL,
                    0x0000'0000'0000'0001ULL << uimm);
  }
}

TEST_F(TESTSUITE, CJ) {
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
    TestCJ(o_bits.parcel, offset);
  }
}

TEST_F(TESTSUITE, CJalr) {
  // C.Jr
  TestJumpAndLinkRegister<0>(0x8102, 42, 42);
  // C.Mv
  TestCOp(0x808a, {{0, 1, 1}});
  // C.Jalr
  TestJumpAndLinkRegister<2>(0x9102, 42, 42);
  // C.Add
  TestCOp(0x908a, {{1, 2, 3}});
}

// Tests for Non-Compressed Instructions.

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
  // Andn
  TestOp(0x403170b3, {{0b0101, 0b0011, 0b0100}});
  // Orn
  TestOp(0x403160b3, {{0b0101, 0b0011, 0xffff'ffff'ffff'fffd}});
  // Xnor
  TestOp(0x403140b3, {{0b0101, 0b0011, 0xffff'ffff'ffff'fff9}});
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
  // Rori
  TestOpImm(0x60015093, {{0xf000'0000'0000'000fULL, 4, 0xff00'0000'0000'0000ULL}});
  // Clz
  TestOpImm(0x60011093, {{0, 0, 64}});
  TestOpImm(0x60011093, {{123, 0, 57}});
  // Ctz
  TestOpImm(0x60111093, {{0, 0, 64}});
  TestOpImm(0x60111093, {{0x01000000'0000, 0, 40}});
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
  // Roriw
  TestOpImm(0x6001509b, {{0x0000'0000'f000'000fULL, 4, 0x0000'0000'0000'0000'ff00'0000}});
  // Clzw
  TestOpImm(0x6001109b, {{0, 0, 32}});
  TestOpImm(0x6001109b, {{123, 0, 25}});
  // Ctzw
  TestOpImm(0x6011109b, {{0, 0, 32}});
  TestOpImm(0x6011109b, {{0x0000'0000'0000'0010, 0, 4}});
}

TEST_F(TESTSUITE, OpFpInstructions) {
  // FAdd.S
  TestOpFp(0x003100d3, {std::tuple{1.0f, 2.0f, 3.0f}});
  // FAdd.D
  TestOpFp(0x023100d3, {std::tuple{1.0, 2.0, 3.0}});
  // FSub.S
  TestOpFp(0x083100d3, {std::tuple{3.0f, 2.0f, 1.0f}});
  // FSub.D
  TestOpFp(0x0a3100d3, {std::tuple{3.0, 2.0, 1.0}});
  // FMul.S
  TestOpFp(0x103100d3, {std::tuple{3.0f, 2.0f, 6.0f}});
  // FMul.D
  TestOpFp(0x123100d3, {std::tuple{3.0, 2.0, 6.0}});
  // FDiv.S
  TestOpFp(0x183100d3, {std::tuple{6.0f, 2.0f, 3.0f}});
  // FDiv.D
  TestOpFp(0x1a3100d3, {std::tuple{6.0, 2.0, 3.0}});

  // FSgnj.S
  TestOpFp(0x203100d3,
           {std::tuple{1.0f, 2.0f, 1.0f},
            {-1.0f, 2.0f, 1.0f},
            {1.0f, -2.0f, -1.0f},
            {-1.0f, -2.0f, -1.0f}});
  // FSgnj.D
  TestOpFp(0x223100d3,
           {
               std::tuple{1.0, 2.0, 1.0},
               {-1.0, 2.0, 1.0},
               {1.0, -2.0, -1.0},
               {-1.0, -2.0, -1.0},
           });
  // FSgnjn.S
  TestOpFp(0x203110d3,
           {
               std::tuple{1.0f, 2.0f, -1.0f},
               {1.0f, 2.0f, -1.0f},
               {1.0f, -2.0f, 1.0f},
               {-1.0f, -2.0f, 1.0f},
           });
  // FSgnjn.D
  TestOpFp(0x223110d3,
           {
               std::tuple{1.0, 2.0, -1.0},
               {1.0, 2.0, -1.0},
               {1.0, -2.0, 1.0},
               {-1.0, -2.0, 1.0},
           });
  // FSgnjx.S
  TestOpFp(0x203120d3,
           {
               std::tuple{1.0f, 2.0f, 1.0f},
               {-1.0f, 2.0f, -1.0f},
               {1.0f, -2.0f, -1.0f},
               {-1.0f, -2.0f, 1.0f},
           });
  // FSgnjx.D
  TestOpFp(0x223120d3,
           {
               std::tuple{1.0, 2.0, 1.0},
               {-1.0, 2.0, -1.0},
               {1.0, -2.0, -1.0},
               {-1.0, -2.0, 1.0},
           });
  // FMin.S
  TestOpFp(0x283100d3,
           {std::tuple{+0.f, +0.f, +0.f},
            {+0.f, -0.f, -0.f},
            {-0.f, +0.f, -0.f},
            {-0.f, -0.f, -0.f},
            {+0.f, 1.f, +0.f},
            {-0.f, 1.f, -0.f}});
  // FMin.D
  TestOpFp(0x2a3100d3,
           {std::tuple{+0.0, +0.0, +0.0},
            {+0.0, -0.0, -0.0},
            {-0.0, +0.0, -0.0},
            {-0.0, -0.0, -0.0},
            {+0.0, 1.0, +0.0},
            {-0.0, 1.0, -0.0}});
  // FMax.S
  TestOpFp(0x283110d3,
           {std::tuple{+0.f, +0.f, +0.f},
            {+0.f, -0.f, +0.f},
            {-0.f, +0.f, +0.f},
            {-0.f, -0.f, -0.f},
            {+0.f, 1.f, 1.f},
            {-0.f, 1.f, 1.f}});
  // FMax.D
  TestOpFp(0x2a3110d3,
           {std::tuple{+0.0, +0.0, +0.0},
            {+0.0, -0.0, +0.0},
            {-0.0, +0.0, +0.0},
            {-0.0, -0.0, -0.0},
            {+0.0, 1.0, 1.0},
            {-0.0, 1.0, 1.0}});
}

TEST_F(TESTSUITE, UpperImmInstructions) {
  // Auipc
  TestAuipc(0xfedcb097, 0xffff'ffff'fedc'b000);
  // Lui
  TestLui(0xfedcb0b7, 0xffff'ffff'fedc'b000);
}

TEST_F(TESTSUITE, TestBranchInstructions) {
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

TEST_F(TESTSUITE, JumpAndLinkInstructions) {
  // Jal
  TestJumpAndLink(0x008000ef, 8);
  // Jal with negative offset.
  TestJumpAndLink(0xffdff0ef, -4);
}

TEST_F(TESTSUITE, JumpAndLinkRegisterInstructions) {
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

TEST_F(TESTSUITE, LoadInstructions) {
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

TEST_F(TESTSUITE, StoreInstructions) {
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
