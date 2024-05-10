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
#include <cstdlib>
#include <tuple>

namespace {

template <typename T>
constexpr T BitUtilLog2(T x) {
  return __builtin_ctz(x);
}

// A wrapper around __uint128 which can be constructed from a pair of uint64_t literals.
class SIMD128 {
 public:
  SIMD128(){};
  constexpr SIMD128(std::tuple<uint64_t, uint64_t> u64_u64) : u64_u64_{u64_u64} {};
  constexpr SIMD128(__uint128_t u128) : u128_{u128} {};

  SIMD128& operator=(const SIMD128& other) {
    u128_ = other.u128_;
    return *this;
  };
  SIMD128& operator|=(const SIMD128& other) {
    u128_ |= other.u128_;
    return *this;
  }
  bool operator==(const SIMD128& other) const { return u128_ == other.u128_; }
  SIMD128 operator>>(size_t shift_amount) const { return u128_ >> shift_amount; }
  SIMD128 operator<<(size_t shift_amount) const { return u128_ << shift_amount; }

 private:
  union {
    std::tuple<uint64_t, uint64_t> u64_u64_;
    __uint128_t u128_;
  };
};

constexpr SIMD128 kVectorCalculationsSource[16] = {
    {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908}},
    {{0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918}},
    {{0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928}},
    {{0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938}},
    {{0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948}},
    {{0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958}},
    {{0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968}},
    {{0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},

    {{0x9e0c'9a09'9604'9200, 0x8e1c'8a18'8614'8211}},
    {{0xbe2c'ba29'b624'b220, 0xae3c'aa38'a634'a231}},
    {{0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251}},
    {{0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271}},
    {{0x1e8c'1a89'1684'1280, 0x0e9c'0a98'0694'0291}},
    {{0x3eac'3aa9'36a4'32a0, 0x2ebc'2ab8'26b4'22b1}},
    {{0x5ecc'5ac9'56c4'52c0, 0x4edc'4ad8'46d4'42d1}},
    {{0x7eec'7ae9'76e4'72e0, 0x6efc'6af8'66f4'62f1}},
};

// Easily recognizable bit pattern for target register.
constexpr SIMD128 kUndisturbedResult{{0x5555'5555'5555'5555, 0x5555'5555'5555'5555}};

SIMD128 GetAgnosticResult() {
  static const bool kRvvAgnosticIsUndisturbed = getenv("RVV_AGNOSTIC_IS_UNDISTURBED") != nullptr;
  if (kRvvAgnosticIsUndisturbed) {
    return kUndisturbedResult;
  }
  return {{~uint64_t{0U}, ~uint64_t{0U}}};
}

const SIMD128 kAgnosticResult = GetAgnosticResult();

// Mask in form suitable for storing in v0 and use in v0.t form.
static constexpr SIMD128 kMask{{0xd5ad'd6b5'ad6b'b5ad, 0x6af7'57bb'deed'7bb5}};

using ExecInsnFunc = void (*)();

void RunTwoVectorArgsOneRes(ExecInsnFunc exec_insn,
                            const SIMD128* src,
                            SIMD128* res,
                            uint64_t vtype,
                            uint64_t vlmax) {
  uint64_t vstart, vl;
  // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
  // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
  asm(  // Load arguments and undisturbed result.
      "vsetvli t0, zero, e64, m8, ta, ma\n\t"
      "vle64.v v8, (%[res])\n\t"
      "vle64.v v16, (%[src])\n\t"
      "addi t0, %[src], 128\n\t"
      "vle64.v v24, (t0)\n\t"
      // Load mask.
      "vsetvli t0, zero, e64, m1, ta, ma\n\t"
      "vle64.v v0, (%[mask])\n\t"
      // Execute tested instruction.
      "vsetvl t0, zero, %[vtype]\n\t"
      "jalr %[exec_insn]\n\t"
      // Save vstart and vl just after insn execution for checks.
      "csrr %[vstart], vstart\n\t"
      "csrr %[vl], vl\n\t"
      // Store the result.
      "vsetvli t0, zero, e64, m8, ta, ma\n\t"
      "vse64.v v8, (%[res])\n\t"
      : [vstart] "=&r"(vstart), [vl] "=&r"(vl)
      : [exec_insn] "r"(exec_insn),
        [src] "r"(src),
        [res] "r"(res),
        [vtype] "r"(vtype),
        [mask] "r"(&kMask)
      : "t0",
        "ra",
        "v0",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v30",
        "v31",
        "memory");
  // Every vector instruction must set vstart to 0, but shouldn't touch vl.
  EXPECT_EQ(vstart, 0);
  EXPECT_EQ(vl, vlmax);
}

template <typename... ExpectedResultType>
void TestVectorReductionInstruction(
    ExecInsnFunc exec_insn,
    ExecInsnFunc exec_masked_insn,
    const SIMD128 (&source)[16],
    std::tuple<const ExpectedResultType (&)[8],
               const ExpectedResultType (&)[8]>... expected_result) {
  // Each expected_result input to this function is the vd[0] value of the reduction, for each
  // of the possible vlmul, i.e. expected_result_vd0_int8[n] = vd[0], int8, no mask, vlmul=n.
  //
  // As vlmul=4 is reserved, expected_result_vd0_*[4] is ignored.
  auto Verify = [&source](ExecInsnFunc exec_insn,
                          uint8_t vsew,
                          uint8_t vlmul,
                          const auto& expected_result) {
    for (uint8_t vta = 0; vta < 2; ++vta) {
      for (uint8_t vma = 0; vma < 2; ++vma) {
        uint64_t vtype = (vma << 7) | (vta << 6) | (vsew << 3) | vlmul;
        uint64_t vlmax = 0;
        asm("vsetvl %0, zero, %1" : "=r"(vlmax) : "r"(vtype));
        if (vlmax == 0) {
          continue;
        }

        SIMD128 result[8];
        // Set undisturbed result vector registers.
        for (size_t index = 0; index < 8; ++index) {
          result[index] = kUndisturbedResult;
        }

        // Exectations for reductions are for swapped source arguments.
        SIMD128 two_sources[16]{};
        memcpy(&two_sources[0], &source[8], sizeof(two_sources[0]) * 8);
        memcpy(&two_sources[8], &source[0], sizeof(two_sources[0]) * 8);

        RunTwoVectorArgsOneRes(exec_insn, &two_sources[0], &result[0], vtype, vlmax);

        // Reduction instructions are unique in that they produce a scalar
        // output to a single vector register as opposed to a register group.
        // This allows us to take some short-cuts when validating:
        //
        // - The mask setting is only useful during computation, as the body
        // of the destination is always only element 0, which will always be
        // written to, regardless of mask setting.
        // - The tail is guaranteed to be 1..VLEN/SEW, so the vlmul setting
        // does not affect the elements that the tail policy applies to in the
        // destination register.

        // Verify that the destination register holds the reduction in the
        // first element and the tail policy applies to the remaining.
        SIMD128 expected_result_register = vta ? kAgnosticResult : kUndisturbedResult;
        size_t vsew_bits = 8 << vsew;
        expected_result_register = (expected_result_register >> vsew_bits) << vsew_bits;
        expected_result_register |= expected_result;
        EXPECT_TRUE(result[0] == expected_result_register) << " vtype=" << vtype;

        // Verify all non-destination registers are undisturbed.
        for (size_t index = 1; index < 8; ++index) {
          EXPECT_TRUE(result[index] == kUndisturbedResult) << " vtype=" << vtype;
        }
      }
    }
  };

  for (int vlmul = 0; vlmul < 8; vlmul++) {
    ((Verify(exec_insn,
             BitUtilLog2(sizeof(ExpectedResultType)),
             vlmul,
             std::get<0>(expected_result)[vlmul]),
      Verify(exec_masked_insn,
             BitUtilLog2(sizeof(ExpectedResultType)),
             vlmul,
             std::get<1>(expected_result)[vlmul])),
     ...);
  }
}

void TestVectorReductionInstruction(ExecInsnFunc exec_insn,
                                    ExecInsnFunc exec_masked_insn,
                                    const uint8_t (&expected_result_vd0_int8)[8],
                                    const uint16_t (&expected_result_vd0_int16)[8],
                                    const uint32_t (&expected_result_vd0_int32)[8],
                                    const uint64_t (&expected_result_vd0_int64)[8],
                                    const uint8_t (&expected_result_vd0_with_mask_int8)[8],
                                    const uint16_t (&expected_result_vd0_with_mask_int16)[8],
                                    const uint32_t (&expected_result_vd0_with_mask_int32)[8],
                                    const uint64_t (&expected_result_vd0_with_mask_int64)[8],
                                    const SIMD128 (&source)[16]) {
  TestVectorReductionInstruction(
      exec_insn,
      exec_masked_insn,
      source,
      std::tuple<const uint8_t(&)[8], const uint8_t(&)[8]>{expected_result_vd0_int8,
                                                           expected_result_vd0_with_mask_int8},
      std::tuple<const uint16_t(&)[8], const uint16_t(&)[8]>{expected_result_vd0_int16,
                                                             expected_result_vd0_with_mask_int16},
      std::tuple<const uint32_t(&)[8], const uint32_t(&)[8]>{expected_result_vd0_int32,
                                                             expected_result_vd0_with_mask_int32},
      std::tuple<const uint64_t(&)[8], const uint64_t(&)[8]>{expected_result_vd0_int64,
                                                             expected_result_vd0_with_mask_int64});
}

[[gnu::naked]] void ExecVredsum() {
  asm("vredsum.vs v8,v16,v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVredsum() {
  asm("vredsum.vs v8,v16,v24,v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVredsum) {
  TestVectorReductionInstruction(
      ExecVredsum,
      ExecMaskedVredsum,
      // expected_result_vd0_int8
      {242, 228, 200, 144, /* unused */ 0, 146, 44, 121},
      // expected_result_vd0_int16
      {0x0172, 0x82e4, 0x88c8, 0xa090, /* unused */ 0, 0x1300, 0xa904, 0xe119},
      // expected_result_vd0_int32
      {0xcb44'b932,
       0x9407'71e4,
       0xa70e'64c8,
       0xd312'5090,
       /* unused */ 0,
       /* unused */ 0,
       0x1907'1300,
       0xb713'ad09},
      // expected_result_vd0_int64
      {0xb32f'a926'9f1b'9511,
       0x1f99'0d88'fb74'e962,
       0xb92c'970e'74e8'52c4,
       0xef4e'ad14'6aca'2888,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x2513'1f0e'1907'1300},
      // expected_result_vd0_with_mask_int8
      {39, 248, 142, 27, /* unused */ 0, 0, 154, 210},
      // expected_result_vd0_with_mask_int16
      {0x5f45, 0xc22f, 0x99d0, 0x98bf, /* unused */ 0, 0x1300, 0x1300, 0x4b15},
      // expected_result_vd0_with_mask_int32
      {0x2d38'1f29,
       0x99a1'838a,
       0x1989'ef5c,
       0x9cf4'4aa1,
       /* unused */ 0,
       /* unused */ 0,
       0x1907'1300,
       0x1907'1300},
      // expected_result_vd0_with_mask_int64
      {0x2513'1f0e'1907'1300,
       0x917c'8370'7560'6751,
       0x4e56'3842'222a'0c13,
       0xc833'9e0e'73df'49b5,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x2513'1f0e'1907'1300},
      kVectorCalculationsSource);
}

}  // namespace
