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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <utility>

namespace {

template <typename T>
constexpr T BitUtilLog2(T x) {
  return __builtin_ctz(x);
}

using uint8_16_t = std::tuple<uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t,
                              uint8_t>;
using uint16_8_t =
    std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>;
using uint32_4_t = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;
using uint64_2_t = std::tuple<uint64_t, uint64_t>;

enum PrintModeEndianess { kLittleEndian, kBigEndian };

// A wrapper around __uint128 which can be constructed from a pair of uint64_t literals.
class SIMD128 {
 public:
  SIMD128(){};

  constexpr SIMD128(uint8_16_t u8) : uint8_{u8} {};
  constexpr SIMD128(uint16_8_t u16) : uint16_{u16} {};
  constexpr SIMD128(uint32_4_t u32) : uint32_{u32} {};
  constexpr SIMD128(uint64_2_t u64) : uint64_{u64} {};
  constexpr SIMD128(__uint128_t u128) : u128_{u128} {};

  [[nodiscard]] constexpr __uint128_t Get() const { return u128_; }

  constexpr SIMD128& operator=(const SIMD128& other) {
    u128_ = other.u128_;
    return *this;
  };
  constexpr SIMD128& operator|=(const SIMD128& other) {
    u128_ |= other.u128_;
    return *this;
  }

  constexpr bool operator==(const SIMD128& other) const { return u128_ == other.u128_; }
  constexpr bool operator!=(const SIMD128& other) const { return u128_ != other.u128_; }
  constexpr SIMD128 operator>>(size_t shift_amount) const { return u128_ >> shift_amount; }
  constexpr SIMD128 operator<<(size_t shift_amount) const { return u128_ << shift_amount; }
  constexpr SIMD128 operator&(SIMD128 other) const { return u128_ & other.u128_; }
  constexpr SIMD128 operator|(SIMD128 other) const { return u128_ | other.u128_; }
  constexpr SIMD128 operator^(SIMD128 other) const { return u128_ ^ other.u128_; }
  constexpr SIMD128 operator~() const { return ~u128_; }
  friend std::ostream& operator<<(std::ostream& os, const SIMD128& simd);

  template <size_t N>
  std::ostream& Print(std::ostream& os) const {
    if constexpr (kSimd128PrintMode == kBigEndian) {
      os << std::uppercase << std::hex << std::setw(4) << std::setfill('0') << std::get<N>(uint16_);
      if constexpr (N > 0) {
        os << '\'';
      }
    } else {
      os << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
         << static_cast<int>(std::get<N * 2>(uint8_));
      os << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
         << static_cast<int>(std::get<N * 2 + 1>(uint8_));
      if constexpr (N < 7) {
        os << '\'';
      }
    }
    return os;
  }

  template <size_t... N>
  std::ostream& PrintEach(std::ostream& os, std::index_sequence<N...>) const {
    os << "0x";
    if constexpr (kSimd128PrintMode == kBigEndian) {
      (Print<7 - N>(os), ...);
    } else {
      (Print<N>(os), ...);
    }
    return os;
  }

 private:
  union {
#ifdef __GNUC__
    [[gnu::may_alias]] uint8_16_t uint8_;
    [[gnu::may_alias]] uint16_8_t uint16_;
    [[gnu::may_alias]] uint32_4_t uint32_;
    [[gnu::may_alias]] uint64_2_t uint64_;
    [[gnu::may_alias]] __uint128_t u128_;
#endif
  };

  // Support for BIG_ENDIAN or LITTLE_ENDIAN printing of SIMD128 values. Change this value
  // if you want to see failure results in LITTLE_ENDIAN.
  static constexpr const PrintModeEndianess kSimd128PrintMode = kBigEndian;
};

// Helps produce easy to read output on failed tests.
std::ostream& operator<<(std::ostream& os, const SIMD128& simd) {
  return simd.PrintEach(os, std::make_index_sequence<8>());
}

constexpr SIMD128 kVectorCalculationsSourceLegacy[16] = {
    {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908}},
    {{0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918}},
    {{0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928}},
    {{0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938}},
    {{0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948}},
    {{0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958}},
    {{0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968}},
    {{0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},

    {{0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211}},
    {{0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231}},
    {{0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251}},
    {{0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271}},
    {{0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291}},
    {{0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1}},
    {{0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1}},
    {{0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
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
// Mask used with vsew = 0 (8bit) elements.
static constexpr SIMD128 kMaskInt8[8] = {
    {{255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255}},
    {{255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255}},
    {{255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255}},
    {{255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 0, 255, 255}},
    {{255, 0, 255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0}},
    {{255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 255}},
    {{255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0}},
    {{255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 0}},
};
// Mask used with vsew = 1 (16bit) elements.
static constexpr SIMD128 kMaskInt16[8] = {
    {{0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff}},
    {{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff}},
    {{0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000}},
    {{0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff}},
    {{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff}},
    {{0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff}},
    {{0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff}},
    {{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff}},
};
// Mask used with vsew = 2 (32bit) elements.
static constexpr SIMD128 kMaskInt32[8] = {
    {{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff}},
    {{0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff}},
    {{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0x0000'0000}},
    {{0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff}},
    {{0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff}},
    {{0x0000'0000, 0xffff'ffff, 0xffff'ffff, 0x0000'0000}},
    {{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff}},
    {{0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff}},
};
// Mask used with vsew = 3 (64bit) elements.
static constexpr SIMD128 kMaskInt64[8] = {
    {{0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff}},
    {{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000}},
    {{0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff}},
};
// To verify operations without masking.
static constexpr SIMD128 kNoMask[8] = {
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
    {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
};

// Half of sub-register lmul.
static constexpr SIMD128 kFractionMaskInt8[5] = {
    {{255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},                // Half of 1/8 reg = 1/16
    {{255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},              // Half of 1/4 reg = 1/8
    {{255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},          // Half of 1/2 reg = 1/4
    {{255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0}},  // Half of full reg = 1/2
    {{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}},  // Full reg
};

template <typename ElementType>
auto MaskForElem() {
  if constexpr (std::is_same_v<ElementType, uint8_t>) {
    return kMaskInt8;
  } else if constexpr (std::is_same_v<ElementType, uint16_t>) {
    return kMaskInt16;
  } else if constexpr (std::is_same_v<ElementType, uint32_t>) {
    return kMaskInt32;
  } else if constexpr (std::is_same_v<ElementType, uint64_t>) {
    return kMaskInt64;
  } else {
    static_assert(false);
  }
}

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

// Supports ExecInsnFuncs that fit the following [inputs...] -> output formats:
//   vector -> vector
//   vector, vector -> vector
//   vector, scalar -> vector
//   vector, float -> vector
// Vectors will be used in v16 first, then v24
// scalar and float will be filled from scalar_src, and will use t0 and ft0,
// respectively.
void RunCommonVectorFunc(ExecInsnFunc exec_insn,
                         const SIMD128* src,
                         SIMD128* res,
                         uint64_t scalar_src,
                         uint64_t vstart,
                         uint64_t vtype,
                         uint64_t vlin) {
  uint64_t vl = vlin;
  // Mask register is, unconditionally, v0, and we need 8 or 24 to handle full 8-registers
  // inputs thus we use v8..v15 for destination and place sources into v24..v31.
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
      "vsetvl t0, %[vl], %[vtype]\n\t"
      "csrw vstart, %[vstart]\n\t"
      "mv t0, %[scalar_src]\n\t"
      "fmv.d.x ft0, %[scalar_src]\n\t"
      "jalr %[exec_insn]\n\t"
      // Save vstart and vl just after insn execution for checks.
      "csrr %[vstart], vstart\n\t"
      "csrr %[vl], vl\n\t"
      // Store the result.
      "vsetvli t0, zero, e64, m8, ta, ma\n\t"
      "vse64.v v8, (%[res])\n\t"
      : [vstart] "=r"(vstart), [vl] "=r"(vl)
      : [exec_insn] "r"(exec_insn),
        [src] "r"(src),
        [res] "r"(res),
        [vtype] "r"(vtype),
        "0"(vstart),
        "1"(vl),
        [mask] "r"(&kMask),
        [scalar_src] "r"(scalar_src)
      : "t0",
        "ra",
        "ft0",
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
  EXPECT_EQ(vl, vlin);
}

enum class TestVectorInstructionKind { kInteger, kFloat };
enum class TestVectorInstructionMode { kDefault, kWidening, kNarrowing, kVMerge };

template <TestVectorInstructionKind kTestVectorInstructionKind,
          TestVectorInstructionMode kTestVectorInstructionMode,
          typename... ExpectedResultType,
          size_t... kResultsCount>
void TestVectorInstructionInternal(ExecInsnFunc exec_insn,
                                   ExecInsnFunc exec_masked_insn,
                                   const SIMD128 dst_result,
                                   const SIMD128 (&source)[16],
                                   const ExpectedResultType (&... expected_result)[kResultsCount]) {
  auto Verify = [&source, dst_result](ExecInsnFunc exec_insn,
                                      uint8_t vsew,
                                      const auto& expected_result,
                                      const auto& mask) {
    uint64_t scalar_src = 0;
    if constexpr (kTestVectorInstructionKind == TestVectorInstructionKind::kInteger) {
      // Set t0 for vx instructions.
      scalar_src = 0xaaaa'aaaa'aaaa'aaaa;
    } else {
      // We only support Float32/Float64 for float instructions, but there are conversion
      // instructions that work with double width floats.
      // These instructions never use float registers though and thus we don't need to store
      // anything into ft0 register, if they are used.
      // For Float32/Float64 case we load 5.625 of the appropriate type into ft0.
      ASSERT_LE(vsew, 3);
      if (vsew == 2) {
        scalar_src = 0xffff'ffff'40b4'0000;  // float 5.625
      } else if (vsew == 3) {
        scalar_src = 0x4016'8000'0000'0000;  // double 5.625
      }
    }
    for (uint8_t vlmul = 0; vlmul < 8; ++vlmul) {
      if constexpr (kTestVectorInstructionMode == TestVectorInstructionMode::kNarrowing ||
                    kTestVectorInstructionMode == TestVectorInstructionMode::kWidening) {
        // Incompatible vlmul for narrowing.
        if (vlmul == 3) {
          continue;
        }
      }
      for (uint8_t vta = 0; vta < 2; ++vta) {
        for (uint8_t vma = 0; vma < 2; ++vma) {
          uint64_t vtype = (vma << 7) | (vta << 6) | (vsew << 3) | vlmul;
          uint64_t vlmax = 0;
          asm("vsetvl %0, zero, %1" : "=r"(vlmax) : "r"(vtype));
          // Incompatible vsew and vlmax. Skip it.
          if (vlmax == 0) {
            continue;
          }
          uint8_t emul =
              (vlmul + (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening)) &
              0b111;

          // To make tests quick enough we don't test vstart and vl change with small register
          // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of first
          // register, last register and half of next-to last register.
          // Don't use vlmul == 3 because that one may not be supported if instruction widens the
          // result.
          uint64_t vstart;
          uint64_t vl;
          if (emul == 2) {
            vstart = vlmax / 8;
            vl = (vlmax * 5) / 8;
          } else {
            vstart = 0;
            vl = vlmax;
          }

          SIMD128 result[8];
          // Set expected_result vector registers into 0b01010101… pattern.
          // Set undisturbed result vector registers.
          std::fill_n(result, 8, dst_result);

          RunCommonVectorFunc(exec_insn, &source[0], &result[0], scalar_src, vstart, vtype, vl);

          // Values for inactive elements (i.e. corresponding mask bit is 0).
          SIMD128 expected_inactive[8];
          if constexpr (kTestVectorInstructionMode == TestVectorInstructionMode::kVMerge) {
            // vs2 is the start of the source vector register group.
            // Note: copy_n input/output args are backwards compared to fill_n below.
            std::copy_n(source, 8, expected_inactive);
          } else {
            // For most instructions, follow basic inactive processing rules based on vma flag.
            std::fill_n(expected_inactive, 8, (vma ? kAgnosticResult : dst_result));
          }

          if (emul < 4) {
            for (size_t index = 0; index < 1 << emul; ++index) {
              if (index == 0 && emul == 2) {
                EXPECT_EQ(result[index],
                          ((dst_result & kFractionMaskInt8[3]) |
                           (SIMD128{expected_result[index]} & mask[index] & ~kFractionMaskInt8[3]) |
                           (expected_inactive[index] & ~mask[index] & ~kFractionMaskInt8[3])));
              } else if (index == 2 && emul == 2) {
                EXPECT_EQ(result[index],
                          ((SIMD128{expected_result[index]} & mask[index] & kFractionMaskInt8[3]) |
                           (expected_inactive[index] & ~mask[index] & kFractionMaskInt8[3]) |
                           ((vta ? kAgnosticResult : dst_result) & ~kFractionMaskInt8[3])));
              } else if (index == 3 && emul == 2 && vta) {
                EXPECT_EQ(result[index], kAgnosticResult);
              } else if (index == 3 && emul == 2) {
                EXPECT_EQ(result[index], dst_result);
              } else {
                EXPECT_EQ(result[index],
                          ((SIMD128{expected_result[index]} & mask[index]) |
                           ((expected_inactive[index] & ~mask[index]))));
              }
            }
          } else {
            EXPECT_EQ(result[0],
                      ((SIMD128{expected_result[0]} & mask[0] & kFractionMaskInt8[emul - 4]) |
                       (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[emul - 4]) |
                       ((vta ? kAgnosticResult : dst_result) & ~kFractionMaskInt8[emul - 4])));
          }
        }
      }
    }
  };

  ((Verify(exec_insn,
           BitUtilLog2(sizeof(std::tuple_element_t<0, ExpectedResultType>)) -
               (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening),
           expected_result,
           kNoMask),
    Verify(exec_masked_insn,
           BitUtilLog2(sizeof(std::tuple_element_t<0, ExpectedResultType>)) -
               (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening),
           expected_result,
           MaskForElem<std::tuple_element_t<0, ExpectedResultType>>())),
   ...);
}

template <TestVectorInstructionKind kTestVectorInstructionKind,
          TestVectorInstructionMode kTestVectorInstructionMode,
          typename... ExpectedResultType,
          size_t... kResultsCount>
void TestVectorInstruction(ExecInsnFunc exec_insn,
                           ExecInsnFunc exec_masked_insn,
                           const SIMD128 (&source)[16],
                           const ExpectedResultType (&... expected_result)[kResultsCount]) {
  TestVectorInstructionInternal<kTestVectorInstructionKind, kTestVectorInstructionMode>(
      exec_insn, exec_masked_insn, kUndisturbedResult, source, expected_result...);
}

void TestVectorInstruction(ExecInsnFunc exec_insn,
                           ExecInsnFunc exec_masked_insn,
                           const uint8_16_t (&expected_result_int8)[8],
                           const uint16_8_t (&expected_result_int16)[8],
                           const uint32_4_t (&expected_result_int32)[8],
                           const uint64_2_t (&expected_result_int64)[8],
                           const SIMD128 (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kInteger, TestVectorInstructionMode::kDefault>(
      exec_insn,
      exec_masked_insn,
      source,
      expected_result_int8,
      expected_result_int16,
      expected_result_int32,
      expected_result_int64);
}

void TestVectorFloatInstruction(ExecInsnFunc exec_insn,
                                ExecInsnFunc exec_masked_insn,
                                const uint32_4_t (&expected_result_int32)[8],
                                const uint64_2_t (&expected_result_int64)[8],
                                const SIMD128 (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kDefault>(
      exec_insn, exec_masked_insn, source, expected_result_int32, expected_result_int64);
}

void TestNarrowingVectorFloatInstruction(ExecInsnFunc exec_insn,
                                         ExecInsnFunc exec_masked_insn,
                                         const uint32_4_t (&expected_result_int32)[4],
                                         const SIMD128 (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kNarrowing>(
      exec_insn, exec_masked_insn, source, expected_result_int32);
}

void TestNarrowingVectorFloatInstruction(ExecInsnFunc exec_insn,
                                         ExecInsnFunc exec_masked_insn,
                                         const uint16_8_t (&expected_result_int16)[4],
                                         const uint32_4_t (&expected_result_int32)[4],
                                         const SIMD128 (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kNarrowing>(
      exec_insn, exec_masked_insn, source, expected_result_int16, expected_result_int32);
}

void TestWideningVectorFloatInstruction(ExecInsnFunc exec_insn,
                                        ExecInsnFunc exec_masked_insn,
                                        const uint64_2_t (&expected_result_int64)[8],
                                        const SIMD128 (&source)[16],
                                        SIMD128 dst_result = kUndisturbedResult) {
  TestVectorInstructionInternal<TestVectorInstructionKind::kFloat,
                                TestVectorInstructionMode::kWidening>(
      exec_insn, exec_masked_insn, dst_result, source, expected_result_int64);
}

void TestWideningVectorFloatInstruction(ExecInsnFunc exec_insn,
                                        ExecInsnFunc exec_masked_insn,
                                        const uint32_4_t (&expected_result_int32)[8],
                                        const uint64_2_t (&expected_result_int64)[8],
                                        const SIMD128 (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kWidening>(
      exec_insn, exec_masked_insn, source, expected_result_int32, expected_result_int64);
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
        EXPECT_EQ(result[0], expected_result_register) << " vtype=" << vtype;

        // Verify all non-destination registers are undisturbed.
        for (size_t index = 1; index < 8; ++index) {
          EXPECT_EQ(result[index], kUndisturbedResult) << " vtype=" << vtype;
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
                                    const uint32_t (&expected_result_vd0_int32)[8],
                                    const uint64_t (&expected_result_vd0_int64)[8],
                                    const uint32_t (&expected_result_vd0_with_mask_int32)[8],
                                    const uint64_t (&expected_result_vd0_with_mask_int64)[8],
                                    const SIMD128 (&source)[16]) {
  TestVectorReductionInstruction(
      exec_insn,
      exec_masked_insn,
      source,
      std::tuple<const uint32_t(&)[8], const uint32_t(&)[8]>{expected_result_vd0_int32,
                                                             expected_result_vd0_with_mask_int32},
      std::tuple<const uint64_t(&)[8], const uint64_t(&)[8]>{expected_result_vd0_int64,
                                                             expected_result_vd0_with_mask_int64});
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

// clang-format off
#define DEFINE_TWO_ARG_ONE_RES_FUNCTION(Name, Asm) \
  [[gnu::naked]] void Exec##Name() {               \
    asm(#Asm " v8,v16,v24\n\t"                     \
        "ret\n\t");                                \
  }                                                \
  [[gnu::naked]] void ExecMasked##Name() {         \
    asm(#Asm " v8,v16,v24,v0.t\n\t"                \
        "ret\n\t");                                \
  }
// clang-format on

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredsum, vredsum.vs)

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

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vfredosum, vfredosum.vs)

TEST(InlineAsmTestRiscv64, TestVfredosum) {
  TestVectorReductionInstruction(ExecVfredosum,
                                 ExecMaskedVfredosum,
                                 // expected_result_vd0_int32
                                 {0x9e0c'9a8e,
                                  0xbe2c'bace,
                                  0xfe6c'fb4e,
                                  0x7e6b'fc4d,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9e0c'9a8e},
                                 // expected_result_vd0_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xfe6c'fa69'f664'f260,
                                  0x7eec'5def'0cee'0dee,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 // expected_result_vd0_with_mask_int32
                                 {0x9604'929d,
                                  0xbe2c'ba29,
                                  0xfe6c'fb4e,
                                  0x7e6b'fa84,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9604'9200},
                                 // expected_result_vd0_with_mask_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xee7c'ea78'e674'e271,
                                  0x6efc'4e0d'ee0d'ee0f,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vfredusum, vfredusum.vs)

// Currently Vfredusum is implemented as Vfredosum (as explicitly permitted by RVV 1.0).
// If we would implement some speedups which would change results then we may need to alter tests.
TEST(InlineAsmTestRiscv64, TestVfredusum) {
  TestVectorReductionInstruction(ExecVfredusum,
                                 ExecMaskedVfredusum,
                                 // expected_result_vd0_int32
                                 {0x9e0c'9a8e,
                                  0xbe2c'bace,
                                  0xfe6c'fb4e,
                                  0x7e6b'fc4d,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9e0c'9a8e},
                                 // expected_result_vd0_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xfe6c'fa69'f664'f260,
                                  0x7eec'5def'0cee'0dee,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 // expected_result_vd0_with_mask_int32
                                 {0x9604'929d,
                                  0xbe2c'ba29,
                                  0xfe6c'fb4e,
                                  0x7e6b'fa84,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9604'9200},
                                 // expected_result_vd0_with_mask_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xee7c'ea78'e674'e271,
                                  0x6efc'4e0d'ee0d'ee0f,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredand, vredand.vs)

TEST(InlineAsmTestRiscv64, TestVredand) {
  TestVectorReductionInstruction(
      ExecVredand,
      ExecMaskedVredand,
      // expected_result_vd0_int8
      {0, 0, 0, 0, /* unused */ 0, 0, 0, 0},
      // expected_result_vd0_int16
      {0x8000, 0x8000, 0x8000, 0x0000, /* unused */ 0, 0x8000, 0x8000, 0x8000},
      // expected_result_vd0_int32
      {0x8200'8000,
       0x8200'8000,
       0x8200'8000,
       0x0200'0000,
       /* unused */ 0,
       /* unused */ 0,
       0x8200'8000,
       0x8200'8000},
      // expected_result_vd0_int64
      {0x8604'8000'8200'8000,
       0x8604'8000'8200'8000,
       0x8604'8000'8200'8000,
       0x0604'0000'0200'0000,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8604'8000'8200'8000},
      // expected_result_vd0_with_mask_int8
      {0, 0, 0, 0, /* unused */ 0, 0, 0, 0},
      // expected_result_vd0_with_mask_int16
      {0x8000, 0x8000, 0x8000, 0x0000, /* unused */ 0, 0x8000, 0x8000, 0x8000},
      // expected_result_vd0_with_mask_int32
      {0x8200'8000,
       0x8200'8000,
       0x8200'8000,
       0x0200'0000,
       /* unused */ 0,
       /* unused */ 0,
       0x8200'8000,
       0x8200'8000},
      // expected_result_vd0_with_mask_int64
      {0x8604'8000'8200'8000,
       0x8604'8000'8200'8000,
       0x8604'8000'8200'8000,
       0x0604'0000'0200'0000,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8604'8000'8200'8000},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredor, vredor.vs)

TEST(InlineAsmTestRiscv64, TestVredor) {
  TestVectorReductionInstruction(
      ExecVredor,
      ExecMaskedVredor,
      // expected_result_vd0_int8
      {159, 191, 255, 255, /* unused */ 0, 146, 150, 159},
      // expected_result_vd0_int16
      {0x9f1d, 0xbf3d, 0xff7d, 0xfffd, /* unused */ 0, 0x9300, 0x9704, 0x9f0d},
      // expected_result_vd0_int32
      {0x9f1e'9b19,
       0xbf3e'bb39,
       0xff7e'fb79,
       0xfffe'fbf9,
       /* unused */ 0,
       /* unused */ 0,
       0x9706'9300,
       0x9f0e'9b09},
      // expected_result_vd0_int64
      {0x9f1e'9f1d'9716'9311,
       0xbf3e'bf3d'b736'b331,
       0xff7e'ff7d'f776'f371,
       0xfffe'fffd'f7f6'f3f1,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9f0e'9f0d'9706'9300},
      // expected_result_vd0_with_mask_int8
      {159, 191, 255, 255, /* unused */ 0, 0, 150, 158},
      // expected_result_vd0_with_mask_int16
      {0x9f1d, 0xbf3d, 0xff7d, 0xfffd, /* unused */ 0, 0x9300, 0x9300, 0x9f0d},
      // expected_result_vd0_with_mask_int32
      {0x9f1e'9b19,
       0xbf3e'bb39,
       0xff7e'fb79,
       0xfffe'fbf9,
       /* unused */ 0,
       /* unused */ 0,
       0x9706'9300,
       0x9706'9300},
      // expected_result_vd0_with_mask_int64
      {0x9f0e'9f0d'9706'9300,
       0xbf3e'bf3d'b736'b331,
       0xff7e'ff7d'f776'f371,
       0xfffe'fffd'f7f6'f3f1,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9f0e'9f0d'9706'9300},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredxor, vredxor.vs)

TEST(InlineAsmTestRiscv64, TestVredxor) {
  TestVectorReductionInstruction(
      ExecVredxor,
      ExecMaskedVredxor,
      // expected_result_vd0_int8
      {0, 0, 0, 0, /* unused */ 0, 146, 0, 1},
      // expected_result_vd0_int16
      {0x8100, 0x8100, 0x8100, 0x8100, /* unused */ 0, 0x1300, 0x8504, 0x8101},
      // expected_result_vd0_int32
      {0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       0x1506'1300,
       0x8b0a'8909},
      // expected_result_vd0_int64
      {0x9716'9515'9312'9111,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x190a'1f0d'1506'1300},
      // expected_result_vd0_with_mask_int8
      {143, 154, 150, 43, /* unused */ 0, 0, 146, 150},
      // expected_result_vd0_with_mask_int16
      {0x1f0d, 0xbd3d, 0x9514, 0x8d0d, /* unused */ 0, 0x1300, 0x1300, 0x1705},
      // expected_result_vd0_with_mask_int32
      {0x1d0e'1b09,
       0x0d1e'0b18,
       0xfb7a'f978,
       0xab2a'a929,
       /* unused */ 0,
       /* unused */ 0,
       0x1506'1300,
       0x1506'1300},
      // expected_result_vd0_with_mask_int64
      {0x190a'1f0d'1506'1300,
       0x091a'0f1c'0516'0311,
       0x293a'2f3c'2536'2331,
       0x77f6'75f5'73f2'71f1,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x190a'1f0d'1506'1300},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredminu, vredminu.vs)

TEST(InlineAsmTestRiscv64, TestVredminu) {
  TestVectorReductionInstruction(
      ExecVredminu,
      ExecMaskedVredminu,
      // expected_result_vd0_int8
      {0, 0, 0, 0, /* unused */ 0, 0, 0, 0},
      // expected_result_vd0_int16
      {0x8100, 0x8100, 0x8100, 0x0291, /* unused */ 0, 0x8100, 0x8100, 0x8100},
      // expected_result_vd0_int32
      {0x83028100,
       0x83028100,
       0x83028100,
       0x06940291,
       /* unused */ 0,
       /* unused */ 0,
       0x83028100,
       0x83028100},
      // expected_result_vd0_int64
      {0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x0e9c'0a98'0694'0291,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8504'8302'8100},
      // expected_result_vd0_with_mask_int8
      {0, 0, 0, 0, /* unused */ 0, 0, 0, 0},
      // expected_result_vd0_with_mask_int16
      {0x8100, 0x8100, 0x8100, 0x0291, /* unused */ 0, 0x8100, 0x8100, 0x8100},
      // expected_result_vd0_with_mask_int32
      {0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       0x0e9c'0a98,
       /* unused */ 0,
       /* unused */ 0,
       0x8302'8100,
       0x8302'8100},
      // expected_result_vd0_with_mask_int64
      {0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x1e8c'1a89'1684'1280,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8504'8302'8100},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredmin, vredmin.vs)

TEST(InlineAsmTestRiscv64, TestVredmin) {
  TestVectorReductionInstruction(
      ExecVredmin,
      ExecMaskedVredmin,
      // expected_result_vd0_int8
      {130, 130, 130, 128, /* unused */ 0, 146, 146, 146},
      // expected_result_vd0_int16
      {0x8100, 0x8100, 0x8100, 0x8100, /* unused */ 0, 0x8100, 0x8100, 0x8100},
      // expected_result_vd0_int32
      {0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       0x8302'8100,
       0x8302'8100},
      // expected_result_vd0_int64
      {0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8504'8302'8100},
      // expected_result_vd0_with_mask_int8
      {138, 138, 138, 128, /* unused */ 0, 0, 150, 150},
      // expected_result_vd0_with_mask_int16
      {0x8100, 0x8100, 0x8100, 0x8100, /* unused */ 0, 0x8100, 0x8100, 0x8100},
      // expected_result_vd0_with_mask_int32
      {0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       0x8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       0x8302'8100,
       0x8302'8100},
      // expected_result_vd0_with_mask_int64
      {0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       0x8706'8504'8302'8100,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x8706'8504'8302'8100},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vfredmin, vfredmin.vs)

TEST(InlineAsmTestRiscv64, TestVfredmin) {
  TestVectorReductionInstruction(ExecVfredmin,
                                 ExecMaskedVfredmin,
                                 // expected_result_vd0_int32
                                 {0x9e0c'9a09,
                                  0xbe2c'ba29,
                                  0xfe6c'fa69,
                                  0xfe6c'fa69,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9e0c'9a09},
                                 // expected_result_vd0_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xfe6c'fa69'f664'f260,
                                  0xfe6c'fa69'f664'f260,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 // expected_result_vd0_with_mask_int32
                                 {0x9604'9200,
                                  0xbe2c'ba29,
                                  0xfe6c'fa69,
                                  0xfe6c'fa69,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9604'9200,
                                  0x9604'9200},
                                 // expected_result_vd0_with_mask_int64
                                 {0x9e0c'9a09'9604'9200,
                                  0xbe2c'ba29'b624'b220,
                                  0xee7c'ea78'e674'e271,
                                  0xee7c'ea78'e674'e271,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x9e0c'9a09'9604'9200},
                                 kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredmaxu, vredmaxu.vs)

TEST(InlineAsmTestRiscv64, TestVredmaxu) {
  TestVectorReductionInstruction(
      ExecVredmaxu,
      ExecMaskedVredmaxu,
      // expected_result_vd0_int8
      {158, 190, 254, 254, /* unused */ 0, 146, 150, 158},
      // expected_result_vd0_int16
      {0x9e0c, 0xbe2c, 0xfe6c, 0xfe6c, /* unused */ 0, 0x9200, 0x9604, 0x9e0c},
      // expected_result_vd0_int32
      {0x9e0c'9a09,
       0xbe2c'ba29,
       0xfe6c'fa69,
       0xfe6c'fa69,
       /* unused */ 0,
       /* unused */ 0,
       0x9604'9200,
       0x9e0c'9a09},
      // expected_result_vd0_int64
      {0x9e0c'9a09'9604'9200,
       0xbe2c'ba29'b624'b220,
       0xfe6c'fa69'f664'f260,
       0xfe6c'fa69'f664'f260,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9e0c'9a09'9604'9200},
      // expected_result_vd0_with_mask_int8
      {158, 186, 254, 254, /* unused */ 0, 0, 150, 158},
      // expected_result_vd0_with_mask_int16
      {0x9e0c, 0xba29, 0xfe6c, 0xfe6c, /* unused */ 0, 0x9200, 0x9200, 0x9e0c},
      // expected_result_vd0_with_mask_int32
      {0x9604'9200,
       0xbe2c'ba29,
       0xfe6c'fa69,
       0xfe6c'fa69,
       /* unused */ 0,
       /* unused */ 0,
       0x9604'9200,
       0x9604'9200},
      // expected_result_vd0_with_mask_int64
      {0x9e0c'9a09'9604'9200,
       0xbe2c'ba29'b624'b220,
       0xee7c'ea78'e674'e271,
       0xee7c'ea78'e674'e271,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9e0c'9a09'9604'9200},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vredmax, vredmax.vs)

TEST(InlineAsmTestRiscv64, TestVredmax) {
  TestVectorReductionInstruction(
      ExecVredmax,
      ExecMaskedVredmax,
      // expected_result_vd0_int8
      {28, 60, 124, 126, /* unused */ 0, 0, 4, 12},
      // expected_result_vd0_int16
      {0x9e0c, 0xbe2c, 0xfe6c, 0x7eec, /* unused */ 0, 0x9200, 0x9604, 0x9e0c},
      // expected_result_vd0_int32
      {0x9e0c'9a09,
       0xbe2c'ba29,
       0xfe6c'fa69,
       0x7eec'7ae9,
       /* unused */ 0,
       /* unused */ 0,
       0x9604'9200,
       0x9e0c'9a09},
      // expected_result_vd0_int64
      {0x9e0c'9a09'9604'9200,
       0xbe2c'ba29'b624'b220,
       0xfe6c'fa69'f664'f260,
       0x7eec'7ae9'76e4'72e0,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9e0c'9a09'9604'9200},
      // expected_result_vd0_with_mask_int8
      {24, 52, 124, 126, /* unused */ 0, 0, 4, 4},
      // expected_result_vd0_with_mask_int16
      {0x9e0c, 0xba29, 0xfe6c, 0x7ae9, /* unused */ 0, 0x9200, 0x9200, 0x9e0c},
      // expected_result_vd0_with_mask_int32
      {0x9604'9200,
       0xbe2c'ba29,
       0xfe6c'fa69,
       0x7eec'7ae9,
       /* unused */ 0,
       /* unused */ 0,
       0x9604'9200,
       0x9604'9200},
      // expected_result_vd0_with_mask_int64
      {0x9e0c'9a09'9604'9200,
       0xbe2c'ba29'b624'b220,
       0xee7c'ea78'e674'e271,
       0x6efc'6af8'66f4'62f1,
       /* unused */ 0,
       /* unused */ 0,
       /* unused */ 0,
       0x9e0c'9a09'9604'9200},
      kVectorCalculationsSource);
}

DEFINE_TWO_ARG_ONE_RES_FUNCTION(Vfredmax, vfredmax.vs)

TEST(InlineAsmTestRiscv64, TestVfredmax) {
  TestVectorReductionInstruction(ExecVfredmax,
                                 ExecMaskedVfredmax,
                                 // expected_result_vd0_int32
                                 {0x8302'8100,
                                  0x8302'8100,
                                  0x8302'8100,
                                  0x7eec'7ae9,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x8302'8100,
                                  0x8302'8100},
                                 // expected_result_vd0_int64
                                 {0x8706'8504'8302'8100,
                                  0x8706'8504'8302'8100,
                                  0x8706'8504'8302'8100,
                                  0x7eec'7ae9'76e4'72e0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x8706'8504'8302'8100},
                                 // expected_result_vd0_with_mask_int32
                                 {0x8302'8100,
                                  0x8302'8100,
                                  0x8302'8100,
                                  0x7eec'7ae9,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x8302'8100,
                                  0x8302'8100},
                                 // expected_result_vd0_with_mask_int64
                                 {0x8706'8504'8302'8100,
                                  0x8706'8504'8302'8100,
                                  0x8706'8504'8302'8100,
                                  0x6efc'6af8'66f4'62f1,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  /* unused */ 0,
                                  0x8706'8504'8302'8100},
                                 kVectorCalculationsSource);
}

#undef DEFINE_TWO_ARG_ONE_RES_FUNCTION

[[gnu::naked]] void ExecVfsqrtv() {
  asm("vfsqrt.v v8,v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfsqrtv() {
  asm("vfsqrt.v v8,v24,v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVfsqrtv) {
  TestVectorFloatInstruction(ExecVfsqrtv,
                             ExecMaskedVfsqrtv,
                             {{0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x2b02'052b, 0x2f05'ea47, 0x2309'a451, 0x270d'53b1},
                              {0x3b10'f937, 0x3f14'7a09, 0x3317'd8b1, 0x371b'31d0},
                              {0x4b1e'85c1, 0x4f21'bb83, 0x4324'd4da, 0x4727'ebbf},
                              {0x5b2b'0054, 0x5f2d'fb2f, 0x5330'dd9e, 0x5733'bf97}},
                             {{0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x2f3d'fd15'c59f'19b3, 0x2745'2e80'5593'4661},
                              {0x3f4e'0e34'c013'd37a, 0x3755'3a9e'ffea'ec9f},
                              {0x4f5e'1f49'ff52'69b6, 0x4765'46b6'c2dc'cddd},
                              {0x5f6e'3055'93df'fb07, 0x5775'52c7'aa27'df73}},
                             kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVfcvtxufv() {
  asm("vfcvt.xu.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtxufv() {
  asm("vfcvt.xu.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfcvtxfv() {
  asm("vfcvt.x.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtxfv() {
  asm("vfcvt.x.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfcvtfxuv() {
  asm("vfcvt.f.xu.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtfxuv() {
  asm("vfcvt.f.xu.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfcvtfxv() {
  asm("vfcvt.f.x.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtfxv() {
  asm("vfcvt.f.x.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfcvtrtzxuf() {
  asm("vfcvt.rtz.xu.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtrtzxuf() {
  asm("vfcvt.rtz.xu.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfcvtrtzxf() {
  asm("vfcvt.rtz.x.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfcvtrtzxf() {
  asm("vfcvt.rtz.x.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtxufv() {
  asm("vfwcvt.xu.f.v v8, v28\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtxufv() {
  asm("vfwcvt.xu.f.v v8, v28, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtxfv() {
  asm("vfwcvt.x.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtxfv() {
  asm("vfwcvt.x.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtffv() {
  asm("vfwcvt.f.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtffv() {
  asm("vfwcvt.f.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtfxuv() {
  asm("vfwcvt.f.xu.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtfxuv() {
  asm("vfwcvt.f.xu.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtfxv() {
  asm("vfwcvt.f.x.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtfxv() {
  asm("vfwcvt.f.x.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtrtzxuf() {
  asm("vfwcvt.rtz.xu.f.v v8, v28\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtrtzxuf() {
  asm("vfwcvt.rtz.xu.f.v v8, v28, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwcvtrtzxf() {
  asm("vfwcvt.rtz.x.f.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwcvtrtzxf() {
  asm("vfwcvt.rtz.x.f.v v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtxufw() {
  asm("vfncvt.xu.f.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtxufw() {
  asm("vfncvt.xu.f.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtxfw() {
  asm("vfncvt.x.f.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtxfw() {
  asm("vfncvt.x.f.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtffw() {
  asm("vfncvt.f.f.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtffw() {
  asm("vfncvt.f.f.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtfxuw() {
  asm("vfncvt.f.xu.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtfxuw() {
  asm("vfncvt.f.xu.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtfxw() {
  asm("vfncvt.f.x.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtfxw() {
  asm("vfncvt.f.x.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtrtzxuf() {
  asm("vfncvt.rtz.xu.f.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtrtzxuf() {
  asm("vfncvt.rtz.xu.f.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfncvtrtzxfw() {
  asm("vfncvt.rtz.x.f.w v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfncvtrtzxfw() {
  asm("vfncvt.rtz.x.f.w v8, v24, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVfcvtxfv) {
  TestVectorFloatInstruction(ExecVfcvtxufv,
                             ExecMaskedVfcvtxufv,
                             {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0xffff'ffff, 0xffff'ffff, 0x0000'6a21, 0x6e25'6c00},
                              {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
                             {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
                              {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfcvtxfv,
                             ExecMaskedVfcvtxfv,
                             {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x8000'0000, 0x8000'0000, 0xffff'cacf, 0xc8cd'6a00},
                              {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x7fff'ffff, 0x7fff'ffff, 0x0000'6a21, 0x6e25'6c00},
                              {0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff}},
                             {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                              {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x7fff'ffff'ffff'ffff, 0x7fff'ffff'ffff'ffff},
                              {0x7fff'ffff'ffff'ffff, 0x7fff'ffff'ffff'ffff}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfcvtfxuv,
                             ExecMaskedVfcvtfxuv,
                             {{0x4f16'0492, 0x4f1e'0c9a, 0x4f06'1482, 0x4f0e'1c8a},
                              {0x4f36'24b2, 0x4f3e'2cba, 0x4f26'34a2, 0x4f2e'3caa},
                              {0x4f56'44d2, 0x4f5e'4cda, 0x4f46'54c2, 0x4f4e'5cca},
                              {0x4f76'64f2, 0x4f7e'6cfa, 0x4f66'74e2, 0x4f6e'7cea},
                              {0x4db4'2094, 0x4df4'60d4, 0x4cd2'8052, 0x4d69'c0aa},
                              {0x4e5a'90ca, 0x4e7a'b0eb, 0x4e1a'd08b, 0x4e3a'f0ab},
                              {0x4ead'88a6, 0x4ebd'98b6, 0x4e8d'a886, 0x4e9d'b896},
                              {0x4eed'c8e6, 0x4efd'd8f6, 0x4ecd'e8c6, 0x4edd'f8d6}},
                             {{0x43e3'c193'4132'c092, 0x43e1'c391'4310'c290},
                              {0x43e7'c597'4536'c496, 0x43e5'c795'4714'c694},
                              {0x43eb'c99b'493a'c89a, 0x43e9'cb99'4b18'ca98},
                              {0x43ef'cd9f'4d3e'cc9e, 0x43ed'cf9d'4f1c'ce9c},
                              {0x43be'8c1a'8916'8412, 0x43ad'3815'300d'2805},
                              {0x43cf'561d'549b'5219, 0x43c7'5e15'5c13'5a11},
                              {0x43d7'b316'b255'b115, 0x43d3'b712'b611'b511},
                              {0x43df'bb1e'ba5d'b91d, 0x43db'bf1a'be19'bd19}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfcvtfxv,
                             ExecMaskedVfcvtfxv,
                             {{0xced3'f6dc, 0xcec3'e6cc, 0xcef3'd6fc, 0xcee3'c6ec},
                              {0xce93'b69c, 0xce83'a68c, 0xceb3'96bc, 0xcea3'86ac},
                              {0xce26'ecb7, 0xce06'cc97, 0xce66'acf7, 0xce46'8cd7},
                              {0xcd19'b0da, 0xcbc9'82cc, 0xcdcc'58ec, 0xcd8c'18ac},
                              {0x4db4'2094, 0x4df4'60d4, 0x4cd2'8052, 0x4d69'c0aa},
                              {0x4e5a'90ca, 0x4e7a'b0eb, 0x4e1a'd08b, 0x4e3a'f0ab},
                              {0x4ead'88a6, 0x4ebd'98b6, 0x4e8d'a886, 0x4e9d'b896},
                              {0x4eed'c8e6, 0x4efd'd8f6, 0x4ecd'e8c6, 0x4edd'f8d6}},
                             {{0xc3d8'7cd9'7d9a'7edc, 0xc3dc'78dd'79de'7adf},
                              {0xc3d0'74d1'7592'76d3, 0xc3d4'70d5'71d6'72d7},
                              {0xc3c0'd992'db14'dd97, 0xc3c8'd19a'd39c'd59f},
                              {0xc379'3059'6099'b0da, 0xc3b1'8315'8719'8b1e},
                              {0x43be'8c1a'8916'8412, 0x43ad'3815'300d'2805},
                              {0x43cf'561d'549b'5219, 0x43c7'5e15'5c13'5a11},
                              {0x43d7'b316'b255'b115, 0x43d3'b712'b611'b511},
                              {0x43df'bb1e'ba5d'b91d, 0x43db'bf1a'be19'bd19}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfcvtrtzxuf,
                             ExecMaskedVfcvtrtzxuf,
                             {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0xffff'ffff, 0xffff'ffff, 0x0000'6a21, 0x6e25'6c00},
                              {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
                             {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
                              {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfcvtrtzxf,
                             ExecMaskedVfcvtrtzxf,
                             {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x8000'0000, 0x8000'0000, 0xffff'cad0, 0xc8cd'6a00},
                              {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                              {0x7fff'ffff, 0x7fff'ffff, 0x0000'6a21, 0x6e25'6c00},
                              {0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff}},
                             {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                              {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                              {0x7fff'ffff'ffff'ffff, 0x7fff'ffff'ffff'ffff},
                              {0x7fff'ffff'ffff'ffff, 0x7fff'ffff'ffff'ffff}},
                             kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtxufv,
                                     ExecMaskedVfwcvtxufv,
                                     {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'6229'6000'0000, 0x662d'6480'0000'0000},
                                      {0x0000'0000'0000'6a21, 0x0000'0000'6e25'6c00},
                                      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
                                      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtxfv,
                                     ExecMaskedVfwcvtxfv,
                                     {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0xffff'cecb'7000'0000, 0xccc9'6dc0'0000'0000},
                                      {0xffff'ffff'ffff'cacf, 0xffff'ffff'c8cd'6a00},
                                      {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                                      {0x8000'0000'0000'0000, 0x8000'0000'0000'0000}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtffv,
                                     ExecMaskedVfwcvtffv,
                                     {{0xbac0'9240'0000'0000, 0xbbc1'9341'2000'0000},
                                      {0xb8c2'9042'2000'0000, 0xb9c3'9143'0000'0000},
                                      {0xbec4'9644'0000'0000, 0xbfc5'9745'2000'0000},
                                      {0xbcc6'9446'2000'0000, 0xbdc7'9547'0000'0000},
                                      {0xc2c8'9a48'0000'0000, 0xc3c9'9b49'2000'0000},
                                      {0xc0ca'984a'2000'0000, 0xc1cb'994b'0000'0000},
                                      {0xc6cc'9e4c'0000'0000, 0xc7cd'9f4d'2000'0000},
                                      {0xc4ce'9c4e'2000'0000, 0xc5cf'9d4f'0000'0000}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtfxuv,
                                     ExecMaskedVfwcvtfxuv,
                                     {{0x4712'0000, 0x4716'0400, 0x471a'0900, 0x471e'0c00},
                                      {0x4702'1100, 0x4706'1400, 0x470a'1800, 0x470e'1c00},
                                      {0x4732'2000, 0x4736'2400, 0x473a'2900, 0x473e'2c00},
                                      {0x4722'3100, 0x4726'3400, 0x472a'3800, 0x472e'3c00},
                                      {0x4752'4000, 0x4756'4400, 0x475a'4900, 0x475e'4c00},
                                      {0x4742'5100, 0x4746'5400, 0x474a'5800, 0x474e'5c00},
                                      {0x4772'6000, 0x4776'6400, 0x477a'6900, 0x477e'6c00},
                                      {0x4762'7100, 0x4766'7400, 0x476a'7800, 0x476e'7c00}},
                                     {{0x41e2'c092'4000'0000, 0x41e3'c193'4120'0000},
                                      {0x41e0'c290'4220'0000, 0x41e1'c391'4300'0000},
                                      {0x41e6'c496'4400'0000, 0x41e7'c597'4520'0000},
                                      {0x41e4'c694'4620'0000, 0x41e5'c795'4700'0000},
                                      {0x41ea'c89a'4800'0000, 0x41eb'c99b'4920'0000},
                                      {0x41e8'ca98'4a20'0000, 0x41e9'cb99'4b00'0000},
                                      {0x41ee'cc9e'4c00'0000, 0x41ef'cd9f'4d20'0000},
                                      {0x41ec'ce9c'4e20'0000, 0x41ed'cf9d'4f00'0000}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtfxv,
                                     ExecMaskedVfwcvtfxv,
                                     {{0xc6dc'0000, 0xc6d3'f800, 0xc6cb'ee00, 0xc6c3'e800},
                                      {0xc6fb'de00, 0xc6f3'd800, 0xc6eb'd000, 0xc6e3'c800},
                                      {0xc69b'c000, 0xc693'b800, 0xc68b'ae00, 0xc683'a800},
                                      {0xc6bb'9e00, 0xc6b3'9800, 0xc6ab'9000, 0xc6a3'8800},
                                      {0xc637'0000, 0xc626'f000, 0xc616'dc00, 0xc606'd000},
                                      {0xc676'bc00, 0xc666'b000, 0xc656'a000, 0xc646'9000},
                                      {0xc55a'0000, 0xc519'c000, 0xc4b2'e000, 0xc3ca'0000},
                                      {0xc5ec'7800, 0xc5cc'6000, 0xc5ac'4000, 0xc58c'2000}},
                                     {{0xc1da'7edb'8000'0000, 0xc1d8'7cd9'7dc0'0000},
                                      {0xc1de'7adf'7bc0'0000, 0xc1dc'78dd'7a00'0000},
                                      {0xc1d2'76d3'7800'0000, 0xc1d0'74d1'75c0'0000},
                                      {0xc1d6'72d7'73c0'0000, 0xc1d4'70d5'7200'0000},
                                      {0xc1c4'dd96'e000'0000, 0xc1c0'd992'db80'0000},
                                      {0xc1cc'd59e'd780'0000, 0xc1c8'd19a'd400'0000},
                                      {0xc1a3'361b'4000'0000, 0xc179'3059'7000'0000},
                                      {0xc1b9'8b1d'8f00'0000, 0xc1b1'8315'8800'0000}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtrtzxuf,
                                     ExecMaskedVfwcvtrtzxuf,
                                     {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'6229'6000'0000, 0x662d'6480'0000'0000},
                                      {0x0000'0000'0000'6a21, 0x0000'0000'6e25'6c00},
                                      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
                                      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwcvtrtzxf,
                                     ExecMaskedVfwcvtrtzxf,
                                     {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                                      {0xffff'cecb'7000'0000, 0xccc9'6dc0'0000'0000},
                                      {0xffff'ffff'ffff'cad0, 0xffff'ffff'c8cd'6a00},
                                      {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                                      {0x8000'0000'0000'0000, 0x8000'0000'0000'0000}},
                                     kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(
      ExecVfncvtxufw,
      ExecMaskedVfncvtxufw,
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0xffff, 0xffff, 0x6a21, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(
      ExecVfncvtxfw,
      ExecMaskedVfncvtxfw,
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x8000, 0x8000, 0xcacf, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x7fff, 0x7fff, 0x6a21, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff}},
      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(ExecVfncvtffw,
                                      ExecMaskedVfncvtffw,
                                      {{0x8000'0000, 0x8000'0000, 0xb165'd14e, 0x8000'0000},
                                       {0xff80'0000, 0xff80'0000, 0xff80'0000, 0xff80'0000},
                                       {0x0000'0000, 0x0000'0000, 0x3561'd54a, 0x0000'0000},
                                       {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                                      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(ExecVfncvtfxuw,
                                      ExecMaskedVfncvtfxuw,
                                      {{0x5f1e'0c9a, 0x5f0e'1c8a, 0x5f3e'2cba, 0x5f2e'3caa},
                                       {0x5f5e'4cda, 0x5f4e'5cca, 0x5f7e'6cfa, 0x5f6e'7cea},
                                       {0x5df4'60d4, 0x5d69'c0aa, 0x5e7a'b0eb, 0x5e3a'f0ab},
                                       {0x5ebd'98b6, 0x5e9d'b896, 0x5efd'd8f6, 0x5edd'f8d6}},
                                      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(ExecVfncvtfxw,
                                      ExecMaskedVfncvtfxw,
                                      {{0xdec3'e6cc, 0xdee3'c6ec, 0xde83'a68c, 0xdea3'86ac},
                                       {0xde06'cc97, 0xde46'8cd7, 0xdbc9'82cb, 0xdd8c'18ac},
                                       {0x5df4'60d4, 0x5d69'c0aa, 0x5e7a'b0eb, 0x5e3a'f0ab},
                                       {0x5ebd'98b6, 0x5e9d'b896, 0x5efd'd8f6, 0x5edd'f8d6}},
                                      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(
      ExecVfncvtrtzxuf,
      ExecMaskedVfncvtrtzxuf,
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0xffff, 0xffff, 0x6a21, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
      kVectorCalculationsSource);
  TestNarrowingVectorFloatInstruction(
      ExecVfncvtrtzxfw,
      ExecMaskedVfncvtrtzxfw,
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x8000, 0x8000, 0xcad0, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x7fff, 0x7fff, 0x6a21, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff, 0x7fff'ffff}},
      kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVaddvv() {
  asm("vadd.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaddvv() {
  asm("vadd.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVaddvx() {
  asm("vadd.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaddvx() {
  asm("vadd.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVaddvi() {
  asm("vadd.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaddvi() {
  asm("vadd.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsadduvv() {
  asm("vsaddu.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsadduvv() {
  asm("vsaddu.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsadduvx() {
  asm("vsaddu.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsadduvx() {
  asm("vsaddu.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsadduvi() {
  asm("vsaddu.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsadduvi() {
  asm("vsaddu.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsaddvv() {
  asm("vsadd.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsaddvv() {
  asm("vsadd.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsaddvx() {
  asm("vsadd.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsaddvx() {
  asm("vsadd.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsaddvi() {
  asm("vsadd.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsaddvi() {
  asm("vsadd.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfaddvv() {
  asm("vfadd.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfaddvv() {
  asm("vfadd.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfaddvf() {
  asm("vfadd.vf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfaddvf() {
  asm("vfadd.vf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwaddvv() {
  asm("vfwadd.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwaddvv() {
  asm("vfwadd.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwaddwv() {
  asm("vfwadd.wv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwaddwv() {
  asm("vfwadd.wv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwaddwf() {
  asm("vfwadd.wf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwaddwf() {
  asm("vfwadd.wf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVadd) {
  TestVectorInstruction(
      ExecVaddvv,
      ExecMaskedVaddvv,
      {{0, 131, 6, 137, 13, 143, 18, 149, 25, 155, 30, 161, 36, 167, 42, 173},
       {48, 179, 54, 185, 61, 191, 66, 197, 73, 203, 78, 209, 84, 215, 90, 221},
       {96, 227, 102, 233, 109, 239, 114, 245, 121, 251, 126, 1, 132, 7, 138, 13},
       {144, 19, 150, 25, 157, 31, 162, 37, 169, 43, 174, 49, 180, 55, 186, 61},
       {192, 67, 198, 73, 205, 79, 210, 85, 217, 91, 222, 97, 228, 103, 234, 109},
       {240, 115, 246, 121, 253, 127, 2, 133, 9, 139, 14, 145, 20, 151, 26, 157},
       {32, 163, 38, 169, 45, 175, 50, 181, 57, 187, 62, 193, 68, 199, 74, 205},
       {80, 211, 86, 217, 93, 223, 98, 229, 105, 235, 110, 241, 116, 247, 122, 253}},
      {{0x8300, 0x8906, 0x8f0d, 0x9512, 0x9b19, 0xa11e, 0xa724, 0xad2a},
       {0xb330, 0xb936, 0xbf3d, 0xc542, 0xcb49, 0xd14e, 0xd754, 0xdd5a},
       {0xe360, 0xe966, 0xef6d, 0xf572, 0xfb79, 0x017e, 0x0784, 0x0d8a},
       {0x1390, 0x1996, 0x1f9d, 0x25a2, 0x2ba9, 0x31ae, 0x37b4, 0x3dba},
       {0x43c0, 0x49c6, 0x4fcd, 0x55d2, 0x5bd9, 0x61de, 0x67e4, 0x6dea},
       {0x73f0, 0x79f6, 0x7ffd, 0x8602, 0x8c09, 0x920e, 0x9814, 0x9e1a},
       {0xa420, 0xaa26, 0xb02d, 0xb632, 0xbc39, 0xc23e, 0xc844, 0xce4a},
       {0xd450, 0xda56, 0xe05d, 0xe662, 0xec69, 0xf26e, 0xf874, 0xfe7a}},
      {{0x8906'8300, 0x9512'8f0d, 0xa11e'9b19, 0xad2a'a724},
       {0xb936'b330, 0xc542'bf3d, 0xd14e'cb49, 0xdd5a'd754},
       {0xe966'e360, 0xf572'ef6d, 0x017e'fb79, 0x0d8b'0784},
       {0x1997'1390, 0x25a3'1f9d, 0x31af'2ba9, 0x3dbb'37b4},
       {0x49c7'43c0, 0x55d3'4fcd, 0x61df'5bd9, 0x6deb'67e4},
       {0x79f7'73f0, 0x8603'7ffd, 0x920f'8c09, 0x9e1b'9814},
       {0xaa27'a420, 0xb633'b02d, 0xc23f'bc39, 0xce4b'c844},
       {0xda57'd450, 0xe663'e05d, 0xf26f'ec69, 0xfe7b'f874}},
      {{0x9512'8f0d'8906'8300, 0xad2a'a724'a11e'9b19},
       {0xc542'bf3d'b936'b330, 0xdd5a'd754'd14e'cb49},
       {0xf572'ef6d'e966'e360, 0x0d8b'0785'017e'fb79},
       {0x25a3'1f9e'1997'1390, 0x3dbb'37b5'31af'2ba9},
       {0x55d3'4fce'49c7'43c0, 0x6deb'67e5'61df'5bd9},
       {0x8603'7ffe'79f7'73f0, 0x9e1b'9815'920f'8c09},
       {0xb633'b02e'aa27'a420, 0xce4b'c845'c23f'bc39},
       {0xe663'e05e'da57'd450, 0xfe7b'f875'f26f'ec69}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVaddvx,
      ExecMaskedVaddvx,
      {{170, 43, 172, 45, 174, 47, 176, 49, 178, 51, 180, 53, 182, 55, 184, 57},
       {186, 59, 188, 61, 190, 63, 192, 65, 194, 67, 196, 69, 198, 71, 200, 73},
       {202, 75, 204, 77, 206, 79, 208, 81, 210, 83, 212, 85, 214, 87, 216, 89},
       {218, 91, 220, 93, 222, 95, 224, 97, 226, 99, 228, 101, 230, 103, 232, 105},
       {234, 107, 236, 109, 238, 111, 240, 113, 242, 115, 244, 117, 246, 119, 248, 121},
       {250, 123, 252, 125, 254, 127, 0, 129, 2, 131, 4, 133, 6, 135, 8, 137},
       {10, 139, 12, 141, 14, 143, 16, 145, 18, 147, 20, 149, 22, 151, 24, 153},
       {26, 155, 28, 157, 30, 159, 32, 161, 34, 163, 36, 165, 38, 167, 40, 169}},
      {{0x2baa, 0x2dac, 0x2fae, 0x31b0, 0x33b2, 0x35b4, 0x37b6, 0x39b8},
       {0x3bba, 0x3dbc, 0x3fbe, 0x41c0, 0x43c2, 0x45c4, 0x47c6, 0x49c8},
       {0x4bca, 0x4dcc, 0x4fce, 0x51d0, 0x53d2, 0x55d4, 0x57d6, 0x59d8},
       {0x5bda, 0x5ddc, 0x5fde, 0x61e0, 0x63e2, 0x65e4, 0x67e6, 0x69e8},
       {0x6bea, 0x6dec, 0x6fee, 0x71f0, 0x73f2, 0x75f4, 0x77f6, 0x79f8},
       {0x7bfa, 0x7dfc, 0x7ffe, 0x8200, 0x8402, 0x8604, 0x8806, 0x8a08},
       {0x8c0a, 0x8e0c, 0x900e, 0x9210, 0x9412, 0x9614, 0x9816, 0x9a18},
       {0x9c1a, 0x9e1c, 0xa01e, 0xa220, 0xa422, 0xa624, 0xa826, 0xaa28}},
      {{0x2dad'2baa, 0x31b1'2fae, 0x35b5'33b2, 0x39b9'37b6},
       {0x3dbd'3bba, 0x41c1'3fbe, 0x45c5'43c2, 0x49c9'47c6},
       {0x4dcd'4bca, 0x51d1'4fce, 0x55d5'53d2, 0x59d9'57d6},
       {0x5ddd'5bda, 0x61e1'5fde, 0x65e5'63e2, 0x69e9'67e6},
       {0x6ded'6bea, 0x71f1'6fee, 0x75f5'73f2, 0x79f9'77f6},
       {0x7dfd'7bfa, 0x8201'7ffe, 0x8605'8402, 0x8a09'8806},
       {0x8e0d'8c0a, 0x9211'900e, 0x9615'9412, 0x9a19'9816},
       {0x9e1d'9c1a, 0xa221'a01e, 0xa625'a422, 0xaa29'a826}},
      {{0x31b1'2faf'2dad'2baa, 0x39b9'37b7'35b5'33b2},
       {0x41c1'3fbf'3dbd'3bba, 0x49c9'47c7'45c5'43c2},
       {0x51d1'4fcf'4dcd'4bca, 0x59d9'57d7'55d5'53d2},
       {0x61e1'5fdf'5ddd'5bda, 0x69e9'67e7'65e5'63e2},
       {0x71f1'6fef'6ded'6bea, 0x79f9'77f7'75f5'73f2},
       {0x8201'7fff'7dfd'7bfa, 0x8a09'8807'8605'8402},
       {0x9211'900f'8e0d'8c0a, 0x9a19'9817'9615'9412},
       {0xa221'a01f'9e1d'9c1a, 0xaa29'a827'a625'a422}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVaddvi,
      ExecMaskedVaddvi,
      {{245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255, 128, 1, 130, 3, 132},
       {5, 134, 7, 136, 9, 138, 11, 140, 13, 142, 15, 144, 17, 146, 19, 148},
       {21, 150, 23, 152, 25, 154, 27, 156, 29, 158, 31, 160, 33, 162, 35, 164},
       {37, 166, 39, 168, 41, 170, 43, 172, 45, 174, 47, 176, 49, 178, 51, 180},
       {53, 182, 55, 184, 57, 186, 59, 188, 61, 190, 63, 192, 65, 194, 67, 196},
       {69, 198, 71, 200, 73, 202, 75, 204, 77, 206, 79, 208, 81, 210, 83, 212},
       {85, 214, 87, 216, 89, 218, 91, 220, 93, 222, 95, 224, 97, 226, 99, 228},
       {101, 230, 103, 232, 105, 234, 107, 236, 109, 238, 111, 240, 113, 242, 115, 244}},
      {{0x80f5, 0x82f7, 0x84f9, 0x86fb, 0x88fd, 0x8aff, 0x8d01, 0x8f03},
       {0x9105, 0x9307, 0x9509, 0x970b, 0x990d, 0x9b0f, 0x9d11, 0x9f13},
       {0xa115, 0xa317, 0xa519, 0xa71b, 0xa91d, 0xab1f, 0xad21, 0xaf23},
       {0xb125, 0xb327, 0xb529, 0xb72b, 0xb92d, 0xbb2f, 0xbd31, 0xbf33},
       {0xc135, 0xc337, 0xc539, 0xc73b, 0xc93d, 0xcb3f, 0xcd41, 0xcf43},
       {0xd145, 0xd347, 0xd549, 0xd74b, 0xd94d, 0xdb4f, 0xdd51, 0xdf53},
       {0xe155, 0xe357, 0xe559, 0xe75b, 0xe95d, 0xeb5f, 0xed61, 0xef63},
       {0xf165, 0xf367, 0xf569, 0xf76b, 0xf96d, 0xfb6f, 0xfd71, 0xff73}},
      {{0x8302'80f5, 0x8706'84f9, 0x8b0a'88fd, 0x8f0e'8d01},
       {0x9312'9105, 0x9716'9509, 0x9b1a'990d, 0x9f1e'9d11},
       {0xa322'a115, 0xa726'a519, 0xab2a'a91d, 0xaf2e'ad21},
       {0xb332'b125, 0xb736'b529, 0xbb3a'b92d, 0xbf3e'bd31},
       {0xc342'c135, 0xc746'c539, 0xcb4a'c93d, 0xcf4e'cd41},
       {0xd352'd145, 0xd756'd549, 0xdb5a'd94d, 0xdf5e'dd51},
       {0xe362'e155, 0xe766'e559, 0xeb6a'e95d, 0xef6e'ed61},
       {0xf372'f165, 0xf776'f569, 0xfb7a'f96d, 0xff7e'fd71}},
      {{0x8706'8504'8302'80f5, 0x8f0e'8d0c'8b0a'88fd},
       {0x9716'9514'9312'9105, 0x9f1e'9d1c'9b1a'990d},
       {0xa726'a524'a322'a115, 0xaf2e'ad2c'ab2a'a91d},
       {0xb736'b534'b332'b125, 0xbf3e'bd3c'bb3a'b92d},
       {0xc746'c544'c342'c135, 0xcf4e'cd4c'cb4a'c93d},
       {0xd756'd554'd352'd145, 0xdf5e'dd5c'db5a'd94d},
       {0xe766'e564'e362'e155, 0xef6e'ed6c'eb6a'e95d},
       {0xf776'f574'f372'f165, 0xff7e'fd7c'fb7a'f96d}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVsadduvv,
      ExecMaskedVsadduvv,
      {{0, 255, 6, 255, 13, 255, 18, 255, 25, 255, 30, 255, 36, 255, 42, 255},
       {48, 255, 54, 255, 61, 255, 66, 255, 73, 255, 78, 255, 84, 255, 90, 255},
       {96, 255, 102, 255, 109, 255, 114, 255, 121, 255, 126, 255, 132, 255, 138, 255},
       {144, 255, 150, 255, 157, 255, 162, 255, 169, 255, 174, 255, 180, 255, 186, 255},
       {192, 211, 198, 217, 205, 223, 210, 229, 217, 203, 222, 209, 228, 215, 234, 221},
       {240, 255, 246, 255, 253, 255, 255, 255, 255, 251, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xd3c0, 0xd9c6, 0xdfcd, 0xe5d2, 0xcbd9, 0xd1de, 0xd7e4, 0xddea},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xfc09, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xd9c6'd3c0, 0xe5d2'dfcd, 0xd1de'cbd9, 0xddea'd7e4},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
      {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xe5d2'dfcd'd9c6'd3c0, 0xddea'd7e4'd1de'cbd9},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVsadduvx,
      ExecMaskedVsadduvx,
      {{170, 255, 172, 255, 174, 255, 176, 255, 178, 255, 180, 255, 182, 255, 184, 255},
       {186, 255, 188, 255, 190, 255, 192, 255, 194, 255, 196, 255, 198, 255, 200, 255},
       {202, 255, 204, 255, 206, 255, 208, 255, 210, 255, 212, 255, 214, 255, 216, 255},
       {218, 255, 220, 255, 222, 255, 224, 255, 226, 255, 228, 255, 230, 255, 232, 255},
       {234, 255, 236, 255, 238, 255, 240, 255, 242, 255, 244, 255, 246, 255, 248, 255},
       {250, 255, 252, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
      {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVsadduvi,
      ExecMaskedVsadduvi,
      {{245, 255, 247, 255, 249, 255, 251, 255, 253, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
      {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVsaddvv,
      ExecMaskedVsaddvv,
      {{0, 128, 6, 128, 13, 128, 18, 128, 25, 128, 30, 128, 36, 128, 42, 128},
       {48, 128, 54, 128, 61, 128, 66, 128, 73, 128, 78, 128, 84, 128, 90, 128},
       {96, 128, 102, 128, 109, 128, 114, 133, 121, 128, 126, 128, 127, 128, 127, 128},
       {127, 163, 127, 169, 127, 175, 127, 181, 127, 155, 127, 161, 127, 167, 127, 173},
       {192, 211, 198, 217, 205, 223, 210, 229, 217, 203, 222, 209, 228, 215, 234, 221},
       {240, 3, 246, 9, 253, 15, 2, 21, 9, 251, 14, 1, 20, 7, 26, 13},
       {32, 51, 38, 57, 45, 63, 50, 69, 57, 43, 62, 49, 68, 55, 74, 61},
       {80, 99, 86, 105, 93, 111, 98, 117, 105, 91, 110, 97, 116, 103, 122, 109}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8572, 0x8000, 0x8000, 0x8000, 0x8000},
       {0xa390, 0xa996, 0xaf9d, 0xb5a2, 0x9ba9, 0xa1ae, 0xa7b4, 0xadba},
       {0xd3c0, 0xd9c6, 0xdfcd, 0xe5d2, 0xcbd9, 0xd1de, 0xd7e4, 0xddea},
       {0x03f0, 0x09f6, 0x0ffd, 0x1602, 0xfc09, 0x020e, 0x0814, 0x0e1a},
       {0x3420, 0x3a26, 0x402d, 0x4632, 0x2c39, 0x323e, 0x3844, 0x3e4a},
       {0x6450, 0x6a56, 0x705d, 0x7662, 0x5c69, 0x626e, 0x6874, 0x6e7a}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8573'7f6d, 0x8000'0000, 0x8000'0000},
       {0xa997'a390, 0xb5a3'af9d, 0xa1af'9ba9, 0xadbb'a7b4},
       {0xd9c6'd3c0, 0xe5d2'dfcd, 0xd1de'cbd9, 0xddea'd7e4},
       {0x09f7'03f0, 0x1603'0ffd, 0x020e'fc09, 0x0e1b'0814},
       {0x3a27'3420, 0x4633'402d, 0x323f'2c39, 0x3e4b'3844},
       {0x6a57'6450, 0x7663'705d, 0x626f'5c69, 0x6e7b'6874}},
      {{0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8573'7f6e'7967'7360, 0x8000'0000'0000'0000},
       {0xb5a3'af9e'a997'a390, 0xadbb'a7b5'a1af'9ba9},
       {0xe5d2'dfcd'd9c6'd3c0, 0xddea'd7e4'd1de'cbd9},
       {0x1603'0ffe'09f7'03f0, 0x0e1b'0815'020e'fc09},
       {0x4633'402e'3a27'3420, 0x3e4b'3845'323f'2c39},
       {0x7663'705e'6a57'6450, 0x6e7b'6875'626f'5c69}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVsaddvx,
      ExecMaskedVsaddvx,
      {{170, 128, 172, 128, 174, 128, 176, 128, 178, 128, 180, 128, 182, 128, 184, 128},
       {186, 128, 188, 128, 190, 128, 192, 128, 194, 128, 196, 128, 198, 128, 200, 128},
       {202, 128, 204, 128, 206, 128, 208, 128, 210, 128, 212, 128, 214, 128, 216, 128},
       {218, 128, 220, 128, 222, 128, 224, 128, 226, 128, 228, 128, 230, 128, 232, 128},
       {234, 128, 236, 128, 238, 128, 240, 128, 242, 128, 244, 128, 246, 128, 248, 128},
       {250, 128, 252, 128, 254, 128, 0, 129, 2, 131, 4, 133, 6, 135, 8, 137},
       {10, 139, 12, 141, 14, 143, 16, 145, 18, 147, 20, 149, 22, 151, 24, 153},
       {26, 155, 28, 157, 30, 159, 32, 161, 34, 163, 36, 165, 38, 167, 40, 169}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8200, 0x8402, 0x8604, 0x8806, 0x8a08},
       {0x8c0a, 0x8e0c, 0x900e, 0x9210, 0x9412, 0x9614, 0x9816, 0x9a18},
       {0x9c1a, 0x9e1c, 0xa01e, 0xa220, 0xa422, 0xa624, 0xa826, 0xaa28}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8201'7ffe, 0x8605'8402, 0x8a09'8806},
       {0x8e0d'8c0a, 0x9211'900e, 0x9615'9412, 0x9a19'9816},
       {0x9e1d'9c1a, 0xa221'a01e, 0xa625'a422, 0xaa29'a826}},
      {{0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
       {0x8201'7fff'7dfd'7bfa, 0x8a09'8807'8605'8402},
       {0x9211'900f'8e0d'8c0a, 0x9a19'9817'9615'9412},
       {0xa221'a01f'9e1d'9c1a, 0xaa29'a827'a625'a422}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVsaddvi,
      ExecMaskedVsaddvi,
      {{245, 128, 247, 128, 249, 128, 251, 128, 253, 128, 255, 128, 1, 130, 3, 132},
       {5, 134, 7, 136, 9, 138, 11, 140, 13, 142, 15, 144, 17, 146, 19, 148},
       {21, 150, 23, 152, 25, 154, 27, 156, 29, 158, 31, 160, 33, 162, 35, 164},
       {37, 166, 39, 168, 41, 170, 43, 172, 45, 174, 47, 176, 49, 178, 51, 180},
       {53, 182, 55, 184, 57, 186, 59, 188, 61, 190, 63, 192, 65, 194, 67, 196},
       {69, 198, 71, 200, 73, 202, 75, 204, 77, 206, 79, 208, 81, 210, 83, 212},
       {85, 214, 87, 216, 89, 218, 91, 220, 93, 222, 95, 224, 97, 226, 99, 228},
       {101, 230, 103, 232, 105, 234, 107, 236, 109, 238, 111, 240, 113, 242, 115, 244}},
      {{0x80f5, 0x82f7, 0x84f9, 0x86fb, 0x88fd, 0x8aff, 0x8d01, 0x8f03},
       {0x9105, 0x9307, 0x9509, 0x970b, 0x990d, 0x9b0f, 0x9d11, 0x9f13},
       {0xa115, 0xa317, 0xa519, 0xa71b, 0xa91d, 0xab1f, 0xad21, 0xaf23},
       {0xb125, 0xb327, 0xb529, 0xb72b, 0xb92d, 0xbb2f, 0xbd31, 0xbf33},
       {0xc135, 0xc337, 0xc539, 0xc73b, 0xc93d, 0xcb3f, 0xcd41, 0xcf43},
       {0xd145, 0xd347, 0xd549, 0xd74b, 0xd94d, 0xdb4f, 0xdd51, 0xdf53},
       {0xe155, 0xe357, 0xe559, 0xe75b, 0xe95d, 0xeb5f, 0xed61, 0xef63},
       {0xf165, 0xf367, 0xf569, 0xf76b, 0xf96d, 0xfb6f, 0xfd71, 0xff73}},
      {{0x8302'80f5, 0x8706'84f9, 0x8b0a'88fd, 0x8f0e'8d01},
       {0x9312'9105, 0x9716'9509, 0x9b1a'990d, 0x9f1e'9d11},
       {0xa322'a115, 0xa726'a519, 0xab2a'a91d, 0xaf2e'ad21},
       {0xb332'b125, 0xb736'b529, 0xbb3a'b92d, 0xbf3e'bd31},
       {0xc342'c135, 0xc746'c539, 0xcb4a'c93d, 0xcf4e'cd41},
       {0xd352'd145, 0xd756'd549, 0xdb5a'd94d, 0xdf5e'dd51},
       {0xe362'e155, 0xe766'e559, 0xeb6a'e95d, 0xef6e'ed61},
       {0xf372'f165, 0xf776'f569, 0xfb7a'f96d, 0xff7e'fd71}},
      {{0x8706'8504'8302'80f5, 0x8f0e'8d0c'8b0a'88fd},
       {0x9716'9514'9312'9105, 0x9f1e'9d1c'9b1a'990d},
       {0xa726'a524'a322'a115, 0xaf2e'ad2c'ab2a'a91d},
       {0xb736'b534'b332'b125, 0xbf3e'bd3c'bb3a'b92d},
       {0xc746'c544'c342'c135, 0xcf4e'cd4c'cb4a'c93d},
       {0xd756'd554'd352'd145, 0xdf5e'dd5c'db5a'd94d},
       {0xe766'e564'e362'e155, 0xef6e'ed6c'eb6a'e95d},
       {0xf776'f574'f372'f165, 0xff7e'fd7c'fb7a'f96d}},
      kVectorCalculationsSource);

  TestVectorFloatInstruction(ExecVfaddvv,
                             ExecMaskedVfaddvv,
                             {{0x9604'9200, 0x9e0c'9a09, 0x8b0a'ae29, 0x8f35'af92},
                              {0xb624'b220, 0xbe2c'ba29, 0xa634'a233, 0xae3c'aa38},
                              {0xd644'd240, 0xde4c'da49, 0xc654'c251, 0xce5c'ca58},
                              {0xf664'f260, 0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78},
                              {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e4fe, 0xeb6a'e968, 0xef6e'ed6c},
                              {0x76e2'8cfd, 0x7eec'78fb, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x9e0c'9a09'9604'9200, 0x8f0e'8d45'9f3b'9531},
                              {0xbe2c'ba29'b624'b220, 0xae3c'aa38'a634'a231},
                              {0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251},
                              {0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0x7eec'7ae9'76e4'72e0, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfaddvf,
                             ExecMaskedVfaddvf,
                             {{0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b3'ffe9, 0x40b3'e8a9, 0x409c'2858},
                              {0xc33d'2140, 0xc746'bfa4, 0xcb4a'c942, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'7f85'0b0d'1315},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);

  TestWideningVectorFloatInstruction(ExecVfwaddvv,
                                     ExecMaskedVfwaddvv,
                                     {{0xbac0'9240'0000'4140, 0xbbc1'9341'2000'0043},
                                      {0xb961'55c5'1088'0000, 0xb9e6'b5f2'4000'0000},
                                      {0xbec4'9644'0000'0000, 0xbfc5'9745'2000'0000},
                                      {0xbcc6'9446'6d4c'8c00, 0xbdc7'9547'004f'4e8e},
                                      {0xc2c8'9a48'0000'0000, 0xc3c9'9b49'2000'0000},
                                      {0xc0ca'984a'2000'0000, 0xc1cb'994b'0000'0000},
                                      {0xc6cc'9e4c'0000'0000, 0xc7cd'9f4d'2000'0000},
                                      {0xc4ce'9c4e'2000'0000, 0xc5cf'9d4f'0000'0000}},
                                     kVectorCalculationsSource);

  TestWideningVectorFloatInstruction(ExecVfwaddwv,
                                     ExecMaskedVfwaddwv,
                                     {{0xbac0'9240'0000'0000, 0xbbc1'9341'2000'0000},
                                      {0xb8c2'9042'2000'0000, 0xb9c3'9143'0000'0000},
                                      {0xbec4'9644'0000'0000, 0xbfc5'9745'2000'0000},
                                      {0xbcc6'9446'2000'0000, 0xbf3e'bd3c'ea65'4738},
                                      {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                                      {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                                      {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                                      {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                                     kVectorCalculationsSource);

  TestWideningVectorFloatInstruction(ExecVfwaddwf,
                                     ExecMaskedVfwaddwf,
                                     {{0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                      {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                      {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                      {0x4016'8000'0000'0000, 0x4016'7f85'0b0d'1315},
                                      {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                                      {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                                      {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                                      {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                                     kVectorCalculationsSource);
}

}  // namespace
