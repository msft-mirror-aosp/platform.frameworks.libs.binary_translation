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

namespace VXRMFlags {

inline constexpr uint64_t RNU = 0b00;
inline constexpr uint64_t RNE = 0b01;
inline constexpr uint64_t RDN = 0b10;
inline constexpr uint64_t ROD = 0b11;

}  // namespace VXRMFlags

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

template <bool kIsMasked, typename ElementType>
auto MaskForElemIfMasked() {
  if constexpr (!kIsMasked) {
    return kNoMask;
  } else {
    return MaskForElem<ElementType>();
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

void TestNarrowingVectorInstruction(ExecInsnFunc exec_insn,
                                    ExecInsnFunc exec_masked_insn,
                                    const uint8_16_t (&expected_result_int8)[4],
                                    const uint16_8_t (&expected_result_int16)[4],
                                    const uint32_4_t (&expected_result_int32)[4],
                                    const SIMD128 (&source)[16]) {
  TestVectorInstruction<TestVectorInstructionKind::kInteger, TestVectorInstructionMode::kNarrowing>(
      exec_insn,
      exec_masked_insn,
      source,
      expected_result_int8,
      expected_result_int16,
      expected_result_int32);
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

template <bool kIsMasked, typename... ExpectedResultType, size_t... kResultsCount>
void TestVectorIota(ExecInsnFunc exec_insn,
                    const SIMD128 (&source)[16],
                    const ExpectedResultType (&... expected_result)[kResultsCount]) {
  auto Verify = [&source](ExecInsnFunc exec_insn,
                          uint8_t vsew,
                          const auto& expected_result,
                          auto elem_mask) {
    for (uint8_t vlmul = 0; vlmul < 8; ++vlmul) {
      for (uint8_t vta = 0; vta < 2; ++vta) {
        for (uint8_t vma = 0; vma < 2; ++vma) {
          uint64_t vtype = (vma << 7) | (vta << 6) | (vsew << 3) | vlmul;
          uint64_t vlmax = 0;
          asm("vsetvl %0, zero, %1" : "=r"(vlmax) : "r"(vtype));
          if (vlmax == 0) {
            continue;
          }

          for (uint8_t vl = 0; vl < vlmax; vl += vlmax) {
            // To make tests quick enough we don't test vl change with small register sets. Only
            // with vlmul == 2 (4 registers) we set vl to skip last register and half of next-to
            // last register.
            uint64_t vlin;
            if (vlmul == 2 && vl == vlmax) {
              vlin = 5 * vlmax / 8;
            } else {
              vlin = vl;
            }

            SIMD128 result[8];
            // Set expected_result vector registers into 0b01010101… pattern.
            // Set undisturbed result vector registers.
            std::fill_n(result, 8, kUndisturbedResult);

            RunCommonVectorFunc(exec_insn, &source[0], &result[0], 0, 0, vtype, vlin);

            SIMD128 expected_inactive[8];
            std::fill_n(expected_inactive, 8, (vma ? kAgnosticResult : kUndisturbedResult));

            // vl of 0 should never change dst registers
            if (vl == 0) {
              for (size_t index = 0; index < 8; ++index) {
                EXPECT_EQ(result[index], kUndisturbedResult);
              }
            } else if (vlmul < 4) {
              for (size_t index = 0; index < 1 << vlmul; ++index) {
                for (size_t index = 0; index < 1 << vlmul; ++index) {
                  if (index == 2 && vlmul == 2) {
                    EXPECT_EQ(
                        result[index],
                        ((SIMD128{expected_result[index]} & elem_mask[index] &
                          kFractionMaskInt8[3]) |
                         (expected_inactive[index] & ~elem_mask[index] & kFractionMaskInt8[3]) |
                         ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[3])));
                  } else if (index == 3 && vlmul == 2) {
                    EXPECT_EQ(result[index], vta ? kAgnosticResult : kUndisturbedResult);
                  } else {
                    EXPECT_EQ(result[index],
                              ((SIMD128{expected_result[index]} & elem_mask[index]) |
                               (expected_inactive[index] & ~elem_mask[index])));
                  }
                }
              }
            } else {
              // vlmul >= 4 only uses 1 register
              EXPECT_EQ(
                  result[0],
                  ((SIMD128{expected_result[0]} & elem_mask[0] & kFractionMaskInt8[vlmul - 4]) |
                   (expected_inactive[0] & ~elem_mask[0] & kFractionMaskInt8[vlmul - 4]) |
                   ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[vlmul - 4])));
            }
          }
        }
      }
    }
  };

  (Verify(exec_insn,
          BitUtilLog2(sizeof(std::tuple_element_t<0, ExpectedResultType>)),
          expected_result,
          MaskForElemIfMasked<kIsMasked, std::tuple_element_t<0, ExpectedResultType>>()),
   ...);
}

template <bool kIsMasked>
void TestVectorIota(ExecInsnFunc exec_insn,
                    const uint8_16_t (&expected_result_int8)[8],
                    const uint16_8_t (&expected_result_int16)[8],
                    const uint32_4_t (&expected_result_int32)[8],
                    const uint64_2_t (&expected_result_int64)[8],
                    const SIMD128 (&source)[16]) {
  TestVectorIota<kIsMasked>(exec_insn,
                            source,
                            expected_result_int8,
                            expected_result_int16,
                            expected_result_int32,
                            expected_result_int64);
}

// clang-format off
#define DEFINE_TWO_ARG_ONE_RES_FUNCTION(Name, Asm) \
  [[gnu::naked]] void Exec##Name() {               \
    asm(#Asm " v8, v16, v24\n\t"                   \
        "ret\n\t");                                \
  }                                                \
  [[gnu::naked]] void ExecMasked##Name() {         \
    asm(#Asm " v8, v16, v24, v0.t\n\t"             \
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
  asm("vfsqrt.v v8, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfsqrtv() {
  asm("vfsqrt.v v8, v24, v0.t\n\t"
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

[[gnu::naked]] void ExecVid() {
  asm("vid.v v8\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVid() {
  asm("vid.v v8, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVid) {
  TestVectorInstruction(
      ExecVid,
      ExecMaskedVid,
      {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
       {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
       {32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47},
       {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
       {64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79},
       {80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95},
       {96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111},
       {112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127}},
      {{0, 1, 2, 3, 4, 5, 6, 7},
       {8, 9, 10, 11, 12, 13, 14, 15},
       {16, 17, 18, 19, 20, 21, 22, 23},
       {24, 25, 26, 27, 28, 29, 30, 31},
       {32, 33, 34, 35, 36, 37, 38, 39},
       {40, 41, 42, 43, 44, 45, 46, 47},
       {48, 49, 50, 51, 52, 53, 54, 55},
       {56, 57, 58, 59, 60, 61, 62, 63}},
      {{0, 1, 2, 3},
       {4, 5, 6, 7},
       {8, 9, 10, 11},
       {12, 13, 14, 15},
       {16, 17, 18, 19},
       {20, 21, 22, 23},
       {24, 25, 26, 27},
       {28, 29, 30, 31}},
      {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {12, 13}, {14, 15}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecViotam() {
  asm("viota.m v8, v16\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedViotam() {
  asm("viota.m v8, v16, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestIota) {
  TestVectorIota<false>(ExecViotam,
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1},
                         {2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5},
                         {6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9},
                         {10, 10, 11, 12, 12, 12, 12, 12, 12, 13, 14, 15, 15, 15, 15, 15},
                         {16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19},
                         {20, 20, 21, 21, 22, 22, 22, 22, 22, 23, 24, 24, 25, 25, 25, 25},
                         {26, 26, 26, 27, 28, 28, 28, 28, 28, 29, 29, 30, 31, 31, 31, 31},
                         {32, 32, 33, 34, 35, 35, 35, 35, 35, 36, 37, 38, 39, 39, 39, 39}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0002, 0x0002, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003},
                         {0x0003, 0x0004, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005},
                         {0x0006, 0x0006, 0x0006, 0x0007, 0x0007, 0x0007, 0x0007, 0x0007},
                         {0x0007, 0x0008, 0x0008, 0x0009, 0x0009, 0x0009, 0x0009, 0x0009},
                         {0x000a, 0x000a, 0x000b, 0x000c, 0x000c, 0x000c, 0x000c, 0x000c},
                         {0x000c, 0x000d, 0x000e, 0x000f, 0x000f, 0x000f, 0x000f, 0x000f}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0002, 0x0000'0002, 0x0000'0003, 0x0000'0003},
                         {0x0000'0003, 0x0000'0003, 0x0000'0003, 0x0000'0003},
                         {0x0000'0003, 0x0000'0004, 0x0000'0005, 0x0000'0005},
                         {0x0000'0005, 0x0000'0005, 0x0000'0005, 0x0000'0005}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorCalculationsSource);
  TestVectorIota<true>(ExecMaskedViotam,
                       {{0, 0x55, 0, 0, 0x55, 0, 0x55, 0, 0, 0x55, 1, 0x55, 1, 1, 0x55, 1},
                        {2, 2, 0x55, 3, 0x55, 3, 3, 0x55, 3, 0x55, 4, 4, 0x55, 4, 0x55, 4},
                        {5, 0x55, 5, 0x55, 6, 6, 0x55, 6, 0x55, 6, 6, 0x55, 7, 0x55, 7, 7},
                        {8, 0x55, 8, 9, 0x55, 9, 0x55, 9, 9, 0x55, 10, 0x55, 11, 0x55, 11, 11},
                        {12, 0x55, 12, 0x55, 12, 12, 0x55, 12, 12, 13, 0x55, 13, 14, 14, 14, 0x55},
                        {14, 0x55, 14, 14, 0x55, 15, 15, 15, 0x55, 15, 16, 16, 17, 0x55, 17, 17},
                        {18, 18, 0x55, 18, 19, 19, 0x55, 19, 19, 20, 20, 0x55, 21, 0x55, 21, 0x55},
                        {21, 21, 22, 0x55, 23, 23, 23, 23, 0x55, 23, 0x55, 24, 0x55, 25, 25, 0x55}},
                       {{0x0000, 0x5555, 0x0000, 0x0000, 0x5555, 0x0000, 0x5555, 0x0000},
                        {0x0000, 0x5555, 0x0001, 0x5555, 0x0001, 0x0001, 0x5555, 0x0001},
                        {0x0002, 0x0002, 0x5555, 0x0003, 0x5555, 0x0003, 0x0003, 0x5555},
                        {0x0003, 0x5555, 0x0004, 0x0004, 0x5555, 0x0004, 0x5555, 0x0004},
                        {0x0005, 0x5555, 0x0005, 0x5555, 0x0006, 0x0006, 0x5555, 0x0006},
                        {0x5555, 0x0006, 0x0006, 0x5555, 0x0007, 0x5555, 0x0007, 0x0007},
                        {0x0008, 0x5555, 0x0008, 0x0009, 0x5555, 0x0009, 0x5555, 0x0009},
                        {0x0009, 0x5555, 0x000a, 0x5555, 0x000b, 0x5555, 0x000b, 0x000b}},
                       {{0x0000'0000, 0x5555'5555, 0x0000'0000, 0x0000'0000},
                        {0x5555'5555, 0x0000'0000, 0x5555'5555, 0x0000'0000},
                        {0x0000'0000, 0x5555'5555, 0x0000'0001, 0x5555'5555},
                        {0x0000'0001, 0x0000'0001, 0x5555'5555, 0x0000'0001},
                        {0x0000'0002, 0x0000'0002, 0x5555'5555, 0x0000'0003},
                        {0x5555'5555, 0x0000'0003, 0x0000'0003, 0x5555'5555},
                        {0x0000'0003, 0x5555'5555, 0x0000'0004, 0x0000'0004},
                        {0x5555'5555, 0x0000'0004, 0x5555'5555, 0x0000'0004}},
                       {{0x0000'0000'0000'0000, 0x5555'5555'5555'5555},
                        {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                        {0x5555'5555'5555'5555, 0x0000'0000'0000'0000},
                        {0x5555'5555'5555'5555, 0x0000'0000'0000'0000},
                        {0x0000'0000'0000'0000, 0x5555'5555'5555'5555},
                        {0x0000'0000'0000'0001, 0x5555'5555'5555'5555},
                        {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                        {0x5555'5555'5555'5555, 0x0000'0000'0000'0001}},
                       kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVrsubvx() {
  asm("vrsub.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVrsubvx() {
  asm("vrsub.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVrsubvi() {
  asm("vrsub.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVrsubvi() {
  asm("vrsub.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfrsubvf() {
  asm("vfrsub.vf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfrsubvf() {
  asm("vfrsub.vf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVrsub) {
  TestVectorInstruction(
      ExecVrsubvx,
      ExecMaskedVrsubvx,
      {{170, 41, 168, 39, 166, 37, 164, 35, 162, 33, 160, 31, 158, 29, 156, 27},
       {154, 25, 152, 23, 150, 21, 148, 19, 146, 17, 144, 15, 142, 13, 140, 11},
       {138, 9, 136, 7, 134, 5, 132, 3, 130, 1, 128, 255, 126, 253, 124, 251},
       {122, 249, 120, 247, 118, 245, 116, 243, 114, 241, 112, 239, 110, 237, 108, 235},
       {106, 233, 104, 231, 102, 229, 100, 227, 98, 225, 96, 223, 94, 221, 92, 219},
       {90, 217, 88, 215, 86, 213, 84, 211, 82, 209, 80, 207, 78, 205, 76, 203},
       {74, 201, 72, 199, 70, 197, 68, 195, 66, 193, 64, 191, 62, 189, 60, 187},
       {58, 185, 56, 183, 54, 181, 52, 179, 50, 177, 48, 175, 46, 173, 44, 171}},
      {{0x29aa, 0x27a8, 0x25a6, 0x23a4, 0x21a2, 0x1fa0, 0x1d9e, 0x1b9c},
       {0x199a, 0x1798, 0x1596, 0x1394, 0x1192, 0x0f90, 0x0d8e, 0x0b8c},
       {0x098a, 0x0788, 0x0586, 0x0384, 0x0182, 0xff80, 0xfd7e, 0xfb7c},
       {0xf97a, 0xf778, 0xf576, 0xf374, 0xf172, 0xef70, 0xed6e, 0xeb6c},
       {0xe96a, 0xe768, 0xe566, 0xe364, 0xe162, 0xdf60, 0xdd5e, 0xdb5c},
       {0xd95a, 0xd758, 0xd556, 0xd354, 0xd152, 0xcf50, 0xcd4e, 0xcb4c},
       {0xc94a, 0xc748, 0xc546, 0xc344, 0xc142, 0xbf40, 0xbd3e, 0xbb3c},
       {0xb93a, 0xb738, 0xb536, 0xb334, 0xb132, 0xaf30, 0xad2e, 0xab2c}},
      {{0x27a8'29aa, 0x23a4'25a6, 0x1fa0'21a2, 0x1b9c'1d9e},
       {0x1798'199a, 0x1394'1596, 0x0f90'1192, 0x0b8c'0d8e},
       {0x0788'098a, 0x0384'0586, 0xff80'0182, 0xfb7b'fd7e},
       {0xf777'f97a, 0xf373'f576, 0xef6f'f172, 0xeb6b'ed6e},
       {0xe767'e96a, 0xe363'e566, 0xdf5f'e162, 0xdb5b'dd5e},
       {0xd757'd95a, 0xd353'd556, 0xcf4f'd152, 0xcb4b'cd4e},
       {0xc747'c94a, 0xc343'c546, 0xbf3f'c142, 0xbb3b'bd3e},
       {0xb737'b93a, 0xb333'b536, 0xaf2f'b132, 0xab2b'ad2e}},
      {{0x23a4'25a6'27a8'29aa, 0x1b9c'1d9e'1fa0'21a2},
       {0x1394'1596'1798'199a, 0x0b8c'0d8e'0f90'1192},
       {0x0384'0586'0788'098a, 0xfb7b'fd7d'ff80'0182},
       {0xf373'f575'f777'f97a, 0xeb6b'ed6d'ef6f'f172},
       {0xe363'e565'e767'e96a, 0xdb5b'dd5d'df5f'e162},
       {0xd353'd555'd757'd95a, 0xcb4b'cd4d'cf4f'd152},
       {0xc343'c545'c747'c94a, 0xbb3b'bd3d'bf3f'c142},
       {0xb333'b535'b737'b93a, 0xab2b'ad2d'af2f'b132}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVrsubvi,
      ExecMaskedVrsubvi,
      {{245, 116, 243, 114, 241, 112, 239, 110, 237, 108, 235, 106, 233, 104, 231, 102},
       {229, 100, 227, 98, 225, 96, 223, 94, 221, 92, 219, 90, 217, 88, 215, 86},
       {213, 84, 211, 82, 209, 80, 207, 78, 205, 76, 203, 74, 201, 72, 199, 70},
       {197, 68, 195, 66, 193, 64, 191, 62, 189, 60, 187, 58, 185, 56, 183, 54},
       {181, 52, 179, 50, 177, 48, 175, 46, 173, 44, 171, 42, 169, 40, 167, 38},
       {165, 36, 163, 34, 161, 32, 159, 30, 157, 28, 155, 26, 153, 24, 151, 22},
       {149, 20, 147, 18, 145, 16, 143, 14, 141, 12, 139, 10, 137, 8, 135, 6},
       {133, 4, 131, 2, 129, 0, 127, 254, 125, 252, 123, 250, 121, 248, 119, 246}},
      {{0x7ef5, 0x7cf3, 0x7af1, 0x78ef, 0x76ed, 0x74eb, 0x72e9, 0x70e7},
       {0x6ee5, 0x6ce3, 0x6ae1, 0x68df, 0x66dd, 0x64db, 0x62d9, 0x60d7},
       {0x5ed5, 0x5cd3, 0x5ad1, 0x58cf, 0x56cd, 0x54cb, 0x52c9, 0x50c7},
       {0x4ec5, 0x4cc3, 0x4ac1, 0x48bf, 0x46bd, 0x44bb, 0x42b9, 0x40b7},
       {0x3eb5, 0x3cb3, 0x3ab1, 0x38af, 0x36ad, 0x34ab, 0x32a9, 0x30a7},
       {0x2ea5, 0x2ca3, 0x2aa1, 0x289f, 0x269d, 0x249b, 0x2299, 0x2097},
       {0x1e95, 0x1c93, 0x1a91, 0x188f, 0x168d, 0x148b, 0x1289, 0x1087},
       {0x0e85, 0x0c83, 0x0a81, 0x087f, 0x067d, 0x047b, 0x0279, 0x0077}},
      {{0x7cfd'7ef5, 0x78f9'7af1, 0x74f5'76ed, 0x70f1'72e9},
       {0x6ced'6ee5, 0x68e9'6ae1, 0x64e5'66dd, 0x60e1'62d9},
       {0x5cdd'5ed5, 0x58d9'5ad1, 0x54d5'56cd, 0x50d1'52c9},
       {0x4ccd'4ec5, 0x48c9'4ac1, 0x44c5'46bd, 0x40c1'42b9},
       {0x3cbd'3eb5, 0x38b9'3ab1, 0x34b5'36ad, 0x30b1'32a9},
       {0x2cad'2ea5, 0x28a9'2aa1, 0x24a5'269d, 0x20a1'2299},
       {0x1c9d'1e95, 0x1899'1a91, 0x1495'168d, 0x1091'1289},
       {0x0c8d'0e85, 0x0889'0a81, 0x0485'067d, 0x0081'0279}},
      {{0x78f9'7afb'7cfd'7ef5, 0x70f1'72f3'74f5'76ed},
       {0x68e9'6aeb'6ced'6ee5, 0x60e1'62e3'64e5'66dd},
       {0x58d9'5adb'5cdd'5ed5, 0x50d1'52d3'54d5'56cd},
       {0x48c9'4acb'4ccd'4ec5, 0x40c1'42c3'44c5'46bd},
       {0x38b9'3abb'3cbd'3eb5, 0x30b1'32b3'34b5'36ad},
       {0x28a9'2aab'2cad'2ea5, 0x20a1'22a3'24a5'269d},
       {0x1899'1a9b'1c9d'1e95, 0x1091'1293'1495'168d},
       {0x0889'0a8b'0c8d'0e85, 0x0081'0283'0485'067d}},
      kVectorCalculationsSourceLegacy);

  TestVectorFloatInstruction(ExecVfrsubvf,
                             ExecMaskedVfrsubvf,
                             {{0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0017, 0x40b4'1757, 0x40cb'd7a8},
                              {0x4348'6140, 0x4746'cae4, 0x4b4a'c94e, 0x4f4e'cd4c},
                              {0x5352'd150, 0x5756'd554, 0x5b5a'd958, 0x5f5e'dd5c},
                              {0x6362'e160, 0x6766'e564, 0x6b6a'e968, 0x6f6e'ed6c},
                              {0x7372'f170, 0x7776'f574, 0x7b7a'f978, 0x7f7e'fd7c}},
                             {{0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'807a'f4f2'eceb},
                              {0x4746'c544'c342'c140, 0x4f4e'cd4c'cb4a'c948},
                              {0x5756'd554'd352'd150, 0x5f5e'dd5c'db5a'd958},
                              {0x6766'e564'e362'e160, 0x6f6e'ed6c'eb6a'e968},
                              {0x7776'f574'f372'f170, 0x7f7e'fd7c'fb7a'f978}},
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

[[gnu::naked]] void ExecVsubvv() {
  asm("vsub.vv v8, v16, v24\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVsubvv() {
  asm("vsub.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVsubvx() {
  asm("vsub.vx v8, v16, t0\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVsubvx() {
  asm("vsub.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVssubuvv() {
  asm("vssubu.vv v8, v16, v24\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVssubuvv() {
  asm("vssubu.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVssubuvx() {
  asm("vssubu.vx v8, v16, t0\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVssubuvx() {
  asm("vssubu.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVssubvv() {
  asm("vssub.vv v8, v16, v24\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVssubvv() {
  asm("vssub.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVssubvx() {
  asm("vssub.vx v8, v16, t0\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecMaskedVssubvx() {
  asm("vssub.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfsubvv() {
  asm("vfsub.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfsubvv() {
  asm("vfsub.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfsubvf() {
  asm("vfsub.vf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfsubvf() {
  asm("vfsub.vf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwsubvv() {
  asm("vfwsub.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwsubvv() {
  asm("vfwsub.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwsubvf() {
  asm("vfwsub.vf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwsubvf() {
  asm("vfwsub.vf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwsubwv() {
  asm("vfwsub.wv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwsubwv() {
  asm("vfwsub.wv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVfwsubwf() {
  asm("vfwsub.wf v8, v16, ft0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVfwsubwf() {
  asm("vfwsub.wf v8, v16, ft0, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVsub) {
  TestVectorInstruction(
      ExecVsubvv,
      ExecMaskedVsubvv,
      {{0, 127, 254, 125, 251, 123, 250, 121, 247, 119, 246, 117, 244, 115, 242, 113},
       {240, 111, 238, 109, 235, 107, 234, 105, 231, 103, 230, 101, 228, 99, 226, 97},
       {224, 95, 222, 93, 219, 91, 218, 89, 215, 87, 214, 85, 212, 83, 210, 81},
       {208, 79, 206, 77, 203, 75, 202, 73, 199, 71, 198, 69, 196, 67, 194, 65},
       {192, 63, 190, 61, 187, 59, 186, 57, 183, 55, 182, 53, 180, 51, 178, 49},
       {176, 47, 174, 45, 171, 43, 170, 41, 167, 39, 166, 37, 164, 35, 162, 33},
       {160, 31, 158, 29, 155, 27, 154, 25, 151, 23, 150, 21, 148, 19, 146, 17},
       {144, 15, 142, 13, 139, 11, 138, 9, 135, 7, 134, 5, 132, 3, 130, 1}},
      {{0x7f00, 0x7cfe, 0x7afb, 0x78fa, 0x76f7, 0x74f6, 0x72f4, 0x70f2},
       {0x6ef0, 0x6cee, 0x6aeb, 0x68ea, 0x66e7, 0x64e6, 0x62e4, 0x60e2},
       {0x5ee0, 0x5cde, 0x5adb, 0x58da, 0x56d7, 0x54d6, 0x52d4, 0x50d2},
       {0x4ed0, 0x4cce, 0x4acb, 0x48ca, 0x46c7, 0x44c6, 0x42c4, 0x40c2},
       {0x3ec0, 0x3cbe, 0x3abb, 0x38ba, 0x36b7, 0x34b6, 0x32b4, 0x30b2},
       {0x2eb0, 0x2cae, 0x2aab, 0x28aa, 0x26a7, 0x24a6, 0x22a4, 0x20a2},
       {0x1ea0, 0x1c9e, 0x1a9b, 0x189a, 0x1697, 0x1496, 0x1294, 0x1092},
       {0x0e90, 0x0c8e, 0x0a8b, 0x088a, 0x0687, 0x0486, 0x0284, 0x0082}},
      {{0x7cfe'7f00, 0x78fa'7afb, 0x74f6'76f7, 0x70f2'72f4},
       {0x6cee'6ef0, 0x68ea'6aeb, 0x64e6'66e7, 0x60e2'62e4},
       {0x5cde'5ee0, 0x58da'5adb, 0x54d6'56d7, 0x50d2'52d4},
       {0x4cce'4ed0, 0x48ca'4acb, 0x44c6'46c7, 0x40c2'42c4},
       {0x3cbe'3ec0, 0x38ba'3abb, 0x34b6'36b7, 0x30b2'32b4},
       {0x2cae'2eb0, 0x28aa'2aab, 0x24a6'26a7, 0x20a2'22a4},
       {0x1c9e'1ea0, 0x189a'1a9b, 0x1496'1697, 0x1092'1294},
       {0x0c8e'0e90, 0x088a'0a8b, 0x0486'0687, 0x0082'0284}},
      {{0x78fa'7afb'7cfe'7f00, 0x70f2'72f4'74f6'76f7},
       {0x68ea'6aeb'6cee'6ef0, 0x60e2'62e4'64e6'66e7},
       {0x58da'5adb'5cde'5ee0, 0x50d2'52d4'54d6'56d7},
       {0x48ca'4acb'4cce'4ed0, 0x40c2'42c4'44c6'46c7},
       {0x38ba'3abb'3cbe'3ec0, 0x30b2'32b4'34b6'36b7},
       {0x28aa'2aab'2cae'2eb0, 0x20a2'22a4'24a6'26a7},
       {0x189a'1a9b'1c9e'1ea0, 0x1092'1294'1496'1697},
       {0x088a'0a8b'0c8e'0e90, 0x0082'0284'0486'0687}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVsubvx,
      ExecMaskedVsubvx,
      {{86, 215, 88, 217, 90, 219, 92, 221, 94, 223, 96, 225, 98, 227, 100, 229},
       {102, 231, 104, 233, 106, 235, 108, 237, 110, 239, 112, 241, 114, 243, 116, 245},
       {118, 247, 120, 249, 122, 251, 124, 253, 126, 255, 128, 1, 130, 3, 132, 5},
       {134, 7, 136, 9, 138, 11, 140, 13, 142, 15, 144, 17, 146, 19, 148, 21},
       {150, 23, 152, 25, 154, 27, 156, 29, 158, 31, 160, 33, 162, 35, 164, 37},
       {166, 39, 168, 41, 170, 43, 172, 45, 174, 47, 176, 49, 178, 51, 180, 53},
       {182, 55, 184, 57, 186, 59, 188, 61, 190, 63, 192, 65, 194, 67, 196, 69},
       {198, 71, 200, 73, 202, 75, 204, 77, 206, 79, 208, 81, 210, 83, 212, 85}},
      {{0xd656, 0xd858, 0xda5a, 0xdc5c, 0xde5e, 0xe060, 0xe262, 0xe464},
       {0xe666, 0xe868, 0xea6a, 0xec6c, 0xee6e, 0xf070, 0xf272, 0xf474},
       {0xf676, 0xf878, 0xfa7a, 0xfc7c, 0xfe7e, 0x0080, 0x0282, 0x0484},
       {0x0686, 0x0888, 0x0a8a, 0x0c8c, 0x0e8e, 0x1090, 0x1292, 0x1494},
       {0x1696, 0x1898, 0x1a9a, 0x1c9c, 0x1e9e, 0x20a0, 0x22a2, 0x24a4},
       {0x26a6, 0x28a8, 0x2aaa, 0x2cac, 0x2eae, 0x30b0, 0x32b2, 0x34b4},
       {0x36b6, 0x38b8, 0x3aba, 0x3cbc, 0x3ebe, 0x40c0, 0x42c2, 0x44c4},
       {0x46c6, 0x48c8, 0x4aca, 0x4ccc, 0x4ece, 0x50d0, 0x52d2, 0x54d4}},
      {{0xd857'd656, 0xdc5b'da5a, 0xe05f'de5e, 0xe463'e262},
       {0xe867'e666, 0xec6b'ea6a, 0xf06f'ee6e, 0xf473'f272},
       {0xf877'f676, 0xfc7b'fa7a, 0x007f'fe7e, 0x0484'0282},
       {0x0888'0686, 0x0c8c'0a8a, 0x1090'0e8e, 0x1494'1292},
       {0x1898'1696, 0x1c9c'1a9a, 0x20a0'1e9e, 0x24a4'22a2},
       {0x28a8'26a6, 0x2cac'2aaa, 0x30b0'2eae, 0x34b4'32b2},
       {0x38b8'36b6, 0x3cbc'3aba, 0x40c0'3ebe, 0x44c4'42c2},
       {0x48c8'46c6, 0x4ccc'4aca, 0x50d0'4ece, 0x54d4'52d2}},
      {{0xdc5b'da59'd857'd656, 0xe463'e261'e05f'de5e},
       {0xec6b'ea69'e867'e666, 0xf473'f271'f06f'ee6e},
       {0xfc7b'fa79'f877'f676, 0x0484'0282'007f'fe7e},
       {0x0c8c'0a8a'0888'0686, 0x1494'1292'1090'0e8e},
       {0x1c9c'1a9a'1898'1696, 0x24a4'22a2'20a0'1e9e},
       {0x2cac'2aaa'28a8'26a6, 0x34b4'32b2'30b0'2eae},
       {0x3cbc'3aba'38b8'36b6, 0x44c4'42c2'40c0'3ebe},
       {0x4ccc'4aca'48c8'46c6, 0x54d4'52d2'50d0'4ece}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(ExecVssubuvv,
                        ExecMaskedVssubuvv,
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 5, 0, 3, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 175, 0, 173, 0, 171, 0, 169, 0, 199, 0, 197, 0, 195, 0, 193},
                         {0, 159, 0, 157, 0, 155, 0, 153, 0, 183, 0, 181, 0, 179, 0, 177},
                         {0, 143, 0, 141, 0, 139, 0, 137, 0, 167, 0, 165, 0, 163, 0, 161},
                         {0, 127, 0, 125, 0, 123, 0, 121, 0, 151, 0, 149, 0, 147, 0, 145}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x06f7, 0x04f6, 0x02f4, 0x00f2},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0xaec0, 0xacbe, 0xaabb, 0xa8ba, 0xc6b7, 0xc4b6, 0xc2b4, 0xc0b2},
                         {0x9eb0, 0x9cae, 0x9aab, 0x98aa, 0xb6a7, 0xb4a6, 0xb2a4, 0xb0a2},
                         {0x8ea0, 0x8c9e, 0x8a9b, 0x889a, 0xa697, 0xa496, 0xa294, 0xa092},
                         {0x7e90, 0x7c8e, 0x7a8b, 0x788a, 0x9687, 0x9486, 0x9284, 0x9082}},
                        {{0x0000'0000, 0x0000'0000, 0x04f6'06f7, 0x00f2'02f4},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0xacbe'aec0, 0xa8ba'aabb, 0xc4b6'c6b7, 0xc0b2'c2b4},
                         {0x9cae'9eb0, 0x98aa'9aab, 0xb4a6'b6a7, 0xb0a2'b2a4},
                         {0x8c9e'8ea0, 0x889a'8a9b, 0xa496'a697, 0xa092'a294},
                         {0x7c8e'7e90, 0x788a'7a8b, 0x9486'9687, 0x9082'9284}},
                        {{0x0000'0000'0000'0000, 0x00f2'02f4'04f6'06f7},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0xa8ba'aabb'acbe'aec0, 0xc0b2'c2b4'c4b6'c6b7},
                         {0x98aa'9aab'9cae'9eb0, 0xb0a2'b2a4'b4a6'b6a7},
                         {0x889a'8a9b'8c9e'8ea0, 0xa092'a294'a496'a697},
                         {0x788a'7a8b'7c8e'7e90, 0x9082'9284'9486'9687}},
                        kVectorCalculationsSource);
  TestVectorInstruction(ExecVssubuvx,
                        ExecMaskedVssubuvx,
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 5},
                         {0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 17, 0, 19, 0, 21},
                         {0, 23, 0, 25, 0, 27, 0, 29, 0, 31, 0, 33, 0, 35, 0, 37},
                         {0, 39, 0, 41, 0, 43, 0, 45, 0, 47, 0, 49, 0, 51, 0, 53},
                         {0, 55, 0, 57, 0, 59, 0, 61, 0, 63, 0, 65, 0, 67, 0, 69},
                         {0, 71, 0, 73, 0, 75, 0, 77, 0, 79, 0, 81, 0, 83, 0, 85}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0080, 0x0282, 0x0484},
                         {0x0686, 0x0888, 0x0a8a, 0x0c8c, 0x0e8e, 0x1090, 0x1292, 0x1494},
                         {0x1696, 0x1898, 0x1a9a, 0x1c9c, 0x1e9e, 0x20a0, 0x22a2, 0x24a4},
                         {0x26a6, 0x28a8, 0x2aaa, 0x2cac, 0x2eae, 0x30b0, 0x32b2, 0x34b4},
                         {0x36b6, 0x38b8, 0x3aba, 0x3cbc, 0x3ebe, 0x40c0, 0x42c2, 0x44c4},
                         {0x46c6, 0x48c8, 0x4aca, 0x4ccc, 0x4ece, 0x50d0, 0x52d2, 0x54d4}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x007f'fe7e, 0x0484'0282},
                         {0x0888'0686, 0x0c8c'0a8a, 0x1090'0e8e, 0x1494'1292},
                         {0x1898'1696, 0x1c9c'1a9a, 0x20a0'1e9e, 0x24a4'22a2},
                         {0x28a8'26a6, 0x2cac'2aaa, 0x30b0'2eae, 0x34b4'32b2},
                         {0x38b8'36b6, 0x3cbc'3aba, 0x40c0'3ebe, 0x44c4'42c2},
                         {0x48c8'46c6, 0x4ccc'4aca, 0x50d0'4ece, 0x54d4'52d2}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0484'0282'007f'fe7e},
                         {0x0c8c'0a8a'0888'0686, 0x1494'1292'1090'0e8e},
                         {0x1c9c'1a9a'1898'1696, 0x24a4'22a2'20a0'1e9e},
                         {0x2cac'2aaa'28a8'26a6, 0x34b4'32b2'30b0'2eae},
                         {0x3cbc'3aba'38b8'36b6, 0x44c4'42c2'40c0'3ebe},
                         {0x4ccc'4aca'48c8'46c6, 0x54d4'52d2'50d0'4ece}},
                        kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVssubvv,
      ExecMaskedVssubvv,
      {{0, 239, 254, 237, 251, 235, 250, 233, 247, 7, 246, 5, 244, 3, 242, 1},
       {240, 223, 238, 221, 235, 219, 234, 217, 231, 247, 230, 245, 228, 243, 226, 241},
       {224, 207, 222, 205, 219, 203, 218, 201, 215, 231, 214, 229, 212, 227, 210, 225},
       {208, 191, 206, 189, 203, 187, 202, 185, 199, 215, 198, 213, 196, 211, 194, 209},
       {127, 175, 127, 173, 127, 171, 127, 169, 127, 199, 127, 197, 127, 195, 127, 193},
       {127, 159, 127, 157, 127, 155, 127, 153, 127, 183, 127, 181, 127, 179, 127, 177},
       {127, 143, 127, 141, 127, 139, 127, 137, 127, 167, 127, 165, 127, 163, 127, 161},
       {127, 128, 127, 128, 127, 128, 127, 128, 127, 151, 127, 149, 127, 147, 127, 145}},
      {{0xef00, 0xecfe, 0xeafb, 0xe8fa, 0x06f7, 0x04f6, 0x02f4, 0x00f2},
       {0xdef0, 0xdcee, 0xdaeb, 0xd8ea, 0xf6e7, 0xf4e6, 0xf2e4, 0xf0e2},
       {0xcee0, 0xccde, 0xcadb, 0xc8da, 0xe6d7, 0xe4d6, 0xe2d4, 0xe0d2},
       {0xbed0, 0xbcce, 0xbacb, 0xb8ca, 0xd6c7, 0xd4c6, 0xd2c4, 0xd0c2},
       {0xaec0, 0xacbe, 0xaabb, 0xa8ba, 0xc6b7, 0xc4b6, 0xc2b4, 0xc0b2},
       {0x9eb0, 0x9cae, 0x9aab, 0x98aa, 0xb6a7, 0xb4a6, 0xb2a4, 0xb0a2},
       {0x8ea0, 0x8c9e, 0x8a9b, 0x889a, 0xa697, 0xa496, 0xa294, 0xa092},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x9687, 0x9486, 0x9284, 0x9082}},
      {{0xecfd'ef00, 0xe8f9'eafb, 0x04f6'06f7, 0x00f2'02f4},
       {0xdced'def0, 0xd8e9'daeb, 0xf4e5'f6e7, 0xf0e1'f2e4},
       {0xccdd'cee0, 0xc8d9'cadb, 0xe4d5'e6d7, 0xe0d1'e2d4},
       {0xbccd'bed0, 0xb8c9'bacb, 0xd4c5'd6c7, 0xd0c1'd2c4},
       {0xacbe'aec0, 0xa8ba'aabb, 0xc4b6'c6b7, 0xc0b2'c2b4},
       {0x9cae'9eb0, 0x98aa'9aab, 0xb4a6'b6a7, 0xb0a2'b2a4},
       {0x8c9e'8ea0, 0x889a'8a9b, 0xa496'a697, 0xa092'a294},
       {0x8000'0000, 0x8000'0000, 0x9486'9687, 0x9082'9284}},
      {{0xe8f9'eafa'ecfd'ef00, 0x00f2'02f4'04f6'06f7},
       {0xd8e9'daea'dced'def0, 0xf0e1'f2e3'f4e5'f6e7},
       {0xc8d9'cada'ccdd'cee0, 0xe0d1'e2d3'e4d5'e6d7},
       {0xb8c9'baca'bccd'bed0, 0xd0c1'd2c3'd4c5'd6c7},
       {0xa8ba'aabb'acbe'aec0, 0xc0b2'c2b4'c4b6'c6b7},
       {0x98aa'9aab'9cae'9eb0, 0xb0a2'b2a4'b4a6'b6a7},
       {0x889a'8a9b'8c9e'8ea0, 0xa092'a294'a496'a697},
       {0x8000'0000'0000'0000, 0x9082'9284'9486'9687}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVssubvx,
      ExecMaskedVssubvx,
      {{86, 215, 88, 217, 90, 219, 92, 221, 94, 223, 96, 225, 98, 227, 100, 229},
       {102, 231, 104, 233, 106, 235, 108, 237, 110, 239, 112, 241, 114, 243, 116, 245},
       {118, 247, 120, 249, 122, 251, 124, 253, 126, 255, 127, 1, 127, 3, 127, 5},
       {127, 7, 127, 9, 127, 11, 127, 13, 127, 15, 127, 17, 127, 19, 127, 21},
       {127, 23, 127, 25, 127, 27, 127, 29, 127, 31, 127, 33, 127, 35, 127, 37},
       {127, 39, 127, 41, 127, 43, 127, 45, 127, 47, 127, 49, 127, 51, 127, 53},
       {127, 55, 127, 57, 127, 59, 127, 61, 127, 63, 127, 65, 127, 67, 127, 69},
       {127, 71, 127, 73, 127, 75, 127, 77, 127, 79, 127, 81, 127, 83, 127, 85}},
      {{0xd656, 0xd858, 0xda5a, 0xdc5c, 0xde5e, 0xe060, 0xe262, 0xe464},
       {0xe666, 0xe868, 0xea6a, 0xec6c, 0xee6e, 0xf070, 0xf272, 0xf474},
       {0xf676, 0xf878, 0xfa7a, 0xfc7c, 0xfe7e, 0x0080, 0x0282, 0x0484},
       {0x0686, 0x0888, 0x0a8a, 0x0c8c, 0x0e8e, 0x1090, 0x1292, 0x1494},
       {0x1696, 0x1898, 0x1a9a, 0x1c9c, 0x1e9e, 0x20a0, 0x22a2, 0x24a4},
       {0x26a6, 0x28a8, 0x2aaa, 0x2cac, 0x2eae, 0x30b0, 0x32b2, 0x34b4},
       {0x36b6, 0x38b8, 0x3aba, 0x3cbc, 0x3ebe, 0x40c0, 0x42c2, 0x44c4},
       {0x46c6, 0x48c8, 0x4aca, 0x4ccc, 0x4ece, 0x50d0, 0x52d2, 0x54d4}},
      {{0xd857'd656, 0xdc5b'da5a, 0xe05f'de5e, 0xe463'e262},
       {0xe867'e666, 0xec6b'ea6a, 0xf06f'ee6e, 0xf473'f272},
       {0xf877'f676, 0xfc7b'fa7a, 0x007f'fe7e, 0x0484'0282},
       {0x0888'0686, 0x0c8c'0a8a, 0x1090'0e8e, 0x1494'1292},
       {0x1898'1696, 0x1c9c'1a9a, 0x20a0'1e9e, 0x24a4'22a2},
       {0x28a8'26a6, 0x2cac'2aaa, 0x30b0'2eae, 0x34b4'32b2},
       {0x38b8'36b6, 0x3cbc'3aba, 0x40c0'3ebe, 0x44c4'42c2},
       {0x48c8'46c6, 0x4ccc'4aca, 0x50d0'4ece, 0x54d4'52d2}},
      {{0xdc5b'da59'd857'd656, 0xe463'e261'e05f'de5e},
       {0xec6b'ea69'e867'e666, 0xf473'f271'f06f'ee6e},
       {0xfc7b'fa79'f877'f676, 0x0484'0282'007f'fe7e},
       {0x0c8c'0a8a'0888'0686, 0x1494'1292'1090'0e8e},
       {0x1c9c'1a9a'1898'1696, 0x24a4'22a2'20a0'1e9e},
       {0x2cac'2aaa'28a8'26a6, 0x34b4'32b2'30b0'2eae},
       {0x3cbc'3aba'38b8'36b6, 0x44c4'42c2'40c0'3ebe},
       {0x4ccc'4aca'48c8'46c6, 0x54d4'52d2'50d0'4ece}},
      kVectorCalculationsSource);

  TestVectorFloatInstruction(ExecVfsubvv,
                             ExecMaskedVfsubvv,
                             {{0x1604'9200, 0x1e0c'9a09, 0x8b0a'63e7, 0x8ece'd50c},
                              {0x3624'b220, 0x3e2c'ba29, 0x2634'a22f, 0x2e3c'aa38},
                              {0x5644'd240, 0x5e4c'da49, 0x4654'c251, 0x4e5c'ca58},
                              {0x7664'f260, 0x7e6c'fa69, 0x6674'e271, 0x6e7c'ea78},
                              {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e5ca, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf6e6'58c3, 0xfeec'7cd7, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x1e0c'9a09'9604'9200, 0x8f0e'8cd3'76d9'7cdf},
                              {0x3e2c'ba29'b624'b220, 0x2e3c'aa38'a634'a231},
                              {0x5e4c'da49'd644'd240, 0x4e5c'ca58'c654'c251},
                              {0x7e6c'fa69'f664'f260, 0x6e7c'ea78'e674'e271},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xfeec'7ae9'76e4'72e0, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(ExecVfsubvf,
                             ExecMaskedVfsubvf,
                             {{0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000},
                              {0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000},
                              {0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000, 0xc0b4'0000},
                              {0xc0b4'0000, 0xc0b4'0017, 0xc0b4'1757, 0xc0cb'd7a8},
                              {0xc348'6140, 0xc746'cae4, 0xcb4a'c94e, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                              {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                              {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                              {0xc016'8000'0000'0000, 0xc016'807a'f4f2'eceb},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);

  TestWideningVectorFloatInstruction(ExecVfwsubvv,
                                     ExecMaskedVfwsubvv,
                                     {{0x3ac0'923f'ffff'bec0, 0x3bc1'9341'1fff'ffbd},
                                      {0xb961'4c7c'ef78'0000, 0xb9d9'daa1'8000'0000},
                                      {0x3ec4'9644'0000'0000, 0x3fc5'9745'2000'0000},
                                      {0x3cc6'9445'd2b3'7400, 0x3dc7'9546'ffb0'b172},
                                      {0x42c8'9a48'0000'0000, 0x43c9'9b49'2000'0000},
                                      {0x40ca'984a'2000'0000, 0x41cb'994b'0000'0000},
                                      {0x46cc'9e4c'0000'0000, 0x47cd'9f4d'2000'0000},
                                      {0x44ce'9c4e'2000'0000, 0x45cf'9d4f'0000'0000}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwsubvf,
                                     ExecMaskedVfwsubvf,
                                     {{0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0003},
                                      {0xc016'8000'0000'02ab, 0xc016'8000'0002'bab5},
                                      {0xc016'8000'02ca'c4c0, 0xc016'8002'dad4'd000},
                                      {0xc016'82ea'e4e0'0000, 0xc019'7af4'f000'0000}},
                                     kVectorCalculationsSource);

  TestWideningVectorFloatInstruction(ExecVfwsubwv,
                                     ExecMaskedVfwsubwv,
                                     {{0x3ac0'9240'0000'0000, 0x3bc1'9341'2000'0000},
                                      {0x38c2'9042'2000'0000, 0x39c3'9143'0000'0000},
                                      {0x3ec4'9644'0000'0000, 0x3fc5'9745'2000'0000},
                                      {0x3cc6'9446'2000'0000, 0xbf3e'bd3c'8c10'2b38},
                                      {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                                      {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                                      {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                                      {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(ExecVfwsubwf,
                                     ExecMaskedVfwsubwf,
                                     {{0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'8000'0000'0000},
                                      {0xc016'8000'0000'0000, 0xc016'807a'f4f2'eceb},
                                      {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                                      {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                                      {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                                      {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                                     kVectorCalculationsSource);
}

[[gnu::naked]] void ExecVandvv() {
  asm("vand.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVandvv() {
  asm("vand.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVandvx() {
  asm("vand.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVandvx() {
  asm("vand.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVandvi() {
  asm("vand.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVandvi() {
  asm("vand.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVand) {
  TestVectorInstruction(
      ExecVandvv,
      ExecMaskedVandvv,
      {{0, 0, 0, 2, 0, 0, 4, 6, 0, 0, 0, 2, 8, 8, 12, 14},
       {0, 0, 0, 2, 0, 0, 4, 6, 16, 16, 16, 18, 24, 24, 28, 30},
       {0, 0, 0, 2, 0, 0, 4, 6, 0, 0, 0, 2, 8, 8, 12, 14},
       {32, 32, 32, 34, 32, 32, 36, 38, 48, 48, 48, 50, 56, 56, 60, 62},
       {0, 128, 0, 130, 0, 128, 4, 134, 0, 128, 0, 130, 8, 136, 12, 142},
       {0, 128, 0, 130, 0, 128, 4, 134, 16, 144, 16, 146, 24, 152, 28, 158},
       {64, 192, 64, 194, 64, 192, 68, 198, 64, 192, 64, 194, 72, 200, 76, 206},
       {96, 224, 96, 226, 96, 224, 100, 230, 112, 240, 112, 242, 120, 248, 124, 254}},
      {{0x0000, 0x0200, 0x0000, 0x0604, 0x0000, 0x0200, 0x0808, 0x0e0c},
       {0x0000, 0x0200, 0x0000, 0x0604, 0x1010, 0x1210, 0x1818, 0x1e1c},
       {0x0000, 0x0200, 0x0000, 0x0604, 0x0000, 0x0200, 0x0808, 0x0e0c},
       {0x2020, 0x2220, 0x2020, 0x2624, 0x3030, 0x3230, 0x3838, 0x3e3c},
       {0x8000, 0x8200, 0x8000, 0x8604, 0x8000, 0x8200, 0x8808, 0x8e0c},
       {0x8000, 0x8200, 0x8000, 0x8604, 0x9010, 0x9210, 0x9818, 0x9e1c},
       {0xc040, 0xc240, 0xc040, 0xc644, 0xc040, 0xc240, 0xc848, 0xce4c},
       {0xe060, 0xe260, 0xe060, 0xe664, 0xf070, 0xf270, 0xf878, 0xfe7c}},
      {{0x0200'0000, 0x0604'0000, 0x0200'0000, 0x0e0c'0808},
       {0x0200'0000, 0x0604'0000, 0x1210'1010, 0x1e1c'1818},
       {0x0200'0000, 0x0604'0000, 0x0200'0000, 0x0e0c'0808},
       {0x2220'2020, 0x2624'2020, 0x3230'3030, 0x3e3c'3838},
       {0x8200'8000, 0x8604'8000, 0x8200'8000, 0x8e0c'8808},
       {0x8200'8000, 0x8604'8000, 0x9210'9010, 0x9e1c'9818},
       {0xc240'c040, 0xc644'c040, 0xc240'c040, 0xce4c'c848},
       {0xe260'e060, 0xe664'e060, 0xf270'f070, 0xfe7c'f878}},
      {{0x0604'0000'0200'0000, 0x0e0c'0808'0200'0000},
       {0x0604'0000'0200'0000, 0x1e1c'1818'1210'1010},
       {0x0604'0000'0200'0000, 0x0e0c'0808'0200'0000},
       {0x2624'2020'2220'2020, 0x3e3c'3838'3230'3030},
       {0x8604'8000'8200'8000, 0x8e0c'8808'8200'8000},
       {0x8604'8000'8200'8000, 0x9e1c'9818'9210'9010},
       {0xc644'c040'c240'c040, 0xce4c'c848'c240'c040},
       {0xe664'e060'e260'e060, 0xfe7c'f878'f270'f070}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(ExecVandvx,
                        ExecMaskedVandvx,
                        {{0, 128, 2, 130, 0, 128, 2, 130, 8, 136, 10, 138, 8, 136, 10, 138},
                         {0, 128, 2, 130, 0, 128, 2, 130, 8, 136, 10, 138, 8, 136, 10, 138},
                         {32, 160, 34, 162, 32, 160, 34, 162, 40, 168, 42, 170, 40, 168, 42, 170},
                         {32, 160, 34, 162, 32, 160, 34, 162, 40, 168, 42, 170, 40, 168, 42, 170},
                         {0, 128, 2, 130, 0, 128, 2, 130, 8, 136, 10, 138, 8, 136, 10, 138},
                         {0, 128, 2, 130, 0, 128, 2, 130, 8, 136, 10, 138, 8, 136, 10, 138},
                         {32, 160, 34, 162, 32, 160, 34, 162, 40, 168, 42, 170, 40, 168, 42, 170},
                         {32, 160, 34, 162, 32, 160, 34, 162, 40, 168, 42, 170, 40, 168, 42, 170}},
                        {{0x8000, 0x8202, 0x8000, 0x8202, 0x8808, 0x8a0a, 0x8808, 0x8a0a},
                         {0x8000, 0x8202, 0x8000, 0x8202, 0x8808, 0x8a0a, 0x8808, 0x8a0a},
                         {0xa020, 0xa222, 0xa020, 0xa222, 0xa828, 0xaa2a, 0xa828, 0xaa2a},
                         {0xa020, 0xa222, 0xa020, 0xa222, 0xa828, 0xaa2a, 0xa828, 0xaa2a},
                         {0x8000, 0x8202, 0x8000, 0x8202, 0x8808, 0x8a0a, 0x8808, 0x8a0a},
                         {0x8000, 0x8202, 0x8000, 0x8202, 0x8808, 0x8a0a, 0x8808, 0x8a0a},
                         {0xa020, 0xa222, 0xa020, 0xa222, 0xa828, 0xaa2a, 0xa828, 0xaa2a},
                         {0xa020, 0xa222, 0xa020, 0xa222, 0xa828, 0xaa2a, 0xa828, 0xaa2a}},
                        {{0x8202'8000, 0x8202'8000, 0x8a0a'8808, 0x8a0a'8808},
                         {0x8202'8000, 0x8202'8000, 0x8a0a'8808, 0x8a0a'8808},
                         {0xa222'a020, 0xa222'a020, 0xaa2a'a828, 0xaa2a'a828},
                         {0xa222'a020, 0xa222'a020, 0xaa2a'a828, 0xaa2a'a828},
                         {0x8202'8000, 0x8202'8000, 0x8a0a'8808, 0x8a0a'8808},
                         {0x8202'8000, 0x8202'8000, 0x8a0a'8808, 0x8a0a'8808},
                         {0xa222'a020, 0xa222'a020, 0xaa2a'a828, 0xaa2a'a828},
                         {0xa222'a020, 0xa222'a020, 0xaa2a'a828, 0xaa2a'a828}},
                        {{0x8202'8000'8202'8000, 0x8a0a'8808'8a0a'8808},
                         {0x8202'8000'8202'8000, 0x8a0a'8808'8a0a'8808},
                         {0xa222'a020'a222'a020, 0xaa2a'a828'aa2a'a828},
                         {0xa222'a020'a222'a020, 0xaa2a'a828'aa2a'a828},
                         {0x8202'8000'8202'8000, 0x8a0a'8808'8a0a'8808},
                         {0x8202'8000'8202'8000, 0x8a0a'8808'8a0a'8808},
                         {0xa222'a020'a222'a020, 0xaa2a'a828'aa2a'a828},
                         {0xa222'a020'a222'a020, 0xaa2a'a828'aa2a'a828}},
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVandvi,
      ExecMaskedVandvi,
      {{0, 129, 0, 129, 4, 133, 4, 133, 0, 129, 0, 129, 4, 133, 4, 133},
       {16, 145, 16, 145, 20, 149, 20, 149, 16, 145, 16, 145, 20, 149, 20, 149},
       {32, 161, 32, 161, 36, 165, 36, 165, 32, 161, 32, 161, 36, 165, 36, 165},
       {48, 177, 48, 177, 52, 181, 52, 181, 48, 177, 48, 177, 52, 181, 52, 181},
       {64, 193, 64, 193, 68, 197, 68, 197, 64, 193, 64, 193, 68, 197, 68, 197},
       {80, 209, 80, 209, 84, 213, 84, 213, 80, 209, 80, 209, 84, 213, 84, 213},
       {96, 225, 96, 225, 100, 229, 100, 229, 96, 225, 96, 225, 100, 229, 100, 229},
       {112, 241, 112, 241, 116, 245, 116, 245, 112, 241, 112, 241, 116, 245, 116, 245}},
      {{0x8100, 0x8300, 0x8504, 0x8704, 0x8900, 0x8b00, 0x8d04, 0x8f04},
       {0x9110, 0x9310, 0x9514, 0x9714, 0x9910, 0x9b10, 0x9d14, 0x9f14},
       {0xa120, 0xa320, 0xa524, 0xa724, 0xa920, 0xab20, 0xad24, 0xaf24},
       {0xb130, 0xb330, 0xb534, 0xb734, 0xb930, 0xbb30, 0xbd34, 0xbf34},
       {0xc140, 0xc340, 0xc544, 0xc744, 0xc940, 0xcb40, 0xcd44, 0xcf44},
       {0xd150, 0xd350, 0xd554, 0xd754, 0xd950, 0xdb50, 0xdd54, 0xdf54},
       {0xe160, 0xe360, 0xe564, 0xe764, 0xe960, 0xeb60, 0xed64, 0xef64},
       {0xf170, 0xf370, 0xf574, 0xf774, 0xf970, 0xfb70, 0xfd74, 0xff74}},
      {{0x8302'8100, 0x8706'8504, 0x8b0a'8900, 0x8f0e'8d04},
       {0x9312'9110, 0x9716'9514, 0x9b1a'9910, 0x9f1e'9d14},
       {0xa322'a120, 0xa726'a524, 0xab2a'a920, 0xaf2e'ad24},
       {0xb332'b130, 0xb736'b534, 0xbb3a'b930, 0xbf3e'bd34},
       {0xc342'c140, 0xc746'c544, 0xcb4a'c940, 0xcf4e'cd44},
       {0xd352'd150, 0xd756'd554, 0xdb5a'd950, 0xdf5e'dd54},
       {0xe362'e160, 0xe766'e564, 0xeb6a'e960, 0xef6e'ed64},
       {0xf372'f170, 0xf776'f574, 0xfb7a'f970, 0xff7e'fd74}},
      {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8900},
       {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9910},
       {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a920},
       {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b930},
       {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c940},
       {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd950},
       {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e960},
       {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f970}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVorvv() {
  asm("vor.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVorvv() {
  asm("vor.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVorvx() {
  asm("vor.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVorvx() {
  asm("vor.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVorvi() {
  asm("vor.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVorvi() {
  asm("vor.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVor) {
  TestVectorInstruction(
      ExecVorvv,
      ExecMaskedVorvv,
      {{0, 131, 6, 135, 13, 143, 14, 143, 25, 155, 30, 159, 28, 159, 30, 159},
       {48, 179, 54, 183, 61, 191, 62, 191, 57, 187, 62, 191, 60, 191, 62, 191},
       {96, 227, 102, 231, 109, 239, 110, 239, 121, 251, 126, 255, 124, 255, 126, 255},
       {112, 243, 118, 247, 125, 255, 126, 255, 121, 251, 126, 255, 124, 255, 126, 255},
       {192, 195, 198, 199, 205, 207, 206, 207, 217, 219, 222, 223, 220, 223, 222, 223},
       {240, 243, 246, 247, 253, 255, 254, 255, 249, 251, 254, 255, 252, 255, 254, 255},
       {224, 227, 230, 231, 237, 239, 238, 239, 249, 251, 254, 255, 252, 255, 254, 255},
       {240, 243, 246, 247, 253, 255, 254, 255, 249, 251, 254, 255, 252, 255, 254, 255}},
      {{0x8300, 0x8706, 0x8f0d, 0x8f0e, 0x9b19, 0x9f1e, 0x9f1c, 0x9f1e},
       {0xb330, 0xb736, 0xbf3d, 0xbf3e, 0xbb39, 0xbf3e, 0xbf3c, 0xbf3e},
       {0xe360, 0xe766, 0xef6d, 0xef6e, 0xfb79, 0xff7e, 0xff7c, 0xff7e},
       {0xf370, 0xf776, 0xff7d, 0xff7e, 0xfb79, 0xff7e, 0xff7c, 0xff7e},
       {0xc3c0, 0xc7c6, 0xcfcd, 0xcfce, 0xdbd9, 0xdfde, 0xdfdc, 0xdfde},
       {0xf3f0, 0xf7f6, 0xfffd, 0xfffe, 0xfbf9, 0xfffe, 0xfffc, 0xfffe},
       {0xe3e0, 0xe7e6, 0xefed, 0xefee, 0xfbf9, 0xfffe, 0xfffc, 0xfffe},
       {0xf3f0, 0xf7f6, 0xfffd, 0xfffe, 0xfbf9, 0xfffe, 0xfffc, 0xfffe}},
      {{0x8706'8300, 0x8f0e'8f0d, 0x9f1e'9b19, 0x9f1e'9f1c},
       {0xb736'b330, 0xbf3e'bf3d, 0xbf3e'bb39, 0xbf3e'bf3c},
       {0xe766'e360, 0xef6e'ef6d, 0xff7e'fb79, 0xff7e'ff7c},
       {0xf776'f370, 0xff7e'ff7d, 0xff7e'fb79, 0xff7e'ff7c},
       {0xc7c6'c3c0, 0xcfce'cfcd, 0xdfde'dbd9, 0xdfde'dfdc},
       {0xf7f6'f3f0, 0xfffe'fffd, 0xfffe'fbf9, 0xfffe'fffc},
       {0xe7e6'e3e0, 0xefee'efed, 0xfffe'fbf9, 0xfffe'fffc},
       {0xf7f6'f3f0, 0xfffe'fffd, 0xfffe'fbf9, 0xfffe'fffc}},
      {{0x8f0e'8f0d'8706'8300, 0x9f1e'9f1c'9f1e'9b19},
       {0xbf3e'bf3d'b736'b330, 0xbf3e'bf3c'bf3e'bb39},
       {0xef6e'ef6d'e766'e360, 0xff7e'ff7c'ff7e'fb79},
       {0xff7e'ff7d'f776'f370, 0xff7e'ff7c'ff7e'fb79},
       {0xcfce'cfcd'c7c6'c3c0, 0xdfde'dfdc'dfde'dbd9},
       {0xfffe'fffd'f7f6'f3f0, 0xfffe'fffc'fffe'fbf9},
       {0xefee'efed'e7e6'e3e0, 0xfffe'fffc'fffe'fbf9},
       {0xfffe'fffd'f7f6'f3f0, 0xfffe'fffc'fffe'fbf9}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVorvx,
      ExecMaskedVorvx,
      {{170, 171, 170, 171, 174, 175, 174, 175, 170, 171, 170, 171, 174, 175, 174, 175},
       {186, 187, 186, 187, 190, 191, 190, 191, 186, 187, 186, 187, 190, 191, 190, 191},
       {170, 171, 170, 171, 174, 175, 174, 175, 170, 171, 170, 171, 174, 175, 174, 175},
       {186, 187, 186, 187, 190, 191, 190, 191, 186, 187, 186, 187, 190, 191, 190, 191},
       {234, 235, 234, 235, 238, 239, 238, 239, 234, 235, 234, 235, 238, 239, 238, 239},
       {250, 251, 250, 251, 254, 255, 254, 255, 250, 251, 250, 251, 254, 255, 254, 255},
       {234, 235, 234, 235, 238, 239, 238, 239, 234, 235, 234, 235, 238, 239, 238, 239},
       {250, 251, 250, 251, 254, 255, 254, 255, 250, 251, 250, 251, 254, 255, 254, 255}},
      {{0xabaa, 0xabaa, 0xafae, 0xafae, 0xabaa, 0xabaa, 0xafae, 0xafae},
       {0xbbba, 0xbbba, 0xbfbe, 0xbfbe, 0xbbba, 0xbbba, 0xbfbe, 0xbfbe},
       {0xabaa, 0xabaa, 0xafae, 0xafae, 0xabaa, 0xabaa, 0xafae, 0xafae},
       {0xbbba, 0xbbba, 0xbfbe, 0xbfbe, 0xbbba, 0xbbba, 0xbfbe, 0xbfbe},
       {0xebea, 0xebea, 0xefee, 0xefee, 0xebea, 0xebea, 0xefee, 0xefee},
       {0xfbfa, 0xfbfa, 0xfffe, 0xfffe, 0xfbfa, 0xfbfa, 0xfffe, 0xfffe},
       {0xebea, 0xebea, 0xefee, 0xefee, 0xebea, 0xebea, 0xefee, 0xefee},
       {0xfbfa, 0xfbfa, 0xfffe, 0xfffe, 0xfbfa, 0xfbfa, 0xfffe, 0xfffe}},
      {{0xabaa'abaa, 0xafae'afae, 0xabaa'abaa, 0xafae'afae},
       {0xbbba'bbba, 0xbfbe'bfbe, 0xbbba'bbba, 0xbfbe'bfbe},
       {0xabaa'abaa, 0xafae'afae, 0xabaa'abaa, 0xafae'afae},
       {0xbbba'bbba, 0xbfbe'bfbe, 0xbbba'bbba, 0xbfbe'bfbe},
       {0xebea'ebea, 0xefee'efee, 0xebea'ebea, 0xefee'efee},
       {0xfbfa'fbfa, 0xfffe'fffe, 0xfbfa'fbfa, 0xfffe'fffe},
       {0xebea'ebea, 0xefee'efee, 0xebea'ebea, 0xefee'efee},
       {0xfbfa'fbfa, 0xfffe'fffe, 0xfbfa'fbfa, 0xfffe'fffe}},
      {{0xafae'afae'abaa'abaa, 0xafae'afae'abaa'abaa},
       {0xbfbe'bfbe'bbba'bbba, 0xbfbe'bfbe'bbba'bbba},
       {0xafae'afae'abaa'abaa, 0xafae'afae'abaa'abaa},
       {0xbfbe'bfbe'bbba'bbba, 0xbfbe'bfbe'bbba'bbba},
       {0xefee'efee'ebea'ebea, 0xefee'efee'ebea'ebea},
       {0xfffe'fffe'fbfa'fbfa, 0xfffe'fffe'fbfa'fbfa},
       {0xefee'efee'ebea'ebea, 0xefee'efee'ebea'ebea},
       {0xfffe'fffe'fbfa'fbfa, 0xfffe'fffe'fbfa'fbfa}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVorvi,
      ExecMaskedVorvi,
      {{245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255},
       {245, 245, 247, 247, 245, 245, 247, 247, 253, 253, 255, 255, 253, 253, 255, 255}},
      {{0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff},
       {0xfff5, 0xfff7, 0xfff5, 0xfff7, 0xfffd, 0xffff, 0xfffd, 0xffff}},
      {{0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fffd, 0xffff'fffd}},
      {{0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fffd}},
      kVectorCalculationsSourceLegacy);
}

[[gnu::naked]] void ExecVxorvv() {
  asm("vxor.vv v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVxorvv() {
  asm("vxor.vv v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVxorvx() {
  asm("vxor.vx v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVxorvx() {
  asm("vxor.vx v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}
[[gnu::naked]] void ExecVxorvi() {
  asm("vxor.vi v8, v16, -0xb\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVxorvi() {
  asm("vxor.vi v8, v16, -0xb, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestVxor) {
  TestVectorInstruction(
      ExecVxorvv,
      ExecMaskedVxorvv,
      {{0, 131, 6, 133, 13, 143, 10, 137, 25, 155, 30, 157, 20, 151, 18, 145},
       {48, 179, 54, 181, 61, 191, 58, 185, 41, 171, 46, 173, 36, 167, 34, 161},
       {96, 227, 102, 229, 109, 239, 106, 233, 121, 251, 126, 253, 116, 247, 114, 241},
       {80, 211, 86, 213, 93, 223, 90, 217, 73, 203, 78, 205, 68, 199, 66, 193},
       {192, 67, 198, 69, 205, 79, 202, 73, 217, 91, 222, 93, 212, 87, 210, 81},
       {240, 115, 246, 117, 253, 127, 250, 121, 233, 107, 238, 109, 228, 103, 226, 97},
       {160, 35, 166, 37, 173, 47, 170, 41, 185, 59, 190, 61, 180, 55, 178, 49},
       {144, 19, 150, 21, 157, 31, 154, 25, 137, 11, 142, 13, 132, 7, 130, 1}},
      {{0x8300, 0x8506, 0x8f0d, 0x890a, 0x9b19, 0x9d1e, 0x9714, 0x9112},
       {0xb330, 0xb536, 0xbf3d, 0xb93a, 0xab29, 0xad2e, 0xa724, 0xa122},
       {0xe360, 0xe566, 0xef6d, 0xe96a, 0xfb79, 0xfd7e, 0xf774, 0xf172},
       {0xd350, 0xd556, 0xdf5d, 0xd95a, 0xcb49, 0xcd4e, 0xc744, 0xc142},
       {0x43c0, 0x45c6, 0x4fcd, 0x49ca, 0x5bd9, 0x5dde, 0x57d4, 0x51d2},
       {0x73f0, 0x75f6, 0x7ffd, 0x79fa, 0x6be9, 0x6dee, 0x67e4, 0x61e2},
       {0x23a0, 0x25a6, 0x2fad, 0x29aa, 0x3bb9, 0x3dbe, 0x37b4, 0x31b2},
       {0x1390, 0x1596, 0x1f9d, 0x199a, 0x0b89, 0x0d8e, 0x0784, 0x0182}},
      {{0x8506'8300, 0x890a'8f0d, 0x9d1e'9b19, 0x9112'9714},
       {0xb536'b330, 0xb93a'bf3d, 0xad2e'ab29, 0xa122'a724},
       {0xe566'e360, 0xe96a'ef6d, 0xfd7e'fb79, 0xf172'f774},
       {0xd556'd350, 0xd95a'df5d, 0xcd4e'cb49, 0xc142'c744},
       {0x45c6'43c0, 0x49ca'4fcd, 0x5dde'5bd9, 0x51d2'57d4},
       {0x75f6'73f0, 0x79fa'7ffd, 0x6dee'6be9, 0x61e2'67e4},
       {0x25a6'23a0, 0x29aa'2fad, 0x3dbe'3bb9, 0x31b2'37b4},
       {0x1596'1390, 0x199a'1f9d, 0x0d8e'0b89, 0x0182'0784}},
      {{0x890a'8f0d'8506'8300, 0x9112'9714'9d1e'9b19},
       {0xb93a'bf3d'b536'b330, 0xa122'a724'ad2e'ab29},
       {0xe96a'ef6d'e566'e360, 0xf172'f774'fd7e'fb79},
       {0xd95a'df5d'd556'd350, 0xc142'c744'cd4e'cb49},
       {0x49ca'4fcd'45c6'43c0, 0x51d2'57d4'5dde'5bd9},
       {0x79fa'7ffd'75f6'73f0, 0x61e2'67e4'6dee'6be9},
       {0x29aa'2fad'25a6'23a0, 0x31b2'37b4'3dbe'3bb9},
       {0x199a'1f9d'1596'1390, 0x0182'0784'0d8e'0b89}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVxorvx,
      ExecMaskedVxorvx,
      {{170, 43, 168, 41, 174, 47, 172, 45, 162, 35, 160, 33, 166, 39, 164, 37},
       {186, 59, 184, 57, 190, 63, 188, 61, 178, 51, 176, 49, 182, 55, 180, 53},
       {138, 11, 136, 9, 142, 15, 140, 13, 130, 3, 128, 1, 134, 7, 132, 5},
       {154, 27, 152, 25, 158, 31, 156, 29, 146, 19, 144, 17, 150, 23, 148, 21},
       {234, 107, 232, 105, 238, 111, 236, 109, 226, 99, 224, 97, 230, 103, 228, 101},
       {250, 123, 248, 121, 254, 127, 252, 125, 242, 115, 240, 113, 246, 119, 244, 117},
       {202, 75, 200, 73, 206, 79, 204, 77, 194, 67, 192, 65, 198, 71, 196, 69},
       {218, 91, 216, 89, 222, 95, 220, 93, 210, 83, 208, 81, 214, 87, 212, 85}},
      {{0x2baa, 0x29a8, 0x2fae, 0x2dac, 0x23a2, 0x21a0, 0x27a6, 0x25a4},
       {0x3bba, 0x39b8, 0x3fbe, 0x3dbc, 0x33b2, 0x31b0, 0x37b6, 0x35b4},
       {0x0b8a, 0x0988, 0x0f8e, 0x0d8c, 0x0382, 0x0180, 0x0786, 0x0584},
       {0x1b9a, 0x1998, 0x1f9e, 0x1d9c, 0x1392, 0x1190, 0x1796, 0x1594},
       {0x6bea, 0x69e8, 0x6fee, 0x6dec, 0x63e2, 0x61e0, 0x67e6, 0x65e4},
       {0x7bfa, 0x79f8, 0x7ffe, 0x7dfc, 0x73f2, 0x71f0, 0x77f6, 0x75f4},
       {0x4bca, 0x49c8, 0x4fce, 0x4dcc, 0x43c2, 0x41c0, 0x47c6, 0x45c4},
       {0x5bda, 0x59d8, 0x5fde, 0x5ddc, 0x53d2, 0x51d0, 0x57d6, 0x55d4}},
      {{0x29a8'2baa, 0x2dac'2fae, 0x21a0'23a2, 0x25a4'27a6},
       {0x39b8'3bba, 0x3dbc'3fbe, 0x31b0'33b2, 0x35b4'37b6},
       {0x0988'0b8a, 0x0d8c'0f8e, 0x0180'0382, 0x0584'0786},
       {0x1998'1b9a, 0x1d9c'1f9e, 0x1190'1392, 0x1594'1796},
       {0x69e8'6bea, 0x6dec'6fee, 0x61e0'63e2, 0x65e4'67e6},
       {0x79f8'7bfa, 0x7dfc'7ffe, 0x71f0'73f2, 0x75f4'77f6},
       {0x49c8'4bca, 0x4dcc'4fce, 0x41c0'43c2, 0x45c4'47c6},
       {0x59d8'5bda, 0x5ddc'5fde, 0x51d0'53d2, 0x55d4'57d6}},
      {{0x2dac'2fae'29a8'2baa, 0x25a4'27a6'21a0'23a2},
       {0x3dbc'3fbe'39b8'3bba, 0x35b4'37b6'31b0'33b2},
       {0x0d8c'0f8e'0988'0b8a, 0x0584'0786'0180'0382},
       {0x1d9c'1f9e'1998'1b9a, 0x1594'1796'1190'1392},
       {0x6dec'6fee'69e8'6bea, 0x65e4'67e6'61e0'63e2},
       {0x7dfc'7ffe'79f8'7bfa, 0x75f4'77f6'71f0'73f2},
       {0x4dcc'4fce'49c8'4bca, 0x45c4'47c6'41c0'43c2},
       {0x5ddc'5fde'59d8'5bda, 0x55d4'57d6'51d0'53d2}},
      kVectorCalculationsSourceLegacy);
  TestVectorInstruction(
      ExecVxorvi,
      ExecMaskedVxorvi,
      {{245, 116, 247, 118, 241, 112, 243, 114, 253, 124, 255, 126, 249, 120, 251, 122},
       {229, 100, 231, 102, 225, 96, 227, 98, 237, 108, 239, 110, 233, 104, 235, 106},
       {213, 84, 215, 86, 209, 80, 211, 82, 221, 92, 223, 94, 217, 88, 219, 90},
       {197, 68, 199, 70, 193, 64, 195, 66, 205, 76, 207, 78, 201, 72, 203, 74},
       {181, 52, 183, 54, 177, 48, 179, 50, 189, 60, 191, 62, 185, 56, 187, 58},
       {165, 36, 167, 38, 161, 32, 163, 34, 173, 44, 175, 46, 169, 40, 171, 42},
       {149, 20, 151, 22, 145, 16, 147, 18, 157, 28, 159, 30, 153, 24, 155, 26},
       {133, 4, 135, 6, 129, 0, 131, 2, 141, 12, 143, 14, 137, 8, 139, 10}},
      {{0x7ef5, 0x7cf7, 0x7af1, 0x78f3, 0x76fd, 0x74ff, 0x72f9, 0x70fb},
       {0x6ee5, 0x6ce7, 0x6ae1, 0x68e3, 0x66ed, 0x64ef, 0x62e9, 0x60eb},
       {0x5ed5, 0x5cd7, 0x5ad1, 0x58d3, 0x56dd, 0x54df, 0x52d9, 0x50db},
       {0x4ec5, 0x4cc7, 0x4ac1, 0x48c3, 0x46cd, 0x44cf, 0x42c9, 0x40cb},
       {0x3eb5, 0x3cb7, 0x3ab1, 0x38b3, 0x36bd, 0x34bf, 0x32b9, 0x30bb},
       {0x2ea5, 0x2ca7, 0x2aa1, 0x28a3, 0x26ad, 0x24af, 0x22a9, 0x20ab},
       {0x1e95, 0x1c97, 0x1a91, 0x1893, 0x169d, 0x149f, 0x1299, 0x109b},
       {0x0e85, 0x0c87, 0x0a81, 0x0883, 0x068d, 0x048f, 0x0289, 0x008b}},
      {{0x7cfd'7ef5, 0x78f9'7af1, 0x74f5'76fd, 0x70f1'72f9},
       {0x6ced'6ee5, 0x68e9'6ae1, 0x64e5'66ed, 0x60e1'62e9},
       {0x5cdd'5ed5, 0x58d9'5ad1, 0x54d5'56dd, 0x50d1'52d9},
       {0x4ccd'4ec5, 0x48c9'4ac1, 0x44c5'46cd, 0x40c1'42c9},
       {0x3cbd'3eb5, 0x38b9'3ab1, 0x34b5'36bd, 0x30b1'32b9},
       {0x2cad'2ea5, 0x28a9'2aa1, 0x24a5'26ad, 0x20a1'22a9},
       {0x1c9d'1e95, 0x1899'1a91, 0x1495'169d, 0x1091'1299},
       {0x0c8d'0e85, 0x0889'0a81, 0x0485'068d, 0x0081'0289}},
      {{0x78f9'7afb'7cfd'7ef5, 0x70f1'72f3'74f5'76fd},
       {0x68e9'6aeb'6ced'6ee5, 0x60e1'62e3'64e5'66ed},
       {0x58d9'5adb'5cdd'5ed5, 0x50d1'52d3'54d5'56dd},
       {0x48c9'4acb'4ccd'4ec5, 0x40c1'42c3'44c5'46cd},
       {0x38b9'3abb'3cbd'3eb5, 0x30b1'32b3'34b5'36bd},
       {0x28a9'2aab'2cad'2ea5, 0x20a1'22a3'24a5'26ad},
       {0x1899'1a9b'1c9d'1e95, 0x1091'1293'1495'169d},
       {0x0889'0a8b'0c8d'0e85, 0x0081'0283'0485'068d}},
      kVectorCalculationsSourceLegacy);
}

}  // namespace

[[gnu::naked]] void ExecVaadduvv() {
  asm("vaaddu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaadduvv() {
  asm("vaaddu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVaadduvx() {
  asm("vaaddu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaadduvx() {
  asm("vaaddu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVaaddvv() {
  asm("vaadd.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaaddvv() {
  asm("vaadd.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVaaddvx() {
  asm("vaadd.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVaaddvx() {
  asm("vaadd.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVasubuvv() {
  asm("vasubu.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVasubuvv() {
  asm("vasubu.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVasubuvx() {
  asm("vasubu.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVasubuvx() {
  asm("vasubu.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVasubvv() {
  asm("vasub.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVasubvv() {
  asm("vasub.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVasubvx() {
  asm("vasub.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVasubvx() {
  asm("vasub.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipuwi() {
  asm("vnclipu.wi  v8, v16, 0xa\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipuwi() {
  asm("vnclipu.wi  v8, v16, 0xa, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipwi() {
  asm("vnclip.wi  v8, v16, 0xa\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipwi() {
  asm("vnclip.wi  v8, v16, 0xa, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipuwx() {
  asm("vnclipu.wx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipuwx() {
  asm("vnclipu.wx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipwx() {
  asm("vnclip.wx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipwx() {
  asm("vnclip.wx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipuwv() {
  asm("vnclipu.wv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipuwv() {
  asm("vnclipu.wv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVnclipwv() {
  asm("vnclip.wv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVnclipwv() {
  asm("vnclip.wv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsmulvv() {
  asm("vsmul.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsmulvv() {
  asm("vsmul.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVsmulvx() {
  asm("vsmul.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVsmulvx() {
  asm("vsmul.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssrlvv() {
  asm("vssrl.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssrlvv() {
  asm("vssrl.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssrlvx() {
  asm("vssrl.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssrlvx() {
  asm("vssrl.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssrlvi() {
  asm("vssrl.vi  v8, v16, 0xa\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssrlvi() {
  asm("vssrl.vi  v8, v16, 0xa, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssravv() {
  asm("vssra.vv  v8, v16, v24\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssravv() {
  asm("vssra.vv  v8, v16, v24, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssravx() {
  asm("vssra.vx  v8, v16, t0\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssravx() {
  asm("vssra.vx  v8, v16, t0, v0.t\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecVssravi() {
  asm("vssra.vi  v8, v16, 0xa\n\t"
      "ret\n\t");
}

[[gnu::naked]] void ExecMaskedVssravi() {
  asm("vssra.vi  v8, v16, 0xa, v0.t\n\t"
      "ret\n\t");
}

TEST(InlineAsmTestRiscv64, TestRDN) {
  uint64_t vxrm;
  asm("csrr %0, vxrm\n\t"
      "csrwi vxrm, %c1\n\t"
      : "=r"(vxrm)
      : "i"(VXRMFlags::RDN));
  TestVectorInstruction(
      ExecVaadduvv,
      ExecMaskedVaadduvv,
      {{0, 137, 3, 140, 6, 143, 9, 146, 12, 133, 15, 136, 18, 139, 21, 142},
       {24, 161, 27, 164, 30, 167, 33, 170, 36, 157, 39, 160, 42, 163, 45, 166},
       {48, 185, 51, 188, 54, 191, 57, 194, 60, 181, 63, 184, 66, 187, 69, 190},
       {72, 209, 75, 212, 78, 215, 81, 218, 84, 205, 87, 208, 90, 211, 93, 214},
       {96, 105, 99, 108, 102, 111, 105, 114, 108, 101, 111, 104, 114, 107, 117, 110},
       {120, 129, 123, 132, 126, 135, 129, 138, 132, 125, 135, 128, 138, 131, 141, 134},
       {144, 153, 147, 156, 150, 159, 153, 162, 156, 149, 159, 152, 162, 155, 165, 158},
       {168, 177, 171, 180, 174, 183, 177, 186, 180, 173, 183, 176, 186, 179, 189, 182}},
      {{0x8980, 0x8c83, 0x8f86, 0x9289, 0x858c, 0x888f, 0x8b92, 0x8e95},
       {0xa198, 0xa49b, 0xa79e, 0xaaa1, 0x9da4, 0xa0a7, 0xa3aa, 0xa6ad},
       {0xb9b0, 0xbcb3, 0xbfb6, 0xc2b9, 0xb5bc, 0xb8bf, 0xbbc2, 0xbec5},
       {0xd1c8, 0xd4cb, 0xd7ce, 0xdad1, 0xcdd4, 0xd0d7, 0xd3da, 0xd6dd},
       {0x69e0, 0x6ce3, 0x6fe6, 0x72e9, 0x65ec, 0x68ef, 0x6bf2, 0x6ef5},
       {0x81f8, 0x84fb, 0x87fe, 0x8b01, 0x7e04, 0x8107, 0x840a, 0x870d},
       {0x9a10, 0x9d13, 0xa016, 0xa319, 0x961c, 0x991f, 0x9c22, 0x9f25},
       {0xb228, 0xb52b, 0xb82e, 0xbb31, 0xae34, 0xb137, 0xb43a, 0xb73d}},
      {{0x8c83'8980, 0x9289'8f86, 0x888f'858c, 0x8e95'8b92},
       {0xa49b'a198, 0xaaa1'a79e, 0xa0a7'9da4, 0xa6ad'a3aa},
       {0xbcb3'b9b0, 0xc2b9'bfb6, 0xb8bf'b5bc, 0xbec5'bbc2},
       {0xd4cb'd1c8, 0xdad1'd7ce, 0xd0d7'cdd4, 0xd6dd'd3da},
       {0x6ce3'69e0, 0x72e9'6fe6, 0x68ef'65ec, 0x6ef5'6bf2},
       {0x84fb'81f8, 0x8b01'87fe, 0x8107'7e04, 0x870d'840a},
       {0x9d13'9a10, 0xa319'a016, 0x991f'961c, 0x9f25'9c22},
       {0xb52b'b228, 0xbb31'b82e, 0xb137'ae34, 0xb73d'b43a}},
      {{0x9289'8f87'0c83'8980, 0x8e95'8b92'888f'858c},
       {0xaaa1'a79f'249b'a198, 0xa6ad'a3aa'a0a7'9da4},
       {0xc2b9'bfb7'3cb3'b9b0, 0xbec5'bbc2'b8bf'b5bc},
       {0xdad1'd7cf'54cb'd1c8, 0xd6dd'd3da'd0d7'cdd4},
       {0x72e9'6fe6'ece3'69e0, 0x6ef5'6bf2'68ef'65ec},
       {0x8b01'87ff'04fb'81f8, 0x870d'840a'8107'7e04},
       {0xa319'a017'1d13'9a10, 0x9f25'9c22'991f'961c},
       {0xbb31'b82f'352b'b228, 0xb73d'b43a'b137'ae34}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaadduvx,
      ExecMaskedVaadduvx,
      {{85, 149, 86, 150, 87, 151, 88, 152, 89, 153, 90, 154, 91, 155, 92, 156},
       {93, 157, 94, 158, 95, 159, 96, 160, 97, 161, 98, 162, 99, 163, 100, 164},
       {101, 165, 102, 166, 103, 167, 104, 168, 105, 169, 106, 170, 107, 171, 108, 172},
       {109, 173, 110, 174, 111, 175, 112, 176, 113, 177, 114, 178, 115, 179, 116, 180},
       {117, 181, 118, 182, 119, 183, 120, 184, 121, 185, 122, 186, 123, 187, 124, 188},
       {125, 189, 126, 190, 127, 191, 128, 192, 129, 193, 130, 194, 131, 195, 132, 196},
       {133, 197, 134, 198, 135, 199, 136, 200, 137, 201, 138, 202, 139, 203, 140, 204},
       {141, 205, 142, 206, 143, 207, 144, 208, 145, 209, 146, 210, 147, 211, 148, 212}},
      {{0x95d5, 0x96d6, 0x97d7, 0x98d8, 0x99d9, 0x9ada, 0x9bdb, 0x9cdc},
       {0x9ddd, 0x9ede, 0x9fdf, 0xa0e0, 0xa1e1, 0xa2e2, 0xa3e3, 0xa4e4},
       {0xa5e5, 0xa6e6, 0xa7e7, 0xa8e8, 0xa9e9, 0xaaea, 0xabeb, 0xacec},
       {0xaded, 0xaeee, 0xafef, 0xb0f0, 0xb1f1, 0xb2f2, 0xb3f3, 0xb4f4},
       {0xb5f5, 0xb6f6, 0xb7f7, 0xb8f8, 0xb9f9, 0xbafa, 0xbbfb, 0xbcfc},
       {0xbdfd, 0xbefe, 0xbfff, 0xc100, 0xc201, 0xc302, 0xc403, 0xc504},
       {0xc605, 0xc706, 0xc807, 0xc908, 0xca09, 0xcb0a, 0xcc0b, 0xcd0c},
       {0xce0d, 0xcf0e, 0xd00f, 0xd110, 0xd211, 0xd312, 0xd413, 0xd514}},
      {{0x96d6'95d5, 0x98d8'97d7, 0x9ada'99d9, 0x9cdc'9bdb},
       {0x9ede'9ddd, 0xa0e0'9fdf, 0xa2e2'a1e1, 0xa4e4'a3e3},
       {0xa6e6'a5e5, 0xa8e8'a7e7, 0xaaea'a9e9, 0xacec'abeb},
       {0xaeee'aded, 0xb0f0'afef, 0xb2f2'b1f1, 0xb4f4'b3f3},
       {0xb6f6'b5f5, 0xb8f8'b7f7, 0xbafa'b9f9, 0xbcfc'bbfb},
       {0xbefe'bdfd, 0xc100'bfff, 0xc302'c201, 0xc504'c403},
       {0xc706'c605, 0xc908'c807, 0xcb0a'ca09, 0xcd0c'cc0b},
       {0xcf0e'ce0d, 0xd110'd00f, 0xd312'd211, 0xd514'd413}},
      {{0x98d8'97d7'96d6'95d5, 0x9cdc'9bdb'9ada'99d9},
       {0xa0e0'9fdf'9ede'9ddd, 0xa4e4'a3e3'a2e2'a1e1},
       {0xa8e8'a7e7'a6e6'a5e5, 0xacec'abeb'aaea'a9e9},
       {0xb0f0'afef'aeee'aded, 0xb4f4'b3f3'b2f2'b1f1},
       {0xb8f8'b7f7'b6f6'b5f5, 0xbcfc'bbfb'bafa'b9f9},
       {0xc100'bfff'befe'bdfd, 0xc504'c403'c302'c201},
       {0xc908'c807'c706'c605, 0xcd0c'cc0b'cb0a'ca09},
       {0xd110'd00f'cf0e'ce0d, 0xd514'd413'd312'd211}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaaddvv,
      ExecMaskedVaaddvv,
      {{0, 137, 3, 140, 6, 143, 9, 146, 12, 133, 15, 136, 18, 139, 21, 142},
       {24, 161, 27, 164, 30, 167, 33, 170, 36, 157, 39, 160, 42, 163, 45, 166},
       {48, 185, 51, 188, 54, 191, 57, 194, 60, 181, 63, 184, 66, 187, 69, 190},
       {72, 209, 75, 212, 78, 215, 81, 218, 84, 205, 87, 208, 90, 211, 93, 214},
       {224, 233, 227, 236, 230, 239, 233, 242, 236, 229, 239, 232, 242, 235, 245, 238},
       {248, 1, 251, 4, 254, 7, 1, 10, 4, 253, 7, 0, 10, 3, 13, 6},
       {16, 25, 19, 28, 22, 31, 25, 34, 28, 21, 31, 24, 34, 27, 37, 30},
       {40, 49, 43, 52, 46, 55, 49, 58, 52, 45, 55, 48, 58, 51, 61, 54}},
      {{0x8980, 0x8c83, 0x8f86, 0x9289, 0x858c, 0x888f, 0x8b92, 0x8e95},
       {0xa198, 0xa49b, 0xa79e, 0xaaa1, 0x9da4, 0xa0a7, 0xa3aa, 0xa6ad},
       {0xb9b0, 0xbcb3, 0xbfb6, 0xc2b9, 0xb5bc, 0xb8bf, 0xbbc2, 0xbec5},
       {0xd1c8, 0xd4cb, 0xd7ce, 0xdad1, 0xcdd4, 0xd0d7, 0xd3da, 0xd6dd},
       {0xe9e0, 0xece3, 0xefe6, 0xf2e9, 0xe5ec, 0xe8ef, 0xebf2, 0xeef5},
       {0x01f8, 0x04fb, 0x07fe, 0x0b01, 0xfe04, 0x0107, 0x040a, 0x070d},
       {0x1a10, 0x1d13, 0x2016, 0x2319, 0x161c, 0x191f, 0x1c22, 0x1f25},
       {0x3228, 0x352b, 0x382e, 0x3b31, 0x2e34, 0x3137, 0x343a, 0x373d}},
      {{0x8c83'8980, 0x9289'8f86, 0x888f'858c, 0x8e95'8b92},
       {0xa49b'a198, 0xaaa1'a79e, 0xa0a7'9da4, 0xa6ad'a3aa},
       {0xbcb3'b9b0, 0xc2b9'bfb6, 0xb8bf'b5bc, 0xbec5'bbc2},
       {0xd4cb'd1c8, 0xdad1'd7ce, 0xd0d7'cdd4, 0xd6dd'd3da},
       {0xece3'69e0, 0xf2e9'6fe6, 0xe8ef'65ec, 0xeef5'6bf2},
       {0x04fb'81f8, 0x0b01'87fe, 0x0107'7e04, 0x070d'840a},
       {0x1d13'9a10, 0x2319'a016, 0x191f'961c, 0x1f25'9c22},
       {0x352b'b228, 0x3b31'b82e, 0x3137'ae34, 0x373d'b43a}},
      {{0x9289'8f87'0c83'8980, 0x8e95'8b92'888f'858c},
       {0xaaa1'a79f'249b'a198, 0xa6ad'a3aa'a0a7'9da4},
       {0xc2b9'bfb7'3cb3'b9b0, 0xbec5'bbc2'b8bf'b5bc},
       {0xdad1'd7cf'54cb'd1c8, 0xd6dd'd3da'd0d7'cdd4},
       {0xf2e9'6fe6'ece3'69e0, 0xeef5'6bf2'68ef'65ec},
       {0x0b01'87ff'04fb'81f8, 0x070d'840a'8107'7e04},
       {0x2319'a017'1d13'9a10, 0x1f25'9c22'991f'961c},
       {0x3b31'b82f'352b'b228, 0x373d'b43a'b137'ae34}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVaaddvx,
      ExecMaskedVaaddvx,
      {{213, 149, 214, 150, 215, 151, 216, 152, 217, 153, 218, 154, 219, 155, 220, 156},
       {221, 157, 222, 158, 223, 159, 224, 160, 225, 161, 226, 162, 227, 163, 228, 164},
       {229, 165, 230, 166, 231, 167, 232, 168, 233, 169, 234, 170, 235, 171, 236, 172},
       {237, 173, 238, 174, 239, 175, 240, 176, 241, 177, 242, 178, 243, 179, 244, 180},
       {245, 181, 246, 182, 247, 183, 248, 184, 249, 185, 250, 186, 251, 187, 252, 188},
       {253, 189, 254, 190, 255, 191, 0, 192, 1, 193, 2, 194, 3, 195, 4, 196},
       {5, 197, 6, 198, 7, 199, 8, 200, 9, 201, 10, 202, 11, 203, 12, 204},
       {13, 205, 14, 206, 15, 207, 16, 208, 17, 209, 18, 210, 19, 211, 20, 212}},
      {{0x95d5, 0x96d6, 0x97d7, 0x98d8, 0x99d9, 0x9ada, 0x9bdb, 0x9cdc},
       {0x9ddd, 0x9ede, 0x9fdf, 0xa0e0, 0xa1e1, 0xa2e2, 0xa3e3, 0xa4e4},
       {0xa5e5, 0xa6e6, 0xa7e7, 0xa8e8, 0xa9e9, 0xaaea, 0xabeb, 0xacec},
       {0xaded, 0xaeee, 0xafef, 0xb0f0, 0xb1f1, 0xb2f2, 0xb3f3, 0xb4f4},
       {0xb5f5, 0xb6f6, 0xb7f7, 0xb8f8, 0xb9f9, 0xbafa, 0xbbfb, 0xbcfc},
       {0xbdfd, 0xbefe, 0xbfff, 0xc100, 0xc201, 0xc302, 0xc403, 0xc504},
       {0xc605, 0xc706, 0xc807, 0xc908, 0xca09, 0xcb0a, 0xcc0b, 0xcd0c},
       {0xce0d, 0xcf0e, 0xd00f, 0xd110, 0xd211, 0xd312, 0xd413, 0xd514}},
      {{0x96d6'95d5, 0x98d8'97d7, 0x9ada'99d9, 0x9cdc'9bdb},
       {0x9ede'9ddd, 0xa0e0'9fdf, 0xa2e2'a1e1, 0xa4e4'a3e3},
       {0xa6e6'a5e5, 0xa8e8'a7e7, 0xaaea'a9e9, 0xacec'abeb},
       {0xaeee'aded, 0xb0f0'afef, 0xb2f2'b1f1, 0xb4f4'b3f3},
       {0xb6f6'b5f5, 0xb8f8'b7f7, 0xbafa'b9f9, 0xbcfc'bbfb},
       {0xbefe'bdfd, 0xc100'bfff, 0xc302'c201, 0xc504'c403},
       {0xc706'c605, 0xc908'c807, 0xcb0a'ca09, 0xcd0c'cc0b},
       {0xcf0e'ce0d, 0xd110'd00f, 0xd312'd211, 0xd514'd413}},
      {{0x98d8'97d7'96d6'95d5, 0x9cdc'9bdb'9ada'99d9},
       {0xa0e0'9fdf'9ede'9ddd, 0xa4e4'a3e3'a2e2'a1e1},
       {0xa8e8'a7e7'a6e6'a5e5, 0xacec'abeb'aaea'a9e9},
       {0xb0f0'afef'aeee'aded, 0xb4f4'b3f3'b2f2'b1f1},
       {0xb8f8'b7f7'b6f6'b5f5, 0xbcfc'bbfb'bafa'b9f9},
       {0xc100'bfff'befe'bdfd, 0xc504'c403'c302'c201},
       {0xc908'c807'c706'c605, 0xcd0c'cc0b'cb0a'ca09},
       {0xd110'd00f'cf0e'ce0d, 0xd514'd413'd312'd211}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubuvv,
      ExecMaskedVasubuvv,
      {{0, 247, 255, 246, 253, 245, 253, 244, 251, 3, 251, 2, 250, 1, 249, 0},
       {248, 239, 247, 238, 245, 237, 245, 236, 243, 251, 243, 250, 242, 249, 241, 248},
       {240, 231, 239, 230, 237, 229, 237, 228, 235, 243, 235, 242, 234, 241, 233, 240},
       {232, 223, 231, 222, 229, 221, 229, 220, 227, 235, 227, 234, 226, 233, 225, 232},
       {224, 87, 223, 86, 221, 85, 221, 84, 219, 99, 219, 98, 218, 97, 217, 96},
       {216, 79, 215, 78, 213, 77, 213, 76, 211, 91, 211, 90, 210, 89, 209, 88},
       {208, 71, 207, 70, 205, 69, 205, 68, 203, 83, 203, 82, 202, 81, 201, 80},
       {200, 63, 199, 62, 197, 61, 197, 60, 195, 75, 195, 74, 194, 73, 193, 72}},
      {{0xf780, 0xf67f, 0xf57d, 0xf47d, 0x037b, 0x027b, 0x017a, 0x0079},
       {0xef78, 0xee77, 0xed75, 0xec75, 0xfb73, 0xfa73, 0xf972, 0xf871},
       {0xe770, 0xe66f, 0xe56d, 0xe46d, 0xf36b, 0xf26b, 0xf16a, 0xf069},
       {0xdf68, 0xde67, 0xdd65, 0xdc65, 0xeb63, 0xea63, 0xe962, 0xe861},
       {0x5760, 0x565f, 0x555d, 0x545d, 0x635b, 0x625b, 0x615a, 0x6059},
       {0x4f58, 0x4e57, 0x4d55, 0x4c55, 0x5b53, 0x5a53, 0x5952, 0x5851},
       {0x4750, 0x464f, 0x454d, 0x444d, 0x534b, 0x524b, 0x514a, 0x5049},
       {0x3f48, 0x3e47, 0x3d45, 0x3c45, 0x4b43, 0x4a43, 0x4942, 0x4841}},
      {{0xf67e'f780, 0xf47c'f57d, 0x027b'037b, 0x0079'017a},
       {0xee76'ef78, 0xec74'ed75, 0xfa72'fb73, 0xf870'f972},
       {0xe66e'e770, 0xe46c'e56d, 0xf26a'f36b, 0xf068'f16a},
       {0xde66'df68, 0xdc64'dd65, 0xea62'eb63, 0xe860'e962},
       {0x565f'5760, 0x545d'555d, 0x625b'635b, 0x6059'615a},
       {0x4e57'4f58, 0x4c55'4d55, 0x5a53'5b53, 0x5851'5952},
       {0x464f'4750, 0x444d'454d, 0x524b'534b, 0x5049'514a},
       {0x3e47'3f48, 0x3c45'3d45, 0x4a43'4b43, 0x4841'4942}},
      {{0xf47c'f57d'767e'f780, 0x0079'017a'027b'037b},
       {0xec74'ed75'6e76'ef78, 0xf870'f971'fa72'fb73},
       {0xe46c'e56d'666e'e770, 0xf068'f169'f26a'f36b},
       {0xdc64'dd65'5e66'df68, 0xe860'e961'ea62'eb63},
       {0x545d'555d'd65f'5760, 0x6059'615a'625b'635b},
       {0x4c55'4d55'ce57'4f58, 0x5851'5952'5a53'5b53},
       {0x444d'454d'c64f'4750, 0x5049'514a'524b'534b},
       {0x3c45'3d45'be47'3f48, 0x4841'4942'4a43'4b43}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubuvx,
      ExecMaskedVasubuvx,
      {{171, 235, 172, 236, 173, 237, 174, 238, 175, 239, 176, 240, 177, 241, 178, 242},
       {179, 243, 180, 244, 181, 245, 182, 246, 183, 247, 184, 248, 185, 249, 186, 250},
       {187, 251, 188, 252, 189, 253, 190, 254, 191, 255, 192, 0, 193, 1, 194, 2},
       {195, 3, 196, 4, 197, 5, 198, 6, 199, 7, 200, 8, 201, 9, 202, 10},
       {203, 11, 204, 12, 205, 13, 206, 14, 207, 15, 208, 16, 209, 17, 210, 18},
       {211, 19, 212, 20, 213, 21, 214, 22, 215, 23, 216, 24, 217, 25, 218, 26},
       {219, 27, 220, 28, 221, 29, 222, 30, 223, 31, 224, 32, 225, 33, 226, 34},
       {227, 35, 228, 36, 229, 37, 230, 38, 231, 39, 232, 40, 233, 41, 234, 42}},
      {{0xeb2b, 0xec2c, 0xed2d, 0xee2e, 0xef2f, 0xf030, 0xf131, 0xf232},
       {0xf333, 0xf434, 0xf535, 0xf636, 0xf737, 0xf838, 0xf939, 0xfa3a},
       {0xfb3b, 0xfc3c, 0xfd3d, 0xfe3e, 0xff3f, 0x0040, 0x0141, 0x0242},
       {0x0343, 0x0444, 0x0545, 0x0646, 0x0747, 0x0848, 0x0949, 0x0a4a},
       {0x0b4b, 0x0c4c, 0x0d4d, 0x0e4e, 0x0f4f, 0x1050, 0x1151, 0x1252},
       {0x1353, 0x1454, 0x1555, 0x1656, 0x1757, 0x1858, 0x1959, 0x1a5a},
       {0x1b5b, 0x1c5c, 0x1d5d, 0x1e5e, 0x1f5f, 0x2060, 0x2161, 0x2262},
       {0x2363, 0x2464, 0x2565, 0x2666, 0x2767, 0x2868, 0x2969, 0x2a6a}},
      {{0xec2b'eb2b, 0xee2d'ed2d, 0xf02f'ef2f, 0xf231'f131},
       {0xf433'f333, 0xf635'f535, 0xf837'f737, 0xfa39'f939},
       {0xfc3b'fb3b, 0xfe3d'fd3d, 0x003f'ff3f, 0x0242'0141},
       {0x0444'0343, 0x0646'0545, 0x0848'0747, 0x0a4a'0949},
       {0x0c4c'0b4b, 0x0e4e'0d4d, 0x1050'0f4f, 0x1252'1151},
       {0x1454'1353, 0x1656'1555, 0x1858'1757, 0x1a5a'1959},
       {0x1c5c'1b5b, 0x1e5e'1d5d, 0x2060'1f5f, 0x2262'2161},
       {0x2464'2363, 0x2666'2565, 0x2868'2767, 0x2a6a'2969}},
      {{0xee2d'ed2c'ec2b'eb2b, 0xf231'f130'f02f'ef2f},
       {0xf635'f534'f433'f333, 0xfa39'f938'f837'f737},
       {0xfe3d'fd3c'fc3b'fb3b, 0x0242'0141'003f'ff3f},
       {0x0646'0545'0444'0343, 0x0a4a'0949'0848'0747},
       {0x0e4e'0d4d'0c4c'0b4b, 0x1252'1151'1050'0f4f},
       {0x1656'1555'1454'1353, 0x1a5a'1959'1858'1757},
       {0x1e5e'1d5d'1c5c'1b5b, 0x2262'2161'2060'1f5f},
       {0x2666'2565'2464'2363, 0x2a6a'2969'2868'2767}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      ExecVasubvv,
      ExecMaskedVasubvv,
      {{0, 247, 255, 246, 253, 245, 253, 244, 251, 3, 251, 2, 250, 1, 249, 0},
       {248, 239, 247, 238, 245, 237, 245, 236, 243, 251, 243, 250, 242, 249, 241, 248},
       {240, 231, 239, 230, 237, 229, 237, 228, 235, 243, 235, 242, 234, 241, 233, 240},
       {232, 223, 231, 222, 229, 221, 229, 220, 227, 235, 227, 234, 226, 233, 225, 232},
       {96, 215, 95, 214, 93, 213, 93, 212, 91, 227, 91, 226, 90, 225, 89, 224},
       {88, 207, 87, 206, 85, 205, 85, 204, 83, 219, 83, 218, 82, 217, 81, 216},
       {80, 199, 79, 198, 77, 197, 77, 196, 75, 211, 75, 210, 74, 209, 73, 208},
       {72, 191, 71, 190, 69, 189, 69, 188, 67, 203, 67, 202, 66, 201, 65, 200}},
      {{0xf780, 0xf67f, 0xf57d, 0xf47d, 0x037b, 0x027b, 0x017a, 0x0079},
       {0xef78, 0xee77, 0xed75, 0xec75, 0xfb73, 0xfa73, 0xf972, 0xf871},
       {0xe770, 0xe66f, 0xe56d, 0xe46d, 0xf36b, 0xf26b, 0xf16a, 0xf069},
       {0xdf68, 0xde67, 0xdd65, 0xdc65, 0xeb63, 0xea63, 0xe962, 0xe861},
       {0xd760, 0xd65f, 0xd55d, 0xd45d, 0xe35b, 0xe25b, 0xe15a, 0xe059},
       {0xcf58, 0xce57, 0xcd55, 0xcc55, 0xdb53, 0xda53, 0xd952, 0xd851},
       {0xc750, 0xc64f, 0xc54d, 0xc44d, 0xd34b, 0xd24b, 0xd14a, 0xd049},
       {0xbf48, 0xbe47, 0xbd45, 0xbc45, 0xcb43, 0xca43, 0xc942, 0xc841}},
      {{0xf67e'f780, 0xf47c'f57d, 0x027b'037b, 0x0079'017a},
       {0xee76'ef78, 0xec74'ed75, 0xfa72'fb73, 0xf870'f972},
       {0xe66e'e770, 0xe46c'e56d, 0xf26a'f36b, 0xf068'f16a},
       {0xde66'df68, 0xdc64'dd65, 0xea62'eb63, 0xe860'e962},
       {0xd65f'5760, 0xd45d'555d, 0xe25b'635b, 0xe059'615a},
       {0xce57'4f58, 0xcc55'4d55, 0xda53'5b53, 0xd851'5952},
       {0xc64f'4750, 0xc44d'454d, 0xd24b'534b, 0xd049'514a},
       {0xbe47'3f48, 0xbc45'3d45, 0xca43'4b43, 0xc841'4942}},
      {{0xf47c'f57d'767e'f780, 0x0079'017a'027b'037b},
       {0xec74'ed75'6e76'ef78, 0xf870'f971'fa72'fb73},
       {0xe46c'e56d'666e'e770, 0xf068'f169'f26a'f36b},
       {0xdc64'dd65'5e66'df68, 0xe860'e961'ea62'eb63},
       {0xd45d'555d'd65f'5760, 0xe059'615a'625b'635b},
       {0xcc55'4d55'ce57'4f58, 0xd851'5952'5a53'5b53},
       {0xc44d'454d'c64f'4750, 0xd049'514a'524b'534b},
       {0xbc45'3d45'be47'3f48, 0xc841'4942'4a43'4b43}},
      kVectorCalculationsSource);
  TestVectorInstruction(ExecVasubvx,
                        ExecMaskedVasubvx,
                        {{43, 235, 44, 236, 45, 237, 46, 238, 47, 239, 48, 240, 49, 241, 50, 242},
                         {51, 243, 52, 244, 53, 245, 54, 246, 55, 247, 56, 248, 57, 249, 58, 250},
                         {59, 251, 60, 252, 61, 253, 62, 254, 63, 255, 64, 0, 65, 1, 66, 2},
                         {67, 3, 68, 4, 69, 5, 70, 6, 71, 7, 72, 8, 73, 9, 74, 10},
                         {75, 11, 76, 12, 77, 13, 78, 14, 79, 15, 80, 16, 81, 17, 82, 18},
                         {83, 19, 84, 20, 85, 21, 86, 22, 87, 23, 88, 24, 89, 25, 90, 26},
                         {91, 27, 92, 28, 93, 29, 94, 30, 95, 31, 96, 32, 97, 33, 98, 34},
                         {99, 35, 100, 36, 101, 37, 102, 38, 103, 39, 104, 40, 105, 41, 106, 42}},
                        {{0xeb2b, 0xec2c, 0xed2d, 0xee2e, 0xef2f, 0xf030, 0xf131, 0xf232},
                         {0xf333, 0xf434, 0xf535, 0xf636, 0xf737, 0xf838, 0xf939, 0xfa3a},
                         {0xfb3b, 0xfc3c, 0xfd3d, 0xfe3e, 0xff3f, 0x0040, 0x0141, 0x0242},
                         {0x0343, 0x0444, 0x0545, 0x0646, 0x0747, 0x0848, 0x0949, 0x0a4a},
                         {0x0b4b, 0x0c4c, 0x0d4d, 0x0e4e, 0x0f4f, 0x1050, 0x1151, 0x1252},
                         {0x1353, 0x1454, 0x1555, 0x1656, 0x1757, 0x1858, 0x1959, 0x1a5a},
                         {0x1b5b, 0x1c5c, 0x1d5d, 0x1e5e, 0x1f5f, 0x2060, 0x2161, 0x2262},
                         {0x2363, 0x2464, 0x2565, 0x2666, 0x2767, 0x2868, 0x2969, 0x2a6a}},
                        {{0xec2b'eb2b, 0xee2d'ed2d, 0xf02f'ef2f, 0xf231'f131},
                         {0xf433'f333, 0xf635'f535, 0xf837'f737, 0xfa39'f939},
                         {0xfc3b'fb3b, 0xfe3d'fd3d, 0x003f'ff3f, 0x0242'0141},
                         {0x0444'0343, 0x0646'0545, 0x0848'0747, 0x0a4a'0949},
                         {0x0c4c'0b4b, 0x0e4e'0d4d, 0x1050'0f4f, 0x1252'1151},
                         {0x1454'1353, 0x1656'1555, 0x1858'1757, 0x1a5a'1959},
                         {0x1c5c'1b5b, 0x1e5e'1d5d, 0x2060'1f5f, 0x2262'2161},
                         {0x2464'2363, 0x2666'2565, 0x2868'2767, 0x2a6a'2969}},
                        {{0xee2d'ed2c'ec2b'eb2b, 0xf231'f130'f02f'ef2f},
                         {0xf635'f534'f433'f333, 0xfa39'f938'f837'f737},
                         {0xfe3d'fd3c'fc3b'fb3b, 0x0242'0141'003f'ff3f},
                         {0x0646'0545'0444'0343, 0x0a4a'0949'0848'0747},
                         {0x0e4e'0d4d'0c4c'0b4b, 0x1252'1151'1050'0f4f},
                         {0x1656'1555'1454'1353, 0x1a5a'1959'1858'1757},
                         {0x1e5e'1d5d'1c5c'1b5b, 0x2262'2161'2060'1f5f},
                         {0x2666'2565'2464'2363, 0x2a6a'2969'2868'2767}},
                        kVectorCalculationsSource);
  TestNarrowingVectorInstruction(ExecVnclipuwi,
                                 ExecMaskedVnclipuwi,
                                 {{32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39},
                                  {40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47},
                                  {48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55},
                                  {56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63}},
                                 {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
                                 {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                                  {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff}},
                                 kVectorCalculationsSource);
  TestNarrowingVectorInstruction(
      ExecVnclipwi,
      ExecMaskedVnclipwi,
      {{224, 224, 225, 225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231},
       {232, 232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 239},
       {240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247},
       {248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0xdfbf}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(ExecVnclipuwx,
                                 ExecMaskedVnclipuwx,
                                 {{32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39},
                                  {40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47},
                                  {48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55},
                                  {56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63}},
                                 {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
                                  {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}},
                                 {{0x0021'c1a1, 0x0023'c3a3, 0x0025'c5a5, 0x0027'c7a7},
                                  {0x0029'c9a9, 0x002b'cbab, 0x002d'cdad, 0x002f'cfaf},
                                  {0x0031'd1b1, 0x0033'd3b3, 0x0035'd5b5, 0x0037'd7b7},
                                  {0x0039'd9b9, 0x003b'dbbb, 0x003d'ddbd, 0x003f'dfbf}},
                                 kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipwx,
      ExecMaskedVnclipwx,
      {{224, 224, 225, 225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231},
       {232, 232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 239},
       {240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247},
       {248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000},
       {0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0xdfbf}},
      {{0xffe1'c1a1, 0xffe3'c3a3, 0xffe5'c5a5, 0xffe7'c7a7},
       {0xffe9'c9a9, 0xffeb'cbab, 0xffed'cdad, 0xffef'cfaf},
       {0xfff1'd1b1, 0xfff3'd3b3, 0xfff5'd5b5, 0xfff7'd7b7},
       {0xfff9'd9b9, 0xfffb'dbbb, 0xfffd'ddbd, 0xffff'dfbf}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipuwv,
      ExecMaskedVnclipuwv,
      {{255, 255, 255, 255, 68, 34, 8, 2, 255, 255, 255, 255, 153, 38, 9, 2},
       {255, 255, 255, 255, 84, 42, 10, 2, 255, 255, 255, 255, 185, 46, 11, 2},
       {255, 255, 255, 255, 100, 50, 12, 3, 255, 255, 255, 255, 217, 54, 13, 3},
       {255, 255, 255, 255, 116, 58, 14, 3, 255, 255, 255, 255, 249, 62, 15, 3}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0x4989, 0x0971, 0x009b, 0x0009},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x5999, 0x0b73, 0x00bb, 0x000b},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x69a9, 0x0d75, 0x00db, 0x000d},
       {0xffff, 0xffff, 0xffff, 0xffff, 0x79b9, 0x0f77, 0x00fb, 0x000f}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xa726'a524, 0x0057'9756, 0x0000'5b9b, 0x0000'00bf},
       {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xe766'e564, 0x0077'b776, 0x0000'7bbb, 0x0000'00ff}},
      kVectorCalculationsSource);

  TestNarrowingVectorInstruction(
      ExecVnclipwv,
      ExecMaskedVnclipwv,
      {{128, 128, 128, 128, 196, 226, 248, 254, 128, 128, 128, 128, 153, 230, 249, 254},
       {128, 128, 128, 128, 212, 234, 250, 254, 128, 128, 128, 128, 185, 238, 251, 254},
       {128, 128, 128, 128, 228, 242, 252, 255, 128, 128, 128, 128, 217, 246, 253, 255},
       {128, 128, 128, 157, 244, 250, 254, 255, 128, 128, 128, 221, 249, 254, 255, 255}},
      {{0x8000, 0x8000, 0x8000, 0x8000, 0xc989, 0xf971, 0xff9b, 0xfff9},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xd999, 0xfb73, 0xffbb, 0xfffb},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xe9a9, 0xfd75, 0xffdb, 0xfffd},
       {0x8000, 0x8000, 0x8000, 0x8000, 0xf9b9, 0xff77, 0xfffb, 0xffff}},
      {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0xa726'a524, 0xffd7'9756, 0xffff'db9b, 0xffff'ffbf},
       {0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
       {0xe766'e564, 0xfff7'b776, 0xffff'fbbb, 0xffff'ffff}},
      kVectorCalculationsSource);

  TestVectorInstruction(
      ExecVsmulvv,
      ExecMaskedVsmulvv,
      {{0, 109, 0, 103, 0, 98, 0, 92, 1, 117, 1, 111, 2, 106, 3, 100},
       {4, 67, 5, 63, 6, 58, 7, 54, 9, 75, 10, 71, 12, 66, 14, 62},
       {16, 34, 18, 30, 20, 27, 22, 23, 25, 42, 27, 38, 30, 35, 33, 31},
       {36, 8, 39, 6, 42, 3, 45, 1, 49, 16, 52, 14, 56, 11, 60, 9},
       {192, 247, 192, 245, 192, 244, 192, 242, 193, 255, 193, 253, 194, 252, 195, 250},
       {196, 237, 197, 237, 198, 236, 199, 236, 201, 245, 202, 245, 204, 244, 206, 244},
       {208, 236, 210, 236, 213, 237, 214, 237, 217, 244, 219, 244, 222, 245, 225, 245},
       {228, 242, 231, 244, 235, 245, 237, 247, 241, 250, 244, 252, 248, 253, 252, 255}},
      {{0x6d24, 0x677e, 0x61f8, 0x5c94, 0x750c, 0x6f68, 0x69e3, 0x647e},
       {0x437e, 0x3eda, 0x3a56, 0x35f4, 0x4b6a, 0x46c8, 0x4245, 0x3de3},
       {0x21e9, 0x1e47, 0x1ac5, 0x1765, 0x29d9, 0x2639, 0x22b8, 0x1f57},
       {0x0863, 0x05c4, 0x0344, 0x00e5, 0x1058, 0x0db9, 0x0b3b, 0x08dc},
       {0xf6ee, 0xf550, 0xf3d2, 0xf276, 0xfee7, 0xfd4a, 0xfbcd, 0xfa71},
       {0xed88, 0xeced, 0xec71, 0xec17, 0xf585, 0xf4eb, 0xf470, 0xf415},
       {0xec33, 0xec9a, 0xed20, 0xedc7, 0xf434, 0xf49b, 0xf523, 0xf5ca},
       {0xf2ee, 0xf456, 0xf5df, 0xf788, 0xfaf3, 0xfc5c, 0xfde5, 0xff8f}},
      {{0x677d'76ae, 0x5c93'1930, 0x6f67'3830, 0x647d'dbb6},
       {0x3eda'09c6, 0x35f3'b250, 0x46c7'cf50, 0x3de2'78dd},
       {0x1e46'b4fd, 0x1764'638f, 0x2638'7e8f, 0x1f57'2e25},
       {0x05c3'7854, 0x00e5'2cef, 0x0db9'45ef, 0x08db'fb8c},
       {0xf550'cd46, 0xf276'7fe1, 0xfd4a'8ed9, 0xfa71'4276},
       {0xeced'a0be, 0xec17'5961, 0xf4eb'6659, 0xf416'1ffe},
       {0xec9a'8c56, 0xedc8'4b00, 0xf49c'55f8, 0xf5cb'15a6},
       {0xf457'900d, 0xf789'54c0, 0xfc5d'5db8, 0xff90'236d}},
      {{0x5c93'192f'ccd4'7781, 0x647d'dbb5'bb66'23af},
       {0x35f3'b24f'43d0'aa38, 0x3de2'78dd'1a4e'4256},
       {0x1764'638e'e2fd'152f, 0x1f57'2e24'a166'993d},
       {0x00e5'2cee'aa59'b866, 0x08db'fb8c'50af'2864},
       {0xf276'7fe1'80cf'f441, 0xfa71'4276'eef1'1fff},
       {0xec17'5961'584c'a798, 0xf416'1ffe'ae59'bf46},
       {0xedc8'4b01'57f9'9330, 0xf5cb'15a6'95f2'96ce},
       {0xf789'54c1'7fd6'b708, 0xff90'236e'a5bb'a696}},
      kVectorCalculationsSource);
  TestVectorInstruction(ExecVsmulvx,
                        ExecMaskedVsmulvx,
                        {{0, 85, 254, 83, 253, 82, 251, 81, 250, 79, 249, 78, 247, 77, 246, 75},
                         {245, 74, 243, 73, 242, 71, 241, 70, 239, 69, 238, 67, 237, 66, 235, 65},
                         {234, 63, 233, 62, 231, 61, 230, 59, 229, 58, 227, 57, 226, 55, 225, 54},
                         {223, 53, 222, 51, 221, 50, 219, 49, 218, 47, 217, 46, 215, 45, 214, 43},
                         {213, 42, 211, 40, 210, 39, 208, 38, 207, 36, 206, 35, 204, 34, 203, 32},
                         {202, 31, 200, 30, 199, 28, 198, 27, 196, 26, 195, 24, 194, 23, 192, 22},
                         {191, 20, 190, 19, 188, 18, 187, 16, 186, 15, 184, 14, 183, 12, 182, 11},
                         {180, 10, 179, 8, 178, 7, 176, 6, 175, 4, 174, 3, 172, 2, 171, 0}},
                        {{0x54ab, 0x5354, 0x51fd, 0x50a7, 0x4f50, 0x4df9, 0x4ca3, 0x4b4c},
                         {0x49f5, 0x489f, 0x4748, 0x45f1, 0x449b, 0x4344, 0x41ed, 0x4097},
                         {0x3f40, 0x3de9, 0x3c93, 0x3b3c, 0x39e5, 0x388f, 0x3738, 0x35e1},
                         {0x348b, 0x3334, 0x31dd, 0x3087, 0x2f30, 0x2dd9, 0x2c83, 0x2b2c},
                         {0x29d5, 0x287e, 0x2728, 0x25d1, 0x247a, 0x2324, 0x21cd, 0x2076},
                         {0x1f20, 0x1dc9, 0x1c72, 0x1b1c, 0x19c5, 0x186e, 0x1718, 0x15c1},
                         {0x146a, 0x1314, 0x11bd, 0x1066, 0x0f10, 0x0db9, 0x0c62, 0x0b0c},
                         {0x09b5, 0x085e, 0x0708, 0x05b1, 0x045a, 0x0304, 0x01ad, 0x0056}},
                        {{0x5353'aa00, 0x50a6'51fd, 0x4df8'f9fb, 0x4b4b'a1f8},
                         {0x489e'49f5, 0x45f0'f1f3, 0x4343'99f0, 0x4096'41ed},
                         {0x3de8'e9eb, 0x3b3b'91e8, 0x388e'39e5, 0x35e0'e1e3},
                         {0x3333'89e0, 0x3086'31dd, 0x2dd8'd9db, 0x2b2b'81d8},
                         {0x287e'29d5, 0x25d0'd1d2, 0x2323'79d0, 0x2076'21cd},
                         {0x1dc8'c9ca, 0x1b1b'71c8, 0x186e'19c5, 0x15c0'c1c2},
                         {0x1313'69c0, 0x1066'11bd, 0x0db8'b9ba, 0x0b0b'61b8},
                         {0x085e'09b5, 0x05b0'b1b2, 0x0303'59b0, 0x0056'01ad}},
                        {{0x50a6'51fc'fdfe'54ab, 0x4b4b'a1f7'a34e'4f50},
                         {0x45f0'f1f2'489e'49f5, 0x4096'41ec'edee'449b},
                         {0x3b3b'91e7'933e'3f40, 0x35e0'e1e2'388e'39e5},
                         {0x3086'31dc'ddde'348b, 0x2b2b'81d7'832e'2f30},
                         {0x25d0'd1d2'287e'29d5, 0x2076'21cc'cdce'247a},
                         {0x1b1b'71c7'731e'1f20, 0x15c0'c1c2'186e'19c5},
                         {0x1066'11bc'bdbe'146a, 0x0b0b'61b7'630e'0f10},
                         {0x05b0'b1b2'085e'09b5, 0x0056'01ac'adae'045a}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvv,
                        ExecMaskedVssrlvv,
                        {{0, 32, 0, 2, 2, 33, 0, 2, 4, 34, 0, 2, 12, 35, 0, 2},
                         {16, 36, 1, 2, 10, 37, 1, 2, 12, 38, 1, 2, 28, 39, 1, 2},
                         {32, 40, 2, 2, 18, 41, 2, 2, 20, 42, 2, 2, 44, 43, 2, 2},
                         {48, 44, 3, 2, 26, 45, 3, 2, 28, 46, 3, 2, 60, 47, 3, 2},
                         {64, 48, 4, 3, 34, 49, 4, 3, 36, 50, 4, 3, 76, 51, 4, 3},
                         {80, 52, 5, 3, 42, 53, 5, 3, 44, 54, 5, 3, 92, 55, 5, 3},
                         {96, 56, 6, 3, 50, 57, 6, 3, 52, 58, 6, 3, 108, 59, 6, 3},
                         {112, 60, 7, 3, 58, 61, 7, 3, 60, 62, 7, 3, 124, 63, 7, 3}},
                        {{0x8100, 0x0830, 0x0042, 0x0008, 0x4484, 0x08b0, 0x008d, 0x0008},
                         {0x9110, 0x0931, 0x004a, 0x0009, 0x4c8c, 0x09b1, 0x009d, 0x0009},
                         {0xa120, 0x0a32, 0x0052, 0x000a, 0x5494, 0x0ab2, 0x00ad, 0x000a},
                         {0xb130, 0x0b33, 0x005a, 0x000b, 0x5c9c, 0x0bb3, 0x00bd, 0x000b},
                         {0xc140, 0x0c34, 0x0062, 0x000c, 0x64a4, 0x0cb4, 0x00cd, 0x000c},
                         {0xd150, 0x0d35, 0x006a, 0x000d, 0x6cac, 0x0db5, 0x00dd, 0x000d},
                         {0xe160, 0x0e36, 0x0072, 0x000e, 0x74b4, 0x0eb6, 0x00ed, 0x000e},
                         {0xf170, 0x0f37, 0x007a, 0x000f, 0x7cbc, 0x0fb7, 0x00fd, 0x000f}},
                        {{0x8302'8100, 0x0043'8342, 0x0000'4585, 0x0000'008f},
                         {0x9312'9110, 0x004b'8b4a, 0x0000'4d8d, 0x0000'009f},
                         {0xa322'a120, 0x0053'9352, 0x0000'5595, 0x0000'00af},
                         {0xb332'b130, 0x005b'9b5a, 0x0000'5d9d, 0x0000'00bf},
                         {0xc342'c140, 0x0063'a362, 0x0000'65a5, 0x0000'00cf},
                         {0xd352'd150, 0x006b'ab6a, 0x0000'6dad, 0x0000'00df},
                         {0xe362'e160, 0x0073'b372, 0x0000'75b5, 0x0000'00ef},
                         {0xf372'f170, 0x007b'bb7a, 0x0000'7dbd, 0x0000'00ff}},
                        {{0x8706'8504'8302'8100, 0x0000'4787'4686'4585},
                         {0x0000'0000'9716'9514, 0x0000'0000'0000'4f8f},
                         {0xa726'a524'a322'a120, 0x0000'5797'5696'5595},
                         {0x0000'0000'b736'b534, 0x0000'0000'0000'5f9f},
                         {0xc746'c544'c342'c140, 0x0000'67a7'66a6'65a5},
                         {0x0000'0000'd756'd554, 0x0000'0000'0000'6faf},
                         {0xe766'e564'e362'e160, 0x0000'77b7'76b6'75b5},
                         {0x0000'0000'f776'f574, 0x0000'0000'0000'7fbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvx,
                        ExecMaskedVssrlvx,
                        {{0, 32, 0, 32, 1, 33, 1, 33, 2, 34, 2, 34, 3, 35, 3, 35},
                         {4, 36, 4, 36, 5, 37, 5, 37, 6, 38, 6, 38, 7, 39, 7, 39},
                         {8, 40, 8, 40, 9, 41, 9, 41, 10, 42, 10, 42, 11, 43, 11, 43},
                         {12, 44, 12, 44, 13, 45, 13, 45, 14, 46, 14, 46, 15, 47, 15, 47},
                         {16, 48, 16, 48, 17, 49, 17, 49, 18, 50, 18, 50, 19, 51, 19, 51},
                         {20, 52, 20, 52, 21, 53, 21, 53, 22, 54, 22, 54, 23, 55, 23, 55},
                         {24, 56, 24, 56, 25, 57, 25, 57, 26, 58, 26, 58, 27, 59, 27, 59},
                         {28, 60, 28, 60, 29, 61, 29, 61, 30, 62, 30, 62, 31, 63, 31, 63}},
                        {{0x0020, 0x0020, 0x0021, 0x0021, 0x0022, 0x0022, 0x0023, 0x0023},
                         {0x0024, 0x0024, 0x0025, 0x0025, 0x0026, 0x0026, 0x0027, 0x0027},
                         {0x0028, 0x0028, 0x0029, 0x0029, 0x002a, 0x002a, 0x002b, 0x002b},
                         {0x002c, 0x002c, 0x002d, 0x002d, 0x002e, 0x002e, 0x002f, 0x002f},
                         {0x0030, 0x0030, 0x0031, 0x0031, 0x0032, 0x0032, 0x0033, 0x0033},
                         {0x0034, 0x0034, 0x0035, 0x0035, 0x0036, 0x0036, 0x0037, 0x0037},
                         {0x0038, 0x0038, 0x0039, 0x0039, 0x003a, 0x003a, 0x003b, 0x003b},
                         {0x003c, 0x003c, 0x003d, 0x003d, 0x003e, 0x003e, 0x003f, 0x003f}},
                        {{0x0020'c0a0, 0x0021'c1a1, 0x0022'c2a2, 0x0023'c3a3},
                         {0x0024'c4a4, 0x0025'c5a5, 0x0026'c6a6, 0x0027'c7a7},
                         {0x0028'c8a8, 0x0029'c9a9, 0x002a'caaa, 0x002b'cbab},
                         {0x002c'ccac, 0x002d'cdad, 0x002e'ceae, 0x002f'cfaf},
                         {0x0030'd0b0, 0x0031'd1b1, 0x0032'd2b2, 0x0033'd3b3},
                         {0x0034'd4b4, 0x0035'd5b5, 0x0036'd6b6, 0x0037'd7b7},
                         {0x0038'd8b8, 0x0039'd9b9, 0x003a'daba, 0x003b'dbbb},
                         {0x003c'dcbc, 0x003d'ddbd, 0x003e'debe, 0x003f'dfbf}},
                        {{0x0000'0000'0021'c1a1, 0x0000'0000'0023'c3a3},
                         {0x0000'0000'0025'c5a5, 0x0000'0000'0027'c7a7},
                         {0x0000'0000'0029'c9a9, 0x0000'0000'002b'cbab},
                         {0x0000'0000'002d'cdad, 0x0000'0000'002f'cfaf},
                         {0x0000'0000'0031'd1b1, 0x0000'0000'0033'd3b3},
                         {0x0000'0000'0035'd5b5, 0x0000'0000'0037'd7b7},
                         {0x0000'0000'0039'd9b9, 0x0000'0000'003b'dbbb},
                         {0x0000'0000'003d'ddbd, 0x0000'0000'003f'dfbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssrlvi,
                        ExecMaskedVssrlvi,
                        {{0, 32, 0, 32, 1, 33, 1, 33, 2, 34, 2, 34, 3, 35, 3, 35},
                         {4, 36, 4, 36, 5, 37, 5, 37, 6, 38, 6, 38, 7, 39, 7, 39},
                         {8, 40, 8, 40, 9, 41, 9, 41, 10, 42, 10, 42, 11, 43, 11, 43},
                         {12, 44, 12, 44, 13, 45, 13, 45, 14, 46, 14, 46, 15, 47, 15, 47},
                         {16, 48, 16, 48, 17, 49, 17, 49, 18, 50, 18, 50, 19, 51, 19, 51},
                         {20, 52, 20, 52, 21, 53, 21, 53, 22, 54, 22, 54, 23, 55, 23, 55},
                         {24, 56, 24, 56, 25, 57, 25, 57, 26, 58, 26, 58, 27, 59, 27, 59},
                         {28, 60, 28, 60, 29, 61, 29, 61, 30, 62, 30, 62, 31, 63, 31, 63}},
                        {{0x0020, 0x0020, 0x0021, 0x0021, 0x0022, 0x0022, 0x0023, 0x0023},
                         {0x0024, 0x0024, 0x0025, 0x0025, 0x0026, 0x0026, 0x0027, 0x0027},
                         {0x0028, 0x0028, 0x0029, 0x0029, 0x002a, 0x002a, 0x002b, 0x002b},
                         {0x002c, 0x002c, 0x002d, 0x002d, 0x002e, 0x002e, 0x002f, 0x002f},
                         {0x0030, 0x0030, 0x0031, 0x0031, 0x0032, 0x0032, 0x0033, 0x0033},
                         {0x0034, 0x0034, 0x0035, 0x0035, 0x0036, 0x0036, 0x0037, 0x0037},
                         {0x0038, 0x0038, 0x0039, 0x0039, 0x003a, 0x003a, 0x003b, 0x003b},
                         {0x003c, 0x003c, 0x003d, 0x003d, 0x003e, 0x003e, 0x003f, 0x003f}},
                        {{0x0020'c0a0, 0x0021'c1a1, 0x0022'c2a2, 0x0023'c3a3},
                         {0x0024'c4a4, 0x0025'c5a5, 0x0026'c6a6, 0x0027'c7a7},
                         {0x0028'c8a8, 0x0029'c9a9, 0x002a'caaa, 0x002b'cbab},
                         {0x002c'ccac, 0x002d'cdad, 0x002e'ceae, 0x002f'cfaf},
                         {0x0030'd0b0, 0x0031'd1b1, 0x0032'd2b2, 0x0033'd3b3},
                         {0x0034'd4b4, 0x0035'd5b5, 0x0036'd6b6, 0x0037'd7b7},
                         {0x0038'd8b8, 0x0039'd9b9, 0x003a'daba, 0x003b'dbbb},
                         {0x003c'dcbc, 0x003d'ddbd, 0x003e'debe, 0x003f'dfbf}},
                        {{0x0021'c1a1'4120'c0a0, 0x0023'c3a3'4322'c2a2},
                         {0x0025'c5a5'4524'c4a4, 0x0027'c7a7'4726'c6a6},
                         {0x0029'c9a9'4928'c8a8, 0x002b'cbab'4b2a'caaa},
                         {0x002d'cdad'4d2c'ccac, 0x002f'cfaf'4f2e'ceae},
                         {0x0031'd1b1'5130'd0b0, 0x0033'd3b3'5332'd2b2},
                         {0x0035'd5b5'5534'd4b4, 0x0037'd7b7'5736'd6b6},
                         {0x0039'd9b9'5938'd8b8, 0x003b'dbbb'5b3a'daba},
                         {0x003d'ddbd'5d3c'dcbc, 0x003f'dfbf'5f3e'debe}},
                        kVectorCalculationsSource);
  TestVectorInstruction(ExecVssravv,
                        ExecMaskedVssravv,
                        {{0, 224, 0, 254, 2, 225, 0, 254, 4, 226, 0, 254, 12, 227, 0, 254},
                         {16, 228, 1, 254, 10, 229, 1, 254, 12, 230, 1, 254, 28, 231, 1, 254},
                         {32, 232, 2, 254, 18, 233, 2, 254, 20, 234, 2, 254, 44, 235, 2, 254},
                         {48, 236, 3, 254, 26, 237, 3, 254, 28, 238, 3, 254, 60, 239, 3, 254},
                         {64, 240, 4, 255, 34, 241, 4, 255, 36, 242, 4, 255, 76, 243, 4, 255},
                         {80, 244, 5, 255, 42, 245, 5, 255, 44, 246, 5, 255, 92, 247, 5, 255},
                         {96, 248, 6, 255, 50, 249, 6, 255, 52, 250, 6, 255, 108, 251, 6, 255},
                         {112, 252, 7, 255, 58, 253, 7, 255, 60, 254, 7, 255, 124, 255, 7, 255}},
                        {{0x8100, 0xf830, 0xffc2, 0xfff8, 0xc484, 0xf8b0, 0xff8d, 0xfff8},
                         {0x9110, 0xf931, 0xffca, 0xfff9, 0xcc8c, 0xf9b1, 0xff9d, 0xfff9},
                         {0xa120, 0xfa32, 0xffd2, 0xfffa, 0xd494, 0xfab2, 0xffad, 0xfffa},
                         {0xb130, 0xfb33, 0xffda, 0xfffb, 0xdc9c, 0xfbb3, 0xffbd, 0xfffb},
                         {0xc140, 0xfc34, 0xffe2, 0xfffc, 0xe4a4, 0xfcb4, 0xffcd, 0xfffc},
                         {0xd150, 0xfd35, 0xffea, 0xfffd, 0xecac, 0xfdb5, 0xffdd, 0xfffd},
                         {0xe160, 0xfe36, 0xfff2, 0xfffe, 0xf4b4, 0xfeb6, 0xffed, 0xfffe},
                         {0xf170, 0xff37, 0xfffa, 0xffff, 0xfcbc, 0xffb7, 0xfffd, 0xffff}},
                        {{0x8302'8100, 0xffc3'8342, 0xffff'c585, 0xffff'ff8f},
                         {0x9312'9110, 0xffcb'8b4a, 0xffff'cd8d, 0xffff'ff9f},
                         {0xa322'a120, 0xffd3'9352, 0xffff'd595, 0xffff'ffaf},
                         {0xb332'b130, 0xffdb'9b5a, 0xffff'dd9d, 0xffff'ffbf},
                         {0xc342'c140, 0xffe3'a362, 0xffff'e5a5, 0xffff'ffcf},
                         {0xd352'd150, 0xffeb'ab6a, 0xffff'edad, 0xffff'ffdf},
                         {0xe362'e160, 0xfff3'b372, 0xffff'f5b5, 0xffff'ffef},
                         {0xf372'f170, 0xfffb'bb7a, 0xffff'fdbd, 0xffff'ffff}},
                        {{0x8706'8504'8302'8100, 0xffff'c787'4686'4585},
                         {0xffff'ffff'9716'9514, 0xffff'ffff'ffff'cf8f},
                         {0xa726'a524'a322'a120, 0xffff'd797'5696'5595},
                         {0xffff'ffff'b736'b534, 0xffff'ffff'ffff'df9f},
                         {0xc746'c544'c342'c140, 0xffff'e7a7'66a6'65a5},
                         {0xffff'ffff'd756'd554, 0xffff'ffff'ffff'efaf},
                         {0xe766'e564'e362'e160, 0xffff'f7b7'76b6'75b5},
                         {0xffff'ffff'f776'f574, 0xffff'ffff'ffff'ffbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssravx,
                        ExecMaskedVssravx,
                        {{0, 224, 0, 224, 1, 225, 1, 225, 2, 226, 2, 226, 3, 227, 3, 227},
                         {4, 228, 4, 228, 5, 229, 5, 229, 6, 230, 6, 230, 7, 231, 7, 231},
                         {8, 232, 8, 232, 9, 233, 9, 233, 10, 234, 10, 234, 11, 235, 11, 235},
                         {12, 236, 12, 236, 13, 237, 13, 237, 14, 238, 14, 238, 15, 239, 15, 239},
                         {16, 240, 16, 240, 17, 241, 17, 241, 18, 242, 18, 242, 19, 243, 19, 243},
                         {20, 244, 20, 244, 21, 245, 21, 245, 22, 246, 22, 246, 23, 247, 23, 247},
                         {24, 248, 24, 248, 25, 249, 25, 249, 26, 250, 26, 250, 27, 251, 27, 251},
                         {28, 252, 28, 252, 29, 253, 29, 253, 30, 254, 30, 254, 31, 255, 31, 255}},
                        {{0xffe0, 0xffe0, 0xffe1, 0xffe1, 0xffe2, 0xffe2, 0xffe3, 0xffe3},
                         {0xffe4, 0xffe4, 0xffe5, 0xffe5, 0xffe6, 0xffe6, 0xffe7, 0xffe7},
                         {0xffe8, 0xffe8, 0xffe9, 0xffe9, 0xffea, 0xffea, 0xffeb, 0xffeb},
                         {0xffec, 0xffec, 0xffed, 0xffed, 0xffee, 0xffee, 0xffef, 0xffef},
                         {0xfff0, 0xfff0, 0xfff1, 0xfff1, 0xfff2, 0xfff2, 0xfff3, 0xfff3},
                         {0xfff4, 0xfff4, 0xfff5, 0xfff5, 0xfff6, 0xfff6, 0xfff7, 0xfff7},
                         {0xfff8, 0xfff8, 0xfff9, 0xfff9, 0xfffa, 0xfffa, 0xfffb, 0xfffb},
                         {0xfffc, 0xfffc, 0xfffd, 0xfffd, 0xfffe, 0xfffe, 0xffff, 0xffff}},
                        {{0xffe0'c0a0, 0xffe1'c1a1, 0xffe2'c2a2, 0xffe3'c3a3},
                         {0xffe4'c4a4, 0xffe5'c5a5, 0xffe6'c6a6, 0xffe7'c7a7},
                         {0xffe8'c8a8, 0xffe9'c9a9, 0xffea'caaa, 0xffeb'cbab},
                         {0xffec'ccac, 0xffed'cdad, 0xffee'ceae, 0xffef'cfaf},
                         {0xfff0'd0b0, 0xfff1'd1b1, 0xfff2'd2b2, 0xfff3'd3b3},
                         {0xfff4'd4b4, 0xfff5'd5b5, 0xfff6'd6b6, 0xfff7'd7b7},
                         {0xfff8'd8b8, 0xfff9'd9b9, 0xfffa'daba, 0xfffb'dbbb},
                         {0xfffc'dcbc, 0xfffd'ddbd, 0xfffe'debe, 0xffff'dfbf}},
                        {{0xffff'ffff'ffe1'c1a1, 0xffff'ffff'ffe3'c3a3},
                         {0xffff'ffff'ffe5'c5a5, 0xffff'ffff'ffe7'c7a7},
                         {0xffff'ffff'ffe9'c9a9, 0xffff'ffff'ffeb'cbab},
                         {0xffff'ffff'ffed'cdad, 0xffff'ffff'ffef'cfaf},
                         {0xffff'ffff'fff1'd1b1, 0xffff'ffff'fff3'd3b3},
                         {0xffff'ffff'fff5'd5b5, 0xffff'ffff'fff7'd7b7},
                         {0xffff'ffff'fff9'd9b9, 0xffff'ffff'fffb'dbbb},
                         {0xffff'ffff'fffd'ddbd, 0xffff'ffff'ffff'dfbf}},
                        kVectorCalculationsSource);

  TestVectorInstruction(ExecVssravi,
                        ExecMaskedVssravi,
                        {{0, 224, 0, 224, 1, 225, 1, 225, 2, 226, 2, 226, 3, 227, 3, 227},
                         {4, 228, 4, 228, 5, 229, 5, 229, 6, 230, 6, 230, 7, 231, 7, 231},
                         {8, 232, 8, 232, 9, 233, 9, 233, 10, 234, 10, 234, 11, 235, 11, 235},
                         {12, 236, 12, 236, 13, 237, 13, 237, 14, 238, 14, 238, 15, 239, 15, 239},
                         {16, 240, 16, 240, 17, 241, 17, 241, 18, 242, 18, 242, 19, 243, 19, 243},
                         {20, 244, 20, 244, 21, 245, 21, 245, 22, 246, 22, 246, 23, 247, 23, 247},
                         {24, 248, 24, 248, 25, 249, 25, 249, 26, 250, 26, 250, 27, 251, 27, 251},
                         {28, 252, 28, 252, 29, 253, 29, 253, 30, 254, 30, 254, 31, 255, 31, 255}},
                        {{0xffe0, 0xffe0, 0xffe1, 0xffe1, 0xffe2, 0xffe2, 0xffe3, 0xffe3},
                         {0xffe4, 0xffe4, 0xffe5, 0xffe5, 0xffe6, 0xffe6, 0xffe7, 0xffe7},
                         {0xffe8, 0xffe8, 0xffe9, 0xffe9, 0xffea, 0xffea, 0xffeb, 0xffeb},
                         {0xffec, 0xffec, 0xffed, 0xffed, 0xffee, 0xffee, 0xffef, 0xffef},
                         {0xfff0, 0xfff0, 0xfff1, 0xfff1, 0xfff2, 0xfff2, 0xfff3, 0xfff3},
                         {0xfff4, 0xfff4, 0xfff5, 0xfff5, 0xfff6, 0xfff6, 0xfff7, 0xfff7},
                         {0xfff8, 0xfff8, 0xfff9, 0xfff9, 0xfffa, 0xfffa, 0xfffb, 0xfffb},
                         {0xfffc, 0xfffc, 0xfffd, 0xfffd, 0xfffe, 0xfffe, 0xffff, 0xffff}},
                        {{0xffe0'c0a0, 0xffe1'c1a1, 0xffe2'c2a2, 0xffe3'c3a3},
                         {0xffe4'c4a4, 0xffe5'c5a5, 0xffe6'c6a6, 0xffe7'c7a7},
                         {0xffe8'c8a8, 0xffe9'c9a9, 0xffea'caaa, 0xffeb'cbab},
                         {0xffec'ccac, 0xffed'cdad, 0xffee'ceae, 0xffef'cfaf},
                         {0xfff0'd0b0, 0xfff1'd1b1, 0xfff2'd2b2, 0xfff3'd3b3},
                         {0xfff4'd4b4, 0xfff5'd5b5, 0xfff6'd6b6, 0xfff7'd7b7},
                         {0xfff8'd8b8, 0xfff9'd9b9, 0xfffa'daba, 0xfffb'dbbb},
                         {0xfffc'dcbc, 0xfffd'ddbd, 0xfffe'debe, 0xffff'dfbf}},
                        {{0xffe1'c1a1'4120'c0a0, 0xffe3'c3a3'4322'c2a2},
                         {0xffe5'c5a5'4524'c4a4, 0xffe7'c7a7'4726'c6a6},
                         {0xffe9'c9a9'4928'c8a8, 0xffeb'cbab'4b2a'caaa},
                         {0xffed'cdad'4d2c'ccac, 0xffef'cfaf'4f2e'ceae},
                         {0xfff1'd1b1'5130'd0b0, 0xfff3'd3b3'5332'd2b2},
                         {0xfff5'd5b5'5534'd4b4, 0xfff7'd7b7'5736'd6b6},
                         {0xfff9'd9b9'5938'd8b8, 0xfffb'dbbb'5b3a'daba},
                         {0xfffd'ddbd'5d3c'dcbc, 0xffff'dfbf'5f3e'debe}},
                        kVectorCalculationsSource);
  asm("csrw vxrm, %0\n\t" ::"r"(vxrm));
}
