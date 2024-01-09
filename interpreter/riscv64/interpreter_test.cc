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

#include <algorithm>  // copy_n, fill_n
#include <cstdint>

#include "berberis/base/bit_util.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/riscv64/interpreter.h"
#include "berberis/intrinsics/guest_fp_flags.h"        // GuestModeFromHostRounding
#include "berberis/intrinsics/guest_rounding_modes.h"  // ScopedRoundingMode
#include "berberis/intrinsics/simd_register.h"
#include "berberis/intrinsics/vector_intrinsics.h"
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
  Riscv64InterpreterTest()
      : state_{
            .cpu = {.vtype = uint64_t{1} << 63, .frm = intrinsics::GuestModeFromHostRounding()}} {}

  void InterpretFence(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    InterpretInsn(&state_);
  }

  // Vector instructions.
  template <size_t kNFfields>
  void TestVlₓreₓₓ(uint32_t insn_bytes) {
    const auto kUndisturbedValue = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&kVectorComparisonSource));
    for (size_t index = 0; index < 8; index++) {
      state_.cpu.v[8 + index] = kUndisturbedValue;
    }
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    for (size_t index = 0; index < 8; index++) {
      EXPECT_EQ(state_.cpu.v[8 + index],
                (index >= kNFfields
                     ? kUndisturbedValue
                     : SIMD128Register{kVectorComparisonSource[index]}.Get<__uint128_t>()));
    }
  }

  template <size_t kNFfields>
  void TestVsₓ(uint32_t insn_bytes) {
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    for (size_t index = 0; index < 8; index++) {
      state_.cpu.v[8 + index] = SIMD128Register{kVectorComparisonSource[index]}.Get<__uint128_t>();
      store_area_[index * 2] = kUndisturbedResult[0];
      store_area_[index * 2 + 1] = kUndisturbedResult[1];
    }
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    for (size_t index = 0; index < 8; index++) {
      EXPECT_EQ(
          store_area_[index * 2],
          (index >= kNFfields ? kUndisturbedResult[0]
                              : SIMD128Register{kVectorComparisonSource[index]}.Get<uint64_t>(0)));
      EXPECT_EQ(
          store_area_[index * 2 + 1],
          (index >= kNFfields ? kUndisturbedResult[1]
                              : SIMD128Register{kVectorComparisonSource[index]}.Get<uint64_t>(1)));
    }
  }

  void TestVectorInstruction(uint32_t insn_bytes,
                             const __v16qu (&expected_result_int8)[8],
                             const __v8hu (&expected_result_int16)[8],
                             const __v4su (&expected_result_int32)[8],
                             const __v2du (&expected_result_int64)[8],
                             const __v2du (&source)[16],
                             // Used for Vmerge, which sets masked off elements to vs2.
                             bool expect_inactive_equals_vs2 = false) {
    auto Verify = [this, &source, expect_inactive_equals_vs2](uint32_t insn_bytes,
                                                              uint8_t vsew,
                                                              uint8_t vlmul_max,
                                                              const auto& expected_result,
                                                              auto mask) {
      // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
      // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (size_t index = 0; index < std::size(source); ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }
      // Set x1 for vx instructions.
      SetXReg<1>(state_.cpu, 0xaaaa'aaaa'aaaa'aaaa);
      for (uint8_t vlmul = 0; vlmul < vlmul_max; ++vlmul) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          for (uint8_t vma = 0; vma < 2; ++vma) {
            auto [vlmax, vtype] =
                intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
            // Incompatible vsew and vlmax. Skip it.
            if (vlmax == 0) {
              continue;
            }

            // To make tests quick enough we don't test vstart and vl change with small register
            // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of first
            // register and half of last register.
            // Don't use vlmul == 3 because that one may not be supported if instruction widens the
            // result.
            if (vlmul == 2) {
              state_.cpu.vstart = vlmax / 8;
              state_.cpu.vl = (vlmax * 5) / 8;
            } else {
              state_.cpu.vstart = 0;
              state_.cpu.vl = vlmax;
            }
            state_.cpu.vtype = vtype;

            // Set expected_result vector registers into 0b01010101… pattern.
            for (size_t index = 0; index < 8; ++index) {
              state_.cpu.v[8 + index] = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
            }

            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

            // Values for inactive elements (i.e. corresponding mask bit is 0).
            const size_t n = std::size(source);
            __m128i expected_inactive[n];
            if (expect_inactive_equals_vs2) {
              // vs2 is the start of the source vector register group.
              // Note: copy_n input/output args are backwards compared to fill_n below.
              std::copy_n(source, n, expected_inactive);
            } else {
              // For most instructions, follow basic inactive processing rules based on vma flag.
              std::fill_n(expected_inactive, n, (vma ? kAgnosticResult : kUndisturbedResult));
            }

            if (vlmul < 4) {
              for (size_t index = 0; index < 1 << vlmul; ++index) {
                if (index == 0 && vlmul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            SIMD128Register{
                                (kUndisturbedResult & kFractionMaskInt8[3]) |
                                (expected_result[index] & mask[index] & ~kFractionMaskInt8[3]) |
                                (expected_inactive[index] & ~mask[index] & ~kFractionMaskInt8[3])}
                                .Get<__uint128_t>());
                } else if (index == 2 && vlmul == 2) {
                  EXPECT_EQ(
                      state_.cpu.v[8 + index],
                      SIMD128Register{
                          (expected_result[index] & mask[index] & kFractionMaskInt8[3]) |
                          (expected_inactive[index] & ~mask[index] & kFractionMaskInt8[3]) |
                          ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[3])}
                          .Get<__uint128_t>());
                } else if (index == 3 && vlmul == 2 && vta) {
                  EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kAgnosticResult});
                } else if (index == 3 && vlmul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kUndisturbedResult});
                } else {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            SIMD128Register{(expected_result[index] & mask[index]) |
                                            (expected_inactive[index] & ~mask[index])}
                                .Get<__uint128_t>());
                }
              }
            } else {
              EXPECT_EQ(
                  state_.cpu.v[8],
                  SIMD128Register{(expected_result[0] & mask[0] & kFractionMaskInt8[vlmul - 4]) |
                                  (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[vlmul - 4]) |
                                  ((vta ? kAgnosticResult : kUndisturbedResult) &
                                   ~kFractionMaskInt8[vlmul - 4])}
                      .Get<__uint128_t>());
            }

            if (vlmul == 2) {
              // Every vector instruction must set vstart to 0, but shouldn't touch vl.
              EXPECT_EQ(state_.cpu.vstart, 0);
              EXPECT_EQ(state_.cpu.vl, (vlmax * 5) / 8);
            }
          }
        }
      }
    };

    // Some instructions don't support use of mask register, but in these instructions bit
    // #25 is set.  Test it and skip masking tests if so.
    if ((insn_bytes & (1 << 25)) == 0) {
      Verify(insn_bytes, 0, 8, expected_result_int8, kMaskInt8);
      Verify(insn_bytes, 1, 8, expected_result_int16, kMaskInt16);
      Verify(insn_bytes, 2, 8, expected_result_int32, kMaskInt32);
      Verify(insn_bytes, 3, 8, expected_result_int64, kMaskInt64);
      Verify(insn_bytes | (1 << 25), 0, 8, expected_result_int8, kNoMask);
      Verify(insn_bytes | (1 << 25), 1, 8, expected_result_int16, kNoMask);
      Verify(insn_bytes | (1 << 25), 2, 8, expected_result_int32, kNoMask);
      Verify(insn_bytes | (1 << 25), 3, 8, expected_result_int64, kNoMask);
    } else {
      Verify(insn_bytes, 0, 1, expected_result_int8, kNoMask);
      Verify(insn_bytes, 1, 1, expected_result_int16, kNoMask);
      Verify(insn_bytes, 2, 1, expected_result_int32, kNoMask);
      Verify(insn_bytes, 3, 1, expected_result_int64, kNoMask);
    }
  }

  void TestVectorMaskInstruction(uint32_t insn_bytes, const __v2du expected_result) {
    // Mask instructions don't look on vtype directly, but they still require valid one because it
    // affects vlmax;
    const __uint128_t undisturbed = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
    const __uint128_t src1 = SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>();
    const __uint128_t src2 = SIMD128Register{kVectorCalculationsSource[8]}.Get<__uint128_t>();
    const __uint128_t expected = SIMD128Register{expected_result}.Get<__uint128_t>();
    auto [vlmax, vtype] = intrinsics::Vsetvl(~0ULL, 3);
    state_.cpu.vtype = vtype;
    for (uint8_t vl = 0; vl <= vlmax; ++vl) {
      state_.cpu.vl = vl;
      for (uint8_t vstart = 0; vstart <= 128; ++vstart) {
        state_.cpu.vstart = vstart;
        // Set expected_result vector registers into 0b01010101… pattern.
        state_.cpu.v[8] = undisturbed;
        state_.cpu.v[16] = src1;
        state_.cpu.v[24] = src2;

        state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
        EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

        for (uint8_t bit_pos = 0; bit_pos < 128; ++bit_pos) {
          __uint128_t bit = __uint128_t{1} << bit_pos;
          if (bit_pos >= vl) {
            EXPECT_EQ(state_.cpu.v[8] & bit, bit);
          } else if (bit_pos < vstart) {
            EXPECT_EQ(state_.cpu.v[8] & bit, undisturbed & bit);
          } else {
            EXPECT_EQ(state_.cpu.v[8] & bit, expected & bit);
          }
        }
      }
    }
  }

  void TestVₓmₓsInstruction(uint32_t insn_bytes,
                            const uint64_t (&expected_result_no_mask)[129],
                            const uint64_t (&expected_result_with_mask)[129],
                            const __v2du source) {
    auto Verify = [this, &source](uint32_t insn_bytes,
                                  const uint64_t (&expected_result)[129]) {
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();

      auto [vlmax, vtype] = intrinsics::Vsetvl(~0ULL, 3);
      state_.cpu.vtype = vtype;
      state_.cpu.vstart = 0;
      state_.cpu.v[16] = SIMD128Register{source}.Get<__uint128_t>();

      for (uint8_t vl = 0; vl <= vlmax; ++vl) {
        state_.cpu.vl = vl;
        SetXReg<1>(state_.cpu, 0xaaaa'aaaa'aaaa'aaaa);

        state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
        EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
        EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result[vl]) << std::to_string(vl);
      }
    };

    Verify(insn_bytes, expected_result_with_mask);
    Verify(insn_bytes | (1 << 25), expected_result_no_mask);
  }

  void TestVectorReductionInstruction(uint32_t insn_bytes,
                                      const uint8_t (&expected_result_vd0_int8)[8],
                                      const uint16_t (&expected_result_vd0_int16)[8],
                                      const uint32_t (&expected_result_vd0_int32)[8],
                                      const uint64_t (&expected_result_vd0_int64)[8],
                                      const uint8_t (&expected_result_vd0_with_mask_int8)[8],
                                      const uint16_t (&expected_result_vd0_with_mask_int16)[8],
                                      const uint32_t (&expected_result_vd0_with_mask_int32)[8],
                                      const uint64_t (&expected_result_vd0_with_mask_int64)[8],
                                      const __v2du (&source)[16]) {
    // Each expected_result input to this function is the vd[0] value of the reduction, for each
    // of the possible vlmul, i.e. expected_result_vd0_int8[n] = vd[0], int8, no mask, vlmul=n.
    //
    // As vlmul=4 is reserved, expected_result_vd0_*[4] is ignored.
    auto Verify = [this, &source](uint32_t insn_bytes,
                                  uint8_t vsew,
                                  uint8_t vlmul,
                                  const auto& expected_result) {
      // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
      // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (size_t index = 0; index < std::size(source); ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }
      for (uint8_t vta = 0; vta < 2; ++vta) {
        for (uint8_t vma = 0; vma < 2; ++vma) {
          auto [vlmax, vtype] =
              intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
          // Incompatible vsew and vlmax. Skip it.
          if (vlmax == 0) {
            continue;
          }

          // Vector reduction instructions must always have a vstart=0.
          state_.cpu.vstart = 0;
          state_.cpu.vl = vlmax;
          state_.cpu.vtype = vtype;

          // Set expected_result vector registers into 0b01010101… pattern.
          for (size_t index = 0; index < 8; ++index) {
            state_.cpu.v[8 + index] = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
          }

          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

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
          size_t vsew_bits = 8 << vsew;
          __uint128_t expected_result_register =
            SIMD128Register{vta ? kAgnosticResult : kUndisturbedResult}.Get<__uint128_t>();
          expected_result_register = (expected_result_register >> vsew_bits) << vsew_bits;
          expected_result_register |= expected_result;
          EXPECT_EQ(state_.cpu.v[8], expected_result_register);

          // Verify all non-destination registers are undisturbed.
          for (size_t index = 1; index < 8; ++index) {
            EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kUndisturbedResult}.Get<__uint128_t>());
          }

          // Every vector instruction must set vstart to 0, but shouldn't touch vl.
          EXPECT_EQ(state_.cpu.vstart, 0);
          EXPECT_EQ(state_.cpu.vl, vlmax);
        }
      }
    };

    for (int vlmul = 0; vlmul < 8; vlmul++) {
      Verify(insn_bytes, 0, vlmul, expected_result_vd0_with_mask_int8[vlmul]);
      Verify(insn_bytes, 1, vlmul, expected_result_vd0_with_mask_int16[vlmul]);
      Verify(insn_bytes, 2, vlmul, expected_result_vd0_with_mask_int32[vlmul]);
      Verify(insn_bytes, 3, vlmul, expected_result_vd0_with_mask_int64[vlmul]);
      Verify(insn_bytes | (1 << 25), 0, vlmul, expected_result_vd0_int8[vlmul]);
      Verify(insn_bytes | (1 << 25), 1, vlmul, expected_result_vd0_int16[vlmul]);
      Verify(insn_bytes | (1 << 25), 2, vlmul, expected_result_vd0_int32[vlmul]);
      Verify(insn_bytes | (1 << 25), 3, vlmul, expected_result_vd0_int64[vlmul]);
    }
  }

 protected:
  static constexpr __v2du kVectorCalculationsSource[16] = {
      {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
      {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
      {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
      {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
      {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
      {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
      {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
      {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978},

      {0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
      {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
      {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
      {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
      {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
      {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
      {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
      {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1},
  };

  static constexpr __v2du kVectorComparisonSource[16] = {
      {0xfff5'fff5'fff5'fff5, 0xfff5'fff5'fff5'fff5},
      {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
      {0xbbbb'bbbb'bbbb'bbbb, 0xaaaa'aaaa'aaaa'aaaa},
      {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
      {0xfff4'fff4'fff4'fff4, 0xfff6'fff6'fff6'fff6},
      {0xfff8'fff8'fff4'fff4, 0xfff5'fff5'fff5'fff5},
      {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
      {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9},

      {0xfff5'fff5'fff5'fff5, 0xfff5'fff5'fff5'fff5},
      {0x1111'1111'1111'1111, 0x1111'1111'1111'1111},
      {0xfff1'fff1'fff1'fff1, 0xfff1'fff1'fff1'fff1},
      {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
      {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
      {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
      {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
      {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1},
  };

  // Right shift tests should use inputs with 1s in the most significant bit to differentiate
  // between logical and arithmetic right shifts.
  static constexpr __v2du kVectorRightShiftSource[16] = {
      {0xfff5'fff5'fff5'fff5, 0xfff5'fff5'fff5'fff5},
      {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
      {0xbbbb'bbbb'bbbb'bbbb, 0xaaaa'aaaa'aaaa'aaaa},
      {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
      {0xfff4'fff4'fff4'fff4, 0xfff6'fff6'fff6'fff6},
      {0xfff8'fff8'fff4'fff4, 0xfff5'fff5'fff5'fff5},
      {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
      {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9},

      {0xfff5'fff5'fff5'fff5, 0xfff5'fff5'fff5'fff5},
      {0x1111'1111'1111'1111, 0x1111'1111'1111'1111},
      {0xfff1'fff1'fff1'fff1, 0xfff1'fff1'fff1'fff1},
      {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
      {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
      {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
      {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
      {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1},
  };

  // Mask in form suitable for storing in v0 and use in v0.t form.
  static constexpr __v2du kMask = {0xd5ad'd6b5'ad6b'b5ad, 0x6af7'57bb'deed'7bb5};
  // Mask used with vsew = 0 (8bit) elements.
  static constexpr __v16qu kMaskInt8[8] = {
      {255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255},
      {255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255},
      {255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255},
      {255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 0, 255, 255},
      {255, 0, 255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0},
      {255, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 255},
      {255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0},
      {255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 0},
  };
  // Mask used with vsew = 1 (16bit) elements.
  static constexpr __v8hu kMaskInt16[8] = {
      {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
      {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff},
      {0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000},
      {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
      {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff},
      {0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff},
      {0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff},
      {0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff},
  };
  // Mask used with vsew = 2 (32bit) elements.
  static constexpr __v4su kMaskInt32[8] = {
      {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
      {0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
      {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0x0000'0000},
      {0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
      {0xffff'ffff, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
      {0x0000'0000, 0xffff'ffff, 0xffff'ffff, 0x0000'0000},
      {0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
      {0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff},
  };
  // Mask used with vsew = 3 (64bit) elements.
  static constexpr __v2du kMaskInt64[8] = {
      {0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000},
      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff},
      {0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000},
      {0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000},
      {0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
      {0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff},
  };
  // To verify operations without masking.
  static constexpr __v16qu kNoMask[8] = {
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
  };
  // Half of sub-register lmul.
  static constexpr __v16qu kFractionMaskInt8[4] = {
      {255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},        // Half of ⅛ reg = ¹⁄₁₆
      {255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},      // Half of ¼ reg = ⅛
      {255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // Half of ½ reg = ¼
      {255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0},  // Half of full reg = ½
  };
  // Agnostic result is -1 on RISC-V, not 0.
  static constexpr __m128i kAgnosticResult = {-1, -1};
  // Undisturbed result is put in registers v8, v9, …, v15 and is expected to get read back.
  static constexpr __m128i kUndisturbedResult = {0x5555'5555'5555'5555, 0x5555'5555'5555'5555};

  // Store area for store instructions.  We need at least 16 uint64_t to handle 8×128bit registers,
  // plus 2× of that to test strided instructions.
  alignas(16) uint64_t store_area_[32];

  ThreadState state_;
};

#define TESTSUITE Riscv64InterpretInsnTest
#define TESTING_INTERPRETER

#include "berberis/test_utils/insn_tests_riscv64-inl.h"

#undef TESTING_INTERPRETER
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

TEST_F(Riscv64InterpreterTest, TestVlₓreₓₓ) {
  TestVlₓreₓₓ<1>(0x2808407);   // vl1re8.v v8, (x1)
  TestVlₓreₓₓ<2>(0x22808407);  // vl2re8.v v8, (x1)
  TestVlₓreₓₓ<4>(0x62808407);  // vl4re8.v v8, (x1)
  TestVlₓreₓₓ<8>(0xe2808407);  // vl8re8.v v8, (x1)

  TestVlₓreₓₓ<1>(0x280d407);   // vl1re16.v v8, (x1)
  TestVlₓreₓₓ<2>(0x2280d407);  // vl2re16.v v8, (x1)
  TestVlₓreₓₓ<4>(0x6280d407);  // vl4re16.v v8, (x1)
  TestVlₓreₓₓ<8>(0xe280d407);  // vl8re16.v v8, (x1)

  TestVlₓreₓₓ<1>(0x280e407);   // vl1re32.v v8, (x1)
  TestVlₓreₓₓ<2>(0x2280e407);  // vl2re32.v v8, (x1)
  TestVlₓreₓₓ<4>(0x6280e407);  // vl4re32.v v8, (x1)
  TestVlₓreₓₓ<8>(0xe280e407);  // vl8re32.v v8, (x1)

  TestVlₓreₓₓ<1>(0x280f407);   // vl1re64.v v8, (x1)
  TestVlₓreₓₓ<2>(0x2280f407);  // vl2re64.v v8, (x1)
  TestVlₓreₓₓ<4>(0x6280f407);  // vl4re64.v v8, (x1)
  TestVlₓreₓₓ<8>(0xe280f407);  // vl8re64.v v8, (x1)
}

TEST_F(Riscv64InterpreterTest, TestVsₓ) {
  TestVsₓ<1>(0x2808427);   // vs1r.v v8, (x1)
  TestVsₓ<2>(0x22808427);  // vs2r.v v8, (x1)
  TestVsₓ<4>(0x62808427);  // vs4r.v v8, (x1)
  TestVsₓ<8>(0xe2808427);  // vs8r.v v8, (x1)
}

TEST_F(Riscv64InterpreterTest, TestVadd) {
  TestVectorInstruction(
      0x10c0457,  // Vadd.vv v8, v16, v24, v0.t
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
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x100c457,  // Vadd.vx v8, v16, x1, v0.t
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
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x10ab457,  // Vadd.vi v8, v16, -0xb, v0.t
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
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVectorMaskInstructions) {
  TestVectorMaskInstruction(0x630c2457,  // vmandn.mm v8, v16, v24
                            {0x8102'8504'8102'8100, 0x8102'8504'890a'8908});
  TestVectorMaskInstruction(0x670c2457,  // vmand.mm v8, v16, v24
                            {0x0604'0000'0200'0000, 0x0e0c'0808'0200'0000});
  TestVectorMaskInstruction(0x6b0c2457,  // vmor.mm v8, v16, v24
                            {0x8f0e'8f0d'8706'8300, 0x9f1e'9f1c'9f1e'9b19});
  TestVectorMaskInstruction(0x6f0c2457,  // vmxor.mm v8, v16, v24
                            {0x890a'8f0d'8506'8300, 0x9112'9714'9d1e'9b19});
  TestVectorMaskInstruction(0x730c2457,  // vmorn.mm v8, v16, v24
                            {0xf7f7'f5f6'fbfb'fdff, 0xefef'edef'ebeb'edee});
  TestVectorMaskInstruction(0x770c2457,  // vmnand.mm v8, v16, v24
                            {0xf9fb'ffff'fdff'ffff, 0xf1f3'f7f7'fdff'ffff});
  TestVectorMaskInstruction(0x7b0c2457,  // vmnor.mm v8, v16, v24
                            {0x70f1'70f2'78f9'7cff, 0x60e1'60e3'60e1'64e6});
  TestVectorMaskInstruction(0x7f0c2457,  // vmxnor.mm v8, v16, v24
                            {0x76f5'70f2'7af9'7cff, 0x6eed'68eb'62e1'64e6});
}

TEST_F(Riscv64InterpreterTest, TestVrsub) {
  TestVectorInstruction(
      0xd00c457,  // Vrsub.vi v8, v16, x1, v0.t
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
      kVectorCalculationsSource);
  TestVectorInstruction(
      0xd0ab457,  // Vrsub.vi v8, v16, -0xb, v0.t
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
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVsub) {
  TestVectorInstruction(
      0x90c0457,  // Vsub.vv v8, v16, v24, v0.t
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
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x900c457,  // Vsub.vx v8, v16, x1, v0.t
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
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVand) {
  TestVectorInstruction(
      0x250c0457,  // Vand.vv v8, v16, v24, v0.t
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
      kVectorCalculationsSource);
  TestVectorInstruction(0x2500c457,  // Vand.vx v8, v16, x1, v0.t
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
                        kVectorCalculationsSource);
  TestVectorInstruction(
      0x250ab457,  // Vand.vi v8, v16, -0xb, v0.t
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
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVor) {
  TestVectorInstruction(
      0x290c0457,  // Vor.vv v8, v16, v24, v0.t
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
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x2900c457,  // Vor.vx v8, v16, x1, v0.t
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
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x290ab457,  // Vor.vi v8, v16, -0xb, v0.t
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
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVxor) {
  TestVectorInstruction(
      0x2d0c0457,  // Vxor.vv v8, v16, v24, v0.t
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
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x2d00c457,  // Vxor.vx v8, v16, x1, v0.t
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
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x2d0ab457,  // Vxor.vi v8, v16, -0xb, v0.t
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
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVmseq) {
  TestVectorInstruction(0x610c0457,  // Vmseq.vv v8, v16, v24, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x6100c457,  // Vmseq.vx v8, v16, x1, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x610ab457,  // Vmseq.vi  v8, v16, -0xb, v0.t
                        {{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsne) {
  TestVectorInstruction(0x650c0457,  // Vmsne.vv v8, v16, v24, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x6500c457,  // Vmsne.vx v8, v16, x1, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x650ab457,  // Vmsne.vi  v8, v16, -0xb, v0.t
                        {{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsltu) {
  TestVectorInstruction(0x690c0457,  // Vmsltu.vv v8, v16, v24, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x6900c457,  // Vmsltu.vx v8, v16, x1, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
}
TEST_F(Riscv64InterpreterTest, TestVmslt) {
  TestVectorInstruction(0x6d0c0457,  // vmslt.vv v8, v16, v24, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x6d00c457,  // Vmslt.vx v8, v16, x1, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
}
TEST_F(Riscv64InterpreterTest, TestVmsleu) {
  TestVectorInstruction(0x710c0457,  // Vmsleu.vv v8, v16, v24, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x7100c457,  // Vmsleu.vx v8, v16, x1, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x710ab457,  // Vmsleu.vi  v8, v16, -0xb, v0.t
                        {{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
}
TEST_F(Riscv64InterpreterTest, TestVmsle) {
  TestVectorInstruction(0x750c0457,  // Vmsle.vv v8, v16, v24, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x7500c457,  // Vmsle.vx v8, v16, x1, v0.t
                        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x750ab457,  // Vmsle.vi  v8, v16, -0xb, v0.t
                        {{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001}},
                        kVectorComparisonSource);
}
TEST_F(Riscv64InterpreterTest, TestVmsgtu) {
  TestVectorInstruction(0x7900c457,  // Vmsgtu.vx v8, v16, x1, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x790ab457,  // Vmsgtu.vi  v8, v16, -0xb, v0.t
                        {{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
}
TEST_F(Riscv64InterpreterTest, TestVmsgt) {
  TestVectorInstruction(0x7d00c457,  // Vmsgt.vx v8, v16, x1, v0.t
                        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0001, 0x0000'0001, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0001, 0x0000'0001, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0001, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
  TestVectorInstruction(0x7d0ab457,  // Vmsgt.vi  v8, v16, -0xb, v0.t
                        {{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                        {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0001, 0x0001},
                         {0x0000, 0x0000, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
                         {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
                        {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0001, 0x0000'0001},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                         {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
                        {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0001},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
                         {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
                        kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVsll) {
  TestVectorInstruction(
      0x950c0457,  // Vsll.vv v8, v16, v24, v0.t
      {{0, 4, 32, 192, 8, 20, 96, 192, 16, 36, 160, 192, 12, 52, 224, 192},
       {16, 68, 32, 192, 40, 84, 96, 192, 48, 100, 160, 192, 28, 116, 224, 192},
       {32, 132, 32, 192, 72, 148, 96, 192, 80, 164, 160, 192, 44, 180, 224, 192},
       {48, 196, 32, 192, 104, 212, 96, 192, 112, 228, 160, 192, 60, 244, 224, 192},
       {64, 4, 32, 192, 136, 20, 96, 192, 144, 36, 160, 192, 76, 52, 224, 192},
       {80, 68, 32, 192, 168, 84, 96, 192, 176, 100, 160, 192, 92, 116, 224, 192},
       {96, 132, 32, 192, 200, 148, 96, 192, 208, 164, 160, 192, 108, 180, 224, 192},
       {112, 196, 32, 192, 232, 212, 96, 192, 240, 228, 160, 192, 124, 244, 224, 192}},
      {{0x8100, 0x3020, 0x0800, 0x6000, 0x1210, 0xb0a0, 0x0c00, 0xe000},
       {0x9110, 0x3120, 0x2800, 0x6000, 0x3230, 0xb1a0, 0x1c00, 0xe000},
       {0xa120, 0x3220, 0x4800, 0x6000, 0x5250, 0xb2a0, 0x2c00, 0xe000},
       {0xb130, 0x3320, 0x6800, 0x6000, 0x7270, 0xb3a0, 0x3c00, 0xe000},
       {0xc140, 0x3420, 0x8800, 0x6000, 0x9290, 0xb4a0, 0x4c00, 0xe000},
       {0xd150, 0x3520, 0xa800, 0x6000, 0xb2b0, 0xb5a0, 0x5c00, 0xe000},
       {0xe160, 0x3620, 0xc800, 0x6000, 0xd2d0, 0xb6a0, 0x6c00, 0xe000},
       {0xf170, 0x3720, 0xe800, 0x6000, 0xf2f0, 0xb7a0, 0x7c00, 0xe000}},
      {{0x8302'8100, 0x0d0a'0800, 0x1210'0000, 0x0c00'0000},
       {0x9312'9110, 0x2d2a'2800, 0x3230'0000, 0x1c00'0000},
       {0xa322'a120, 0x4d4a'4800, 0x5250'0000, 0x2c00'0000},
       {0xb332'b130, 0x6d6a'6800, 0x7270'0000, 0x3c00'0000},
       {0xc342'c140, 0x8d8a'8800, 0x9290'0000, 0x4c00'0000},
       {0xd352'd150, 0xadaa'a800, 0xb2b0'0000, 0x5c00'0000},
       {0xe362'e160, 0xcdca'c800, 0xd2d0'0000, 0x6c00'0000},
       {0xf372'f170, 0xedea'e800, 0xf2f0'0000, 0x7c00'0000}},
      {{0x8706'8504'8302'8100, 0x1a19'1615'1210'0000},
       {0x9312'9110'0000'0000, 0x3230'0000'0000'0000},
       {0xa726'a524'a322'a120, 0x5a59'5655'5250'0000},
       {0xb332'b130'0000'0000, 0x7270'0000'0000'0000},
       {0xc746'c544'c342'c140, 0x9a99'9695'9290'0000},
       {0xd352'd150'0000'0000, 0xb2b0'0000'0000'0000},
       {0xe766'e564'e362'e160, 0xdad9'd6d5'd2d0'0000},
       {0xf372'f170'0000'0000, 0xf2f0'0000'0000'0000}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x9500c457,  // Vsll.vx v8, v16, x1, v0.t
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188},
       {192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252},
       {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188},
       {192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252}},
      {{0x0000, 0x0800, 0x1000, 0x1800, 0x2000, 0x2800, 0x3000, 0x3800},
       {0x4000, 0x4800, 0x5000, 0x5800, 0x6000, 0x6800, 0x7000, 0x7800},
       {0x8000, 0x8800, 0x9000, 0x9800, 0xa000, 0xa800, 0xb000, 0xb800},
       {0xc000, 0xc800, 0xd000, 0xd800, 0xe000, 0xe800, 0xf000, 0xf800},
       {0x0000, 0x0800, 0x1000, 0x1800, 0x2000, 0x2800, 0x3000, 0x3800},
       {0x4000, 0x4800, 0x5000, 0x5800, 0x6000, 0x6800, 0x7000, 0x7800},
       {0x8000, 0x8800, 0x9000, 0x9800, 0xa000, 0xa800, 0xb000, 0xb800},
       {0xc000, 0xc800, 0xd000, 0xd800, 0xe000, 0xe800, 0xf000, 0xf800}},
      {{0x0a04'0000, 0x1a14'1000, 0x2a24'2000, 0x3a34'3000},
       {0x4a44'4000, 0x5a54'5000, 0x6a64'6000, 0x7a74'7000},
       {0x8a84'8000, 0x9a94'9000, 0xaaa4'a000, 0xbab4'b000},
       {0xcac4'c000, 0xdad4'd000, 0xeae4'e000, 0xfaf4'f000},
       {0x0b05'0000, 0x1b15'1000, 0x2b25'2000, 0x3b35'3000},
       {0x4b45'4000, 0x5b55'5000, 0x6b65'6000, 0x7b75'7000},
       {0x8b85'8000, 0x9b95'9000, 0xaba5'a000, 0xbbb5'b000},
       {0xcbc5'c000, 0xdbd5'd000, 0xebe5'e000, 0xfbf5'f000}},
      {{0x0a04'0000'0000'0000, 0x2a24'2000'0000'0000},
       {0x4a44'4000'0000'0000, 0x6a64'6000'0000'0000},
       {0x8a84'8000'0000'0000, 0xaaa4'a000'0000'0000},
       {0xcac4'c000'0000'0000, 0xeae4'e000'0000'0000},
       {0x0b05'0000'0000'0000, 0x2b25'2000'0000'0000},
       {0x4b45'4000'0000'0000, 0x6b65'6000'0000'0000},
       {0x8b85'8000'0000'0000, 0xaba5'a000'0000'0000},
       {0xcbc5'c000'0000'0000, 0xebe5'e000'0000'0000}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x9505b457,  // Vsll.vi v8, v16, 0xb, v0.t
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248},
       {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248},
       {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248},
       {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248}},
      {{0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000},
       {0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000},
       {0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000},
       {0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000},
       {0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000, 0xf000}},
      {{0x1408'0000, 0x3428'2000, 0x5448'4000, 0x7468'6000},
       {0x9488'8000, 0xb4a8'a000, 0xd4c8'c000, 0xf4e8'e000},
       {0x1509'0000, 0x3529'2000, 0x5549'4000, 0x7569'6000},
       {0x9589'8000, 0xb5a9'a000, 0xd5c9'c000, 0xf5e9'e000},
       {0x160a'0000, 0x362a'2000, 0x564a'4000, 0x766a'6000},
       {0x968a'8000, 0xb6aa'a000, 0xd6ca'c000, 0xf6ea'e000},
       {0x170b'0000, 0x372b'2000, 0x574b'4000, 0x776b'6000},
       {0x978b'8000, 0xb7ab'a000, 0xd7cb'c000, 0xf7eb'e000}},
      {{0x3428'2418'1408'0000, 0x7468'6458'5448'4000},
       {0xb4a8'a498'9488'8000, 0xf4e8'e4d8'd4c8'c000},
       {0x3529'2519'1509'0000, 0x7569'6559'5549'4000},
       {0xb5a9'a599'9589'8000, 0xf5e9'e5d9'd5c9'c000},
       {0x362a'261a'160a'0000, 0x766a'665a'564a'4000},
       {0xb6aa'a69a'968a'8000, 0xf6ea'e6da'd6ca'c000},
       {0x372b'271b'170b'0000, 0x776b'675b'574b'4000},
       {0xb7ab'a79b'978b'8000, 0xf7eb'e7db'd7cb'c000}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVsrl) {
  TestVectorInstruction(0xa10c0457,  // Vsrl.vv v8, v16, v24, v0.t
                        {{7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1},
                         {85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85},
                         {93, 1, 93, 1, 93, 1, 93, 1, 85, 1, 85, 1, 85, 1, 85, 1},
                         {170, 42, 10, 2, 85, 42, 10, 2, 8, 4, 1, 0, 17, 4, 1, 0},
                         {244, 63, 15, 3, 122, 63, 15, 3, 123, 63, 15, 3, 246, 63, 15, 3},
                         {244, 63, 15, 3, 124, 63, 15, 3, 122, 63, 15, 3, 245, 63, 15, 3},
                         {187, 46, 11, 2, 93, 46, 11, 2, 93, 46, 11, 2, 187, 46, 11, 2},
                         {169, 42, 10, 2, 84, 42, 10, 2, 84, 42, 10, 2, 169, 42, 10, 2}},
                        {{0x07ff, 0x07ff, 0x07ff, 0x07ff, 0x07ff, 0x07ff, 0x07ff, 0x07ff},
                         {0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555},
                         {0x5ddd, 0x5ddd, 0x5ddd, 0x5ddd, 0x5555, 0x5555, 0x5555, 0x5555},
                         {0xaaaa, 0x0aaa, 0x0055, 0x000a, 0x0888, 0x0111, 0x0011, 0x0001},
                         {0xfff4, 0x0fff, 0x007f, 0x000f, 0x7ffb, 0x0fff, 0x00ff, 0x000f},
                         {0xfff4, 0x0fff, 0x007f, 0x000f, 0x7ffa, 0x0fff, 0x00ff, 0x000f},
                         {0xbbbb, 0x0a9b, 0x005d, 0x000a, 0x5ddd, 0x0a9b, 0x00bb, 0x000a},
                         {0xa9a9, 0x0a9a, 0x0054, 0x000a, 0x54d4, 0x0a9a, 0x00a9, 0x000a}},
                        {{0x0000'07ff, 0x0000'07ff, 0x0000'07ff, 0x0000'07ff},
                         {0x0000'5555, 0x0000'5555, 0x0000'5555, 0x0000'5555},
                         {0x0000'5ddd, 0x0000'5ddd, 0x0000'5555, 0x0000'5555},
                         {0xaaaa'aaaa, 0x0055'5555, 0x0000'0888, 0x0000'0011},
                         {0xfff4'fff4, 0x007f'fa7f, 0x0000'7ffb, 0x0000'00ff},
                         {0xfff4'fff4, 0x007f'fc7f, 0x0000'7ffa, 0x0000'00ff},
                         {0xa9bb'bbbb, 0x0054'dddd, 0x0000'54dd, 0x0000'00a9},
                         {0xa9a9'a9a9, 0x0054'd4d4, 0x0000'54d4, 0x0000'00a9}},
                        {{0x0000'0000'0000'07ff, 0x0000'0000'0000'07ff},
                         {0x0000'5555'5555'5555, 0x0000'5555'5555'5555},
                         {0x0000'0000'0000'5ddd, 0x0000'0000'0000'5555},
                         {0x0000'0000'aaaa'aaaa, 0x0000'0000'0000'0888},
                         {0xfff4'fff4'fff4'fff4, 0x0000'7ffb'7ffb'7ffb},
                         {0x0000'0000'fff8'fff8, 0x0000'0000'0000'7ffa},
                         {0xa9bb'bbbb'a9bb'bbbb, 0x0000'54dd'dddd'd4dd},
                         {0x0000'0000'a9a9'a9a9, 0x0000'0000'0000'54d4}},
                        kVectorRightShiftSource);
  TestVectorInstruction(0xa100c457,  // Vsrl.vx v8, v16, x1, v0.t
                        {{61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63},
                         {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                         {46, 46, 46, 46, 46, 46, 46, 46, 42, 42, 42, 42, 42, 42, 42, 42},
                         {42, 42, 42, 42, 42, 42, 42, 42, 4, 4, 4, 4, 4, 4, 4, 4},
                         {61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63, 61, 63},
                         {61, 63, 61, 63, 62, 63, 62, 63, 61, 63, 61, 63, 61, 63, 61, 63},
                         {46, 46, 46, 42, 46, 46, 46, 42, 46, 46, 46, 42, 46, 46, 46, 42},
                         {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42}},
                        {{0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f},
                         {0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a},
                         {0x002e, 0x002e, 0x002e, 0x002e, 0x002a, 0x002a, 0x002a, 0x002a},
                         {0x002a, 0x002a, 0x002a, 0x002a, 0x0004, 0x0004, 0x0004, 0x0004},
                         {0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f},
                         {0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f, 0x003f},
                         {0x002e, 0x002a, 0x002e, 0x002a, 0x002e, 0x002a, 0x002e, 0x002a},
                         {0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a, 0x002a}},
                        {{0x003f'fd7f, 0x003f'fd7f, 0x003f'fd7f, 0x003f'fd7f},
                         {0x002a'aaaa, 0x002a'aaaa, 0x002a'aaaa, 0x002a'aaaa},
                         {0x002e'eeee, 0x002e'eeee, 0x002a'aaaa, 0x002a'aaaa},
                         {0x002a'aaaa, 0x002a'aaaa, 0x0004'4444, 0x0004'4444},
                         {0x003f'fd3f, 0x003f'fd3f, 0x003f'fdbf, 0x003f'fdbf},
                         {0x003f'fd3f, 0x003f'fe3f, 0x003f'fd7f, 0x003f'fd7f},
                         {0x002a'6eee, 0x002a'6eee, 0x002a'6eee, 0x002a'6eee},
                         {0x002a'6a6a, 0x002a'6a6a, 0x002a'6a6a, 0x002a'6a6a}},
                        {{0x0000'0000'003f'fd7f, 0x0000'0000'003f'fd7f},
                         {0x0000'0000'002a'aaaa, 0x0000'0000'002a'aaaa},
                         {0x0000'0000'002e'eeee, 0x0000'0000'002a'aaaa},
                         {0x0000'0000'002a'aaaa, 0x0000'0000'0004'4444},
                         {0x0000'0000'003f'fd3f, 0x0000'0000'003f'fdbf},
                         {0x0000'0000'003f'fe3f, 0x0000'0000'003f'fd7f},
                         {0x0000'0000'002a'6eee, 0x0000'0000'002a'6eee},
                         {0x0000'0000'002a'6a6a, 0x0000'0000'002a'6a6a}},
                        kVectorRightShiftSource);
  TestVectorInstruction(0xa101b457,  // Vsrl.vi v8, v16, 0x3, v0.t
                        {{30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31},
                         {21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21},
                         {23, 23, 23, 23, 23, 23, 23, 23, 21, 21, 21, 21, 21, 21, 21, 21},
                         {21, 21, 21, 21, 21, 21, 21, 21, 2, 2, 2, 2, 2, 2, 2, 2},
                         {30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31, 30, 31},
                         {30, 31, 30, 31, 31, 31, 31, 31, 30, 31, 30, 31, 30, 31, 30, 31},
                         {23, 23, 23, 21, 23, 23, 23, 21, 23, 23, 23, 21, 23, 23, 23, 21},
                         {21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21}},
                        {{0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe},
                         {0x1555, 0x1555, 0x1555, 0x1555, 0x1555, 0x1555, 0x1555, 0x1555},
                         {0x1777, 0x1777, 0x1777, 0x1777, 0x1555, 0x1555, 0x1555, 0x1555},
                         {0x1555, 0x1555, 0x1555, 0x1555, 0x0222, 0x0222, 0x0222, 0x0222},
                         {0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe},
                         {0x1ffe, 0x1ffe, 0x1fff, 0x1fff, 0x1ffe, 0x1ffe, 0x1ffe, 0x1ffe},
                         {0x1777, 0x1537, 0x1777, 0x1537, 0x1777, 0x1537, 0x1777, 0x1537},
                         {0x1535, 0x1535, 0x1535, 0x1535, 0x1535, 0x1535, 0x1535, 0x1535}},
                        {{0x1ffe'bffe, 0x1ffe'bffe, 0x1ffe'bffe, 0x1ffe'bffe},
                         {0x1555'5555, 0x1555'5555, 0x1555'5555, 0x1555'5555},
                         {0x1777'7777, 0x1777'7777, 0x1555'5555, 0x1555'5555},
                         {0x1555'5555, 0x1555'5555, 0x0222'2222, 0x0222'2222},
                         {0x1ffe'9ffe, 0x1ffe'9ffe, 0x1ffe'dffe, 0x1ffe'dffe},
                         {0x1ffe'9ffe, 0x1fff'1fff, 0x1ffe'bffe, 0x1ffe'bffe},
                         {0x1537'7777, 0x1537'7777, 0x1537'7777, 0x1537'7777},
                         {0x1535'3535, 0x1535'3535, 0x1535'3535, 0x1535'3535}},
                        {{0x1ffe'bffe'bffe'bffe, 0x1ffe'bffe'bffe'bffe},
                         {0x1555'5555'5555'5555, 0x1555'5555'5555'5555},
                         {0x1777'7777'7777'7777, 0x1555'5555'5555'5555},
                         {0x1555'5555'5555'5555, 0x0222'2222'2222'2222},
                         {0x1ffe'9ffe'9ffe'9ffe, 0x1ffe'dffe'dffe'dffe},
                         {0x1fff'1fff'1ffe'9ffe, 0x1ffe'bffe'bffe'bffe},
                         {0x1537'7777'7537'7777, 0x1537'7777'7537'7777},
                         {0x1535'3535'3535'3535, 0x1535'3535'3535'3535}},
                        kVectorRightShiftSource);
}

TEST_F(Riscv64InterpreterTest, TestVsra) {
  TestVectorInstruction(
      0xa50c0457,  // Vsra.vv v8, v16, v24, v0.t
      {{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
       {213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213},
       {221, 255, 221, 255, 221, 255, 221, 255, 213, 255, 213, 255, 213, 255, 213, 255},
       {170, 234, 250, 254, 213, 234, 250, 254, 8, 4, 1, 0, 17, 4, 1, 0},
       {244, 255, 255, 255, 250, 255, 255, 255, 251, 255, 255, 255, 246, 255, 255, 255},
       {244, 255, 255, 255, 252, 255, 255, 255, 250, 255, 255, 255, 245, 255, 255, 255},
       {187, 238, 251, 254, 221, 238, 251, 254, 221, 238, 251, 254, 187, 238, 251, 254},
       {169, 234, 250, 254, 212, 234, 250, 254, 212, 234, 250, 254, 169, 234, 250, 254}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xd555, 0xd555, 0xd555, 0xd555, 0xd555, 0xd555, 0xd555, 0xd555},
       {0xdddd, 0xdddd, 0xdddd, 0xdddd, 0xd555, 0xd555, 0xd555, 0xd555},
       {0xaaaa, 0xfaaa, 0xffd5, 0xfffa, 0x0888, 0x0111, 0x0011, 0x0001},
       {0xfff4, 0xffff, 0xffff, 0xffff, 0xfffb, 0xffff, 0xffff, 0xffff},
       {0xfff4, 0xffff, 0xffff, 0xffff, 0xfffa, 0xffff, 0xffff, 0xffff},
       {0xbbbb, 0xfa9b, 0xffdd, 0xfffa, 0xdddd, 0xfa9b, 0xffbb, 0xfffa},
       {0xa9a9, 0xfa9a, 0xffd4, 0xfffa, 0xd4d4, 0xfa9a, 0xffa9, 0xfffa}},
      {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
       {0xffff'd555, 0xffff'd555, 0xffff'd555, 0xffff'd555},
       {0xffff'dddd, 0xffff'dddd, 0xffff'd555, 0xffff'd555},
       {0xaaaa'aaaa, 0xffd5'5555, 0x0000'0888, 0x0000'0011},
       {0xfff4'fff4, 0xffff'fa7f, 0xffff'fffb, 0xffff'ffff},
       {0xfff4'fff4, 0xffff'fc7f, 0xffff'fffa, 0xffff'ffff},
       {0xa9bb'bbbb, 0xffd4'dddd, 0xffff'd4dd, 0xffff'ffa9},
       {0xa9a9'a9a9, 0xffd4'd4d4, 0xffff'd4d4, 0xffff'ffa9}},
      {{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
       {0xffff'd555'5555'5555, 0xffff'd555'5555'5555},
       {0xffff'ffff'ffff'dddd, 0xffff'ffff'ffff'd555},
       {0xffff'ffff'aaaa'aaaa, 0x0000'0000'0000'0888},
       {0xfff4'fff4'fff4'fff4, 0xffff'fffb'7ffb'7ffb},
       {0xffff'ffff'fff8'fff8, 0xffff'ffff'ffff'fffa},
       {0xa9bb'bbbb'a9bb'bbbb, 0xffff'd4dd'dddd'd4dd},
       {0xffff'ffff'a9a9'a9a9, 0xffff'ffff'ffff'd4d4}},
      kVectorRightShiftSource);
  TestVectorInstruction(
      0xa500c457,  // Vsra.vx v8, v16, x1, v0.t
      {{253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255},
       {234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234},
       {238, 238, 238, 238, 238, 238, 238, 238, 234, 234, 234, 234, 234, 234, 234, 234},
       {234, 234, 234, 234, 234, 234, 234, 234, 4, 4, 4, 4, 4, 4, 4, 4},
       {253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255, 253, 255},
       {253, 255, 253, 255, 254, 255, 254, 255, 253, 255, 253, 255, 253, 255, 253, 255},
       {238, 238, 238, 234, 238, 238, 238, 234, 238, 238, 238, 234, 238, 238, 238, 234},
       {234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234}},
      {{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea},
       {0xffee, 0xffee, 0xffee, 0xffee, 0xffea, 0xffea, 0xffea, 0xffea},
       {0xffea, 0xffea, 0xffea, 0xffea, 0x0004, 0x0004, 0x0004, 0x0004},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
       {0xffee, 0xffea, 0xffee, 0xffea, 0xffee, 0xffea, 0xffee, 0xffea},
       {0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea, 0xffea}},
      {{0xffff'fd7f, 0xffff'fd7f, 0xffff'fd7f, 0xffff'fd7f},
       {0xffea'aaaa, 0xffea'aaaa, 0xffea'aaaa, 0xffea'aaaa},
       {0xffee'eeee, 0xffee'eeee, 0xffea'aaaa, 0xffea'aaaa},
       {0xffea'aaaa, 0xffea'aaaa, 0x0004'4444, 0x0004'4444},
       {0xffff'fd3f, 0xffff'fd3f, 0xffff'fdbf, 0xffff'fdbf},
       {0xffff'fd3f, 0xffff'fe3f, 0xffff'fd7f, 0xffff'fd7f},
       {0xffea'6eee, 0xffea'6eee, 0xffea'6eee, 0xffea'6eee},
       {0xffea'6a6a, 0xffea'6a6a, 0xffea'6a6a, 0xffea'6a6a}},
      {{0xffff'ffff'ffff'fd7f, 0xffff'ffff'ffff'fd7f},
       {0xffff'ffff'ffea'aaaa, 0xffff'ffff'ffea'aaaa},
       {0xffff'ffff'ffee'eeee, 0xffff'ffff'ffea'aaaa},
       {0xffff'ffff'ffea'aaaa, 0x0000'0000'0004'4444},
       {0xffff'ffff'ffff'fd3f, 0xffff'ffff'ffff'fdbf},
       {0xffff'ffff'ffff'fe3f, 0xffff'ffff'ffff'fd7f},
       {0xffff'ffff'ffea'6eee, 0xffff'ffff'ffea'6eee},
       {0xffff'ffff'ffea'6a6a, 0xffff'ffff'ffea'6a6a}},
      kVectorRightShiftSource);
  TestVectorInstruction(
      0xa501b457,  // Vsra.vi v8, v16, 0x3, v0.t
      {{254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {247, 247, 247, 247, 247, 247, 247, 247, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 2, 2, 2, 2, 2, 2, 2, 2},
       {254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255},
       {254, 255, 254, 255, 255, 255, 255, 255, 254, 255, 254, 255, 254, 255, 254, 255},
       {247, 247, 247, 245, 247, 247, 247, 245, 247, 247, 247, 245, 247, 247, 247, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245}},
      {{0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe},
       {0xf555, 0xf555, 0xf555, 0xf555, 0xf555, 0xf555, 0xf555, 0xf555},
       {0xf777, 0xf777, 0xf777, 0xf777, 0xf555, 0xf555, 0xf555, 0xf555},
       {0xf555, 0xf555, 0xf555, 0xf555, 0x0222, 0x0222, 0x0222, 0x0222},
       {0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe, 0xfffe},
       {0xfffe, 0xfffe, 0xffff, 0xffff, 0xfffe, 0xfffe, 0xfffe, 0xfffe},
       {0xf777, 0xf537, 0xf777, 0xf537, 0xf777, 0xf537, 0xf777, 0xf537},
       {0xf535, 0xf535, 0xf535, 0xf535, 0xf535, 0xf535, 0xf535, 0xf535}},
      {{0xfffe'bffe, 0xfffe'bffe, 0xfffe'bffe, 0xfffe'bffe},
       {0xf555'5555, 0xf555'5555, 0xf555'5555, 0xf555'5555},
       {0xf777'7777, 0xf777'7777, 0xf555'5555, 0xf555'5555},
       {0xf555'5555, 0xf555'5555, 0x0222'2222, 0x0222'2222},
       {0xfffe'9ffe, 0xfffe'9ffe, 0xfffe'dffe, 0xfffe'dffe},
       {0xfffe'9ffe, 0xffff'1fff, 0xfffe'bffe, 0xfffe'bffe},
       {0xf537'7777, 0xf537'7777, 0xf537'7777, 0xf537'7777},
       {0xf535'3535, 0xf535'3535, 0xf535'3535, 0xf535'3535}},
      {{0xfffe'bffe'bffe'bffe, 0xfffe'bffe'bffe'bffe},
       {0xf555'5555'5555'5555, 0xf555'5555'5555'5555},
       {0xf777'7777'7777'7777, 0xf555'5555'5555'5555},
       {0xf555'5555'5555'5555, 0x0222'2222'2222'2222},
       {0xfffe'9ffe'9ffe'9ffe, 0xfffe'dffe'dffe'dffe},
       {0xffff'1fff'1ffe'9ffe, 0xfffe'bffe'bffe'bffe},
       {0xf537'7777'7537'7777, 0xf537'7777'7537'7777},
       {0xf535'3535'3535'3535, 0xf535'3535'3535'3535}},
      kVectorRightShiftSource);
}

TEST_F(Riscv64InterpreterTest, TestVmacc) {
  TestVectorInstruction(0xb5882457,  // vmacc.vv v8, v16, v24, v0.t
                        {{85, 87, 93, 103, 121, 135, 157, 183, 221, 247, 29, 71, 117, 167, 221, 23},
                         {85, 151, 221, 39, 137, 199, 29, 119, 237, 55, 157, 7, 117, 231, 93, 215},
                         {85, 215, 93, 231, 153, 7, 157, 55, 253, 119, 29, 199, 117, 39, 221, 151},
                         {85, 23, 221, 167, 169, 71, 29, 247, 13, 183, 157, 135, 117, 103, 93, 87},
                         {85, 87, 93, 103, 185, 135, 157, 183, 29, 247, 29, 71, 117, 167, 221, 23},
                         {85, 151, 221, 39, 201, 199, 29, 119, 45, 55, 157, 7, 117, 231, 93, 215},
                         {85, 215, 93, 231, 217, 7, 157, 55, 61, 119, 29, 199, 117, 39, 221, 151},
                         {85, 23, 221, 167, 233, 71, 29, 247, 77, 183, 157, 135, 117, 103, 93, 87}},
                        {{0x5555, 0x6d5d, 0x2a79, 0xfd9d, 0xfedd, 0x0e1d, 0xc675, 0x9edd},
                         {0x9755, 0xafdd, 0x7d89, 0x411d, 0x52ed, 0x529d, 0x0b75, 0xe45d},
                         {0xdd55, 0xf65d, 0xd499, 0x889d, 0xaafd, 0x9b1d, 0x5475, 0x2ddd},
                         {0x2755, 0x40dd, 0x2fa9, 0xd41d, 0x070d, 0xe79d, 0xa175, 0x7b5d},
                         {0x7555, 0x8f5d, 0x8eb9, 0x239d, 0x671d, 0x381d, 0xf275, 0xccdd},
                         {0xc755, 0xe1dd, 0xf1c9, 0x771d, 0xcb2d, 0x8c9d, 0x4775, 0x225d},
                         {0x1d55, 0x385d, 0x58d9, 0xce9d, 0x333d, 0xe51d, 0xa075, 0x7bdd},
                         {0x7755, 0x92dd, 0xc3e9, 0x2a1d, 0x9f4d, 0x419d, 0xfd75, 0xd95d}},
                        {{0x5e57'5555, 0xc9f2'2a79, 0xb34a'fedd, 0x0e55'c675},
                         {0xf52b'9755, 0x73d8'7d89, 0x6033'52ed, 0xae30'0b75},
                         {0x9807'dd55, 0x29c6'd499, 0x1923'aafd, 0x5a12'5475},
                         {0x46ec'2755, 0xebbd'2fa9, 0xde1c'070d, 0x11fc'a175},
                         {0x01d8'7555, 0xb9bb'8eb9, 0xaf1c'671d, 0xd5ee'f275},
                         {0xc8cc'c755, 0x93c1'f1c9, 0x8c24'cb2d, 0xa5e9'4775},
                         {0x9bc9'1d55, 0x79d0'58d9, 0x7535'333d, 0x81eb'a075},
                         {0x7acd'7755, 0x6be6'c3e9, 0x6a4d'9f4d, 0x69f5'fd75}},
                        {{0x51a4'026b'5e57'5555, 0xfbed'024a'b34a'fedd},
                         {0xa533'ff24'f52b'9755, 0x5d89'090c'6033'52ed},
                         {0x14dc'0fee'9807'dd55, 0xdb3d'23de'1923'aafd},
                         {0xa09c'34c8'46ec'2755, 0x7509'52bf'de1c'070d},
                         {0x4874'6db2'01d8'7555, 0x2aed'95b1'af1c'671d},
                         {0x0c64'baab'c8cc'c755, 0xfce9'ecb3'8c24'cb2d},
                         {0xec6d'1bb5'9bc9'1d55, 0xeafe'57c5'7535'333d},
                         {0xe88d'90cf'7acd'7755, 0xf52a'd6e7'6a4d'9f4d}},
                        kVectorCalculationsSource);
  TestVectorInstruction(
      0xb500e457,  // vmacc.vx v8, x1, v16, v0.t
      {{85, 255, 169, 83, 253, 167, 81, 251, 165, 79, 249, 163, 77, 247, 161, 75},
       {245, 159, 73, 243, 157, 71, 241, 155, 69, 239, 153, 67, 237, 151, 65, 235},
       {149, 63, 233, 147, 61, 231, 145, 59, 229, 143, 57, 227, 141, 55, 225, 139},
       {53, 223, 137, 51, 221, 135, 49, 219, 133, 47, 217, 131, 45, 215, 129, 43},
       {213, 127, 41, 211, 125, 39, 209, 123, 37, 207, 121, 35, 205, 119, 33, 203},
       {117, 31, 201, 115, 29, 199, 113, 27, 197, 111, 25, 195, 109, 23, 193, 107},
       {21, 191, 105, 19, 189, 103, 17, 187, 101, 15, 185, 99, 13, 183, 97, 11},
       {181, 95, 9, 179, 93, 7, 177, 91, 5, 175, 89, 3, 173, 87, 1, 171}},
      {{0xff55, 0xa8a9, 0x51fd, 0xfb51, 0xa4a5, 0x4df9, 0xf74d, 0xa0a1},
       {0x49f5, 0xf349, 0x9c9d, 0x45f1, 0xef45, 0x9899, 0x41ed, 0xeb41},
       {0x9495, 0x3de9, 0xe73d, 0x9091, 0x39e5, 0xe339, 0x8c8d, 0x35e1},
       {0xdf35, 0x8889, 0x31dd, 0xdb31, 0x8485, 0x2dd9, 0xd72d, 0x8081},
       {0x29d5, 0xd329, 0x7c7d, 0x25d1, 0xcf25, 0x7879, 0x21cd, 0xcb21},
       {0x7475, 0x1dc9, 0xc71d, 0x7071, 0x19c5, 0xc319, 0x6c6d, 0x15c1},
       {0xbf15, 0x6869, 0x11bd, 0xbb11, 0x6465, 0x0db9, 0xb70d, 0x6061},
       {0x09b5, 0xb309, 0x5c5d, 0x05b1, 0xaf05, 0x5859, 0x01ad, 0xab01}},
      {{0xa8a8'ff55, 0x50a6'51fd, 0xf8a3'a4a5, 0xa0a0'f74d},
       {0x489e'49f5, 0xf09b'9c9d, 0x9898'ef45, 0x4096'41ed},
       {0xe893'9495, 0x9090'e73d, 0x388e'39e5, 0xe08b'8c8d},
       {0x8888'df35, 0x3086'31dd, 0xd883'8485, 0x8080'd72d},
       {0x287e'29d5, 0xd07b'7c7d, 0x7878'cf25, 0x2076'21cd},
       {0xc873'7475, 0x7070'c71d, 0x186e'19c5, 0xc06b'6c6d},
       {0x6868'bf15, 0x1066'11bd, 0xb863'6465, 0x6060'b70d},
       {0x085e'09b5, 0xb05b'5c5d, 0x5858'af05, 0x0056'01ad}},
      {{0xfb50'fca7'a8a8'ff55, 0xa0a0'f74c'f8a3'a4a5},
       {0x45f0'f1f2'489e'49f5, 0xeb40'ec97'9898'ef45},
       {0x9090'e73c'e893'9495, 0x35e0'e1e2'388e'39e5},
       {0xdb30'dc87'8888'df35, 0x8080'd72c'd883'8485},
       {0x25d0'd1d2'287e'29d5, 0xcb20'cc77'7878'cf25},
       {0x7070'c71c'c873'7475, 0x15c0'c1c2'186e'19c5},
       {0xbb10'bc67'6868'bf15, 0x6060'b70c'b863'6465},
       {0x05b0'b1b2'085e'09b5, 0xab00'ac57'5858'af05}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVnmsac) {
  TestVectorInstruction(0xbd882457,  // vnmsac.vv v8, v16, v24, v0.t
                        {{85, 83, 77, 67, 49, 35, 13, 243, 205, 179, 141, 99, 53, 3, 205, 147},
                         {85, 19, 205, 131, 33, 227, 141, 51, 189, 115, 13, 163, 53, 195, 77, 211},
                         {85, 211, 77, 195, 17, 163, 13, 115, 173, 51, 141, 227, 53, 131, 205, 19},
                         {85, 147, 205, 3, 1, 99, 141, 179, 157, 243, 13, 35, 53, 67, 77, 83},
                         {85, 83, 77, 67, 241, 35, 13, 243, 141, 179, 141, 99, 53, 3, 205, 147},
                         {85, 19, 205, 131, 225, 227, 141, 51, 125, 115, 13, 163, 53, 195, 77, 211},
                         {85, 211, 77, 195, 209, 163, 13, 115, 109, 51, 141, 227, 53, 131, 205, 19},
                         {85, 147, 205, 3, 193, 99, 141, 179, 93, 243, 13, 35, 53, 67, 77, 83}},
                        {{0x5555, 0x3d4d, 0x8031, 0xad0d, 0xabcd, 0x9c8d, 0xe435, 0x0bcd},
                         {0x1355, 0xfacd, 0x2d21, 0x698d, 0x57bd, 0x580d, 0x9f35, 0xc64d},
                         {0xcd55, 0xb44d, 0xd611, 0x220d, 0xffad, 0x0f8d, 0x5635, 0x7ccd},
                         {0x8355, 0x69cd, 0x7b01, 0xd68d, 0xa39d, 0xc30d, 0x0935, 0x2f4d},
                         {0x3555, 0x1b4d, 0x1bf1, 0x870d, 0x438d, 0x728d, 0xb835, 0xddcd},
                         {0xe355, 0xc8cd, 0xb8e1, 0x338d, 0xdf7d, 0x1e0d, 0x6335, 0x884d},
                         {0x8d55, 0x724d, 0x51d1, 0xdc0d, 0x776d, 0xc58d, 0x0a35, 0x2ecd},
                         {0x3355, 0x17cd, 0xe6c1, 0x808d, 0x0b5d, 0x690d, 0xad35, 0xd14d}},
                        {{0x4c53'5555, 0xe0b8'8031, 0xf75f'abcd, 0x9c54'e435},
                         {0xb57f'1355, 0x36d2'2d21, 0x4a77'57bd, 0xfc7a'9f35},
                         {0x12a2'cd55, 0x80e3'd611, 0x9186'ffad, 0x5098'5635},
                         {0x63be'8355, 0xbeed'7b01, 0xcc8e'a39d, 0x98ae'0935},
                         {0xa8d2'3555, 0xf0ef'1bf1, 0xfb8e'438d, 0xd4bb'b835},
                         {0xe1dd'e355, 0x16e8'b8e1, 0x1e85'df7d, 0x04c1'6335},
                         {0x0ee1'8d55, 0x30da'51d1, 0x3575'776d, 0x28bf'0a35},
                         {0x2fdd'3355, 0x3ec3'e6c1, 0x405d'0b5d, 0x40b4'ad35}},
                        {{0x5906'a83f'4c53'5555, 0xaebd'a85f'f75f'abcd},
                         {0x0576'ab85'b57f'1355, 0x4d21'a19e'4a77'57bd},
                         {0x95ce'9abc'12a2'cd55, 0xcf6d'86cc'9186'ffad},
                         {0x0a0e'75e2'63be'8355, 0x35a1'57ea'cc8e'a39d},
                         {0x6236'3cf8'a8d2'3555, 0x7fbd'14f8'fb8e'438d},
                         {0x9e45'effe'e1dd'e355, 0xadc0'bdf7'1e85'df7d},
                         {0xbe3d'8ef5'0ee1'8d55, 0xbfac'52e5'3575'776d},
                         {0xc21d'19db'2fdd'3355, 0xb57f'd3c3'405d'0b5d}},
                        kVectorCalculationsSource);
  TestVectorInstruction(
      0xbd00e457,  // vnmsac.vx v8, x1, v16, v0.t
      {{85, 171, 1, 87, 173, 3, 89, 175, 5, 91, 177, 7, 93, 179, 9, 95},
       {181, 11, 97, 183, 13, 99, 185, 15, 101, 187, 17, 103, 189, 19, 105, 191},
       {21, 107, 193, 23, 109, 195, 25, 111, 197, 27, 113, 199, 29, 115, 201, 31},
       {117, 203, 33, 119, 205, 35, 121, 207, 37, 123, 209, 39, 125, 211, 41, 127},
       {213, 43, 129, 215, 45, 131, 217, 47, 133, 219, 49, 135, 221, 51, 137, 223},
       {53, 139, 225, 55, 141, 227, 57, 143, 229, 59, 145, 231, 61, 147, 233, 63},
       {149, 235, 65, 151, 237, 67, 153, 239, 69, 155, 241, 71, 157, 243, 73, 159},
       {245, 75, 161, 247, 77, 163, 249, 79, 165, 251, 81, 167, 253, 83, 169, 255}},
      {{0xab55, 0x0201, 0x58ad, 0xaf59, 0x0605, 0x5cb1, 0xb35d, 0x0a09},
       {0x60b5, 0xb761, 0x0e0d, 0x64b9, 0xbb65, 0x1211, 0x68bd, 0xbf69},
       {0x1615, 0x6cc1, 0xc36d, 0x1a19, 0x70c5, 0xc771, 0x1e1d, 0x74c9},
       {0xcb75, 0x2221, 0x78cd, 0xcf79, 0x2625, 0x7cd1, 0xd37d, 0x2a29},
       {0x80d5, 0xd781, 0x2e2d, 0x84d9, 0xdb85, 0x3231, 0x88dd, 0xdf89},
       {0x3635, 0x8ce1, 0xe38d, 0x3a39, 0x90e5, 0xe791, 0x3e3d, 0x94e9},
       {0xeb95, 0x4241, 0x98ed, 0xef99, 0x4645, 0x9cf1, 0xf39d, 0x4a49},
       {0xa0f5, 0xf7a1, 0x4e4d, 0xa4f9, 0xfba5, 0x5251, 0xa8fd, 0xffa9}},
      {{0x0201'ab55, 0x5a04'58ad, 0xb207'0605, 0x0a09'b35d},
       {0x620c'60b5, 0xba0f'0e0d, 0x1211'bb65, 0x6a14'68bd},
       {0xc217'1615, 0x1a19'c36d, 0x721c'70c5, 0xca1f'1e1d},
       {0x2221'cb75, 0x7a24'78cd, 0xd227'2625, 0x2a29'd37d},
       {0x822c'80d5, 0xda2f'2e2d, 0x3231'db85, 0x8a34'88dd},
       {0xe237'3635, 0x3a39'e38d, 0x923c'90e5, 0xea3f'3e3d},
       {0x4241'eb95, 0x9a44'98ed, 0xf247'4645, 0x4a49'f39d},
       {0xa24c'a0f5, 0xfa4f'4e4d, 0x5251'fba5, 0xaa54'a8fd}},
      {{0xaf59'ae03'0201'ab55, 0x0a09'b35d'b207'0605},
       {0x64b9'b8b8'620c'60b5, 0xbf69'be13'1211'bb65},
       {0x1a19'c36d'c217'1615, 0x74c9'c8c8'721c'70c5},
       {0xcf79'ce23'2221'cb75, 0x2a29'd37d'd227'2625},
       {0x84d9'd8d8'822c'80d5, 0xdf89'de33'3231'db85},
       {0x3a39'e38d'e237'3635, 0x94e9'e8e8'923c'90e5},
       {0xef99'ee43'4241'eb95, 0x4a49'f39d'f247'4645},
       {0xa4f9'f8f8'a24c'a0f5, 0xffa9'fe53'5251'fba5}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVmadd) {
  TestVectorInstruction(
      0xa5882457,  // vmadd.vv v8, v16, v24, v0.t
      {{0, 215, 174, 133, 93, 51, 10, 225, 185, 143, 102, 61, 20, 235, 194, 153},
       {112, 71, 30, 245, 205, 163, 122, 81, 41, 255, 214, 173, 132, 91, 50, 9},
       {224, 183, 142, 101, 61, 19, 234, 193, 153, 111, 70, 29, 244, 203, 162, 121},
       {80, 39, 254, 213, 173, 131, 90, 49, 9, 223, 182, 141, 100, 59, 18, 233},
       {192, 151, 110, 69, 29, 243, 202, 161, 121, 79, 38, 253, 212, 171, 130, 89},
       {48, 7, 222, 181, 141, 99, 58, 17, 233, 191, 150, 109, 68, 27, 242, 201},
       {160, 119, 78, 37, 253, 211, 170, 129, 89, 47, 6, 221, 180, 139, 98, 57},
       {16, 231, 190, 149, 109, 67, 26, 241, 201, 159, 118, 77, 36, 251, 210, 169}},
      {{0xd700, 0x2fae, 0x885d, 0xe10a, 0x39b9, 0x9266, 0xeb14, 0x43c2},
       {0x9c70, 0xf51e, 0x4dcd, 0xa67a, 0xff29, 0x57d6, 0xb084, 0x0932},
       {0x61e0, 0xba8e, 0x133d, 0x6bea, 0xc499, 0x1d46, 0x75f4, 0xcea2},
       {0x2750, 0x7ffe, 0xd8ad, 0x315a, 0x8a09, 0xe2b6, 0x3b64, 0x9412},
       {0xecc0, 0x456e, 0x9e1d, 0xf6ca, 0x4f79, 0xa826, 0x00d4, 0x5982},
       {0xb230, 0x0ade, 0x638d, 0xbc3a, 0x14e9, 0x6d96, 0xc644, 0x1ef2},
       {0x77a0, 0xd04e, 0x28fd, 0x81aa, 0xda59, 0x3306, 0x8bb4, 0xe462},
       {0x3d10, 0x95be, 0xee6d, 0x471a, 0x9fc9, 0xf876, 0x5124, 0xa9d2}},
      {{0x2fad'd700, 0x8bb4'885d, 0xe7bb'39b9, 0x43c1'eb14},
       {0x9fc8'9c70, 0xfbcf'4dcd, 0x57d5'ff29, 0xb3dc'b084},
       {0x0fe3'61e0, 0x6bea'133d, 0xc7f0'c499, 0x23f7'75f4},
       {0x7ffe'2750, 0xdc04'd8ad, 0x380b'8a09, 0x9412'3b64},
       {0xf018'ecc0, 0x4c1f'9e1d, 0xa826'4f79, 0x042d'00d4},
       {0x6033'b230, 0xbc3a'638d, 0x1841'14e9, 0x7447'c644},
       {0xd04e'77a0, 0x2c55'28fd, 0x885b'da59, 0xe462'8bb4},
       {0x4069'3d10, 0x9c6f'ee6d, 0xf876'9fc9, 0x547d'5124}},
      {{0xe109'ddb2'2fad'd700, 0x43c1'eb13'e7bb'39b9},
       {0xa679'f877'9fc8'9c70, 0x0932'05d9'57d5'ff29},
       {0x6bea'133d'0fe3'61e0, 0xcea2'209e'c7f0'c499},
       {0x315a'2e02'7ffe'2750, 0x9412'3b64'380b'8a09},
       {0xf6ca'48c7'f018'ecc0, 0x5982'5629'a826'4f79},
       {0xbc3a'638d'6033'b230, 0x1ef2'70ef'1841'14e9},
       {0x81aa'7e52'd04e'77a0, 0xe462'8bb4'885b'da59},
       {0x471a'9918'4069'3d10, 0xa9d2'a679'f876'9fc9}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0xa500e457,  // vmadd.vx v8, x1, v16, v0.t
      {{114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255, 128, 1},
       {130, 3, 132, 5, 134, 7, 136, 9, 138, 11, 140, 13, 142, 15, 144, 17},
       {146, 19, 148, 21, 150, 23, 152, 25, 154, 27, 156, 29, 158, 31, 160, 33},
       {162, 35, 164, 37, 166, 39, 168, 41, 170, 43, 172, 45, 174, 47, 176, 49},
       {178, 51, 180, 53, 182, 55, 184, 57, 186, 59, 188, 61, 190, 63, 192, 65},
       {194, 67, 196, 69, 198, 71, 200, 73, 202, 75, 204, 77, 206, 79, 208, 81},
       {210, 83, 212, 85, 214, 87, 216, 89, 218, 91, 220, 93, 222, 95, 224, 97},
       {226, 99, 228, 101, 230, 103, 232, 105, 234, 107, 236, 109, 238, 111, 240, 113}},
      {{0x9d72, 0x9f74, 0xa176, 0xa378, 0xa57a, 0xa77c, 0xa97e, 0xab80},
       {0xad82, 0xaf84, 0xb186, 0xb388, 0xb58a, 0xb78c, 0xb98e, 0xbb90},
       {0xbd92, 0xbf94, 0xc196, 0xc398, 0xc59a, 0xc79c, 0xc99e, 0xcba0},
       {0xcda2, 0xcfa4, 0xd1a6, 0xd3a8, 0xd5aa, 0xd7ac, 0xd9ae, 0xdbb0},
       {0xddb2, 0xdfb4, 0xe1b6, 0xe3b8, 0xe5ba, 0xe7bc, 0xe9be, 0xebc0},
       {0xedc2, 0xefc4, 0xf1c6, 0xf3c8, 0xf5ca, 0xf7cc, 0xf9ce, 0xfbd0},
       {0xfdd2, 0xffd4, 0x01d6, 0x03d8, 0x05da, 0x07dc, 0x09de, 0x0be0},
       {0x0de2, 0x0fe4, 0x11e6, 0x13e8, 0x15ea, 0x17ec, 0x19ee, 0x1bf0}},
      {{0xf4c9'9d72, 0xf8cd'a176, 0xfcd1'a57a, 0x00d5'a97e},
       {0x04d9'ad82, 0x08dd'b186, 0x0ce1'b58a, 0x10e5'b98e},
       {0x14e9'bd92, 0x18ed'c196, 0x1cf1'c59a, 0x20f5'c99e},
       {0x24f9'cda2, 0x28fd'd1a6, 0x2d01'd5aa, 0x3105'd9ae},
       {0x3509'ddb2, 0x390d'e1b6, 0x3d11'e5ba, 0x4115'e9be},
       {0x4519'edc2, 0x491d'f1c6, 0x4d21'f5ca, 0x5125'f9ce},
       {0x5529'fdd2, 0x592e'01d6, 0x5d32'05da, 0x6136'09de},
       {0x653a'0de2, 0x693e'11e6, 0x6d42'15ea, 0x7146'19ee}},
      {{0xa378'4c20'f4c9'9d72, 0xab80'5428'fcd1'a57a},
       {0xb388'5c31'04d9'ad82, 0xbb90'6439'0ce1'b58a},
       {0xc398'6c41'14e9'bd92, 0xcba0'7449'1cf1'c59a},
       {0xd3a8'7c51'24f9'cda2, 0xdbb0'8459'2d01'd5aa},
       {0xe3b8'8c61'3509'ddb2, 0xebc0'9469'3d11'e5ba},
       {0xf3c8'9c71'4519'edc2, 0xfbd0'a479'4d21'f5ca},
       {0x03d8'ac81'5529'fdd2, 0x0be0'b489'5d32'05da},
       {0x13e8'bc91'653a'0de2, 0x1bf0'c499'6d42'15ea}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVnmsub) {
  TestVectorInstruction(
      0xad882457,  // vnmsub.vv v8, v16, v24, v0.t
      {{0, 45, 90, 135, 181, 225, 14, 59, 105, 149, 194, 239, 28, 73, 118, 163},
       {208, 253, 42, 87, 133, 177, 222, 11, 57, 101, 146, 191, 236, 25, 70, 115},
       {160, 205, 250, 39, 85, 129, 174, 219, 9, 53, 98, 143, 188, 233, 22, 67},
       {112, 157, 202, 247, 37, 81, 126, 171, 217, 5, 50, 95, 140, 185, 230, 19},
       {64, 109, 154, 199, 245, 33, 78, 123, 169, 213, 2, 47, 92, 137, 182, 227},
       {16, 61, 106, 151, 197, 241, 30, 75, 121, 165, 210, 255, 44, 89, 134, 179},
       {224, 13, 58, 103, 149, 193, 238, 27, 73, 117, 162, 207, 252, 41, 86, 131},
       {176, 221, 10, 55, 101, 145, 190, 235, 25, 69, 114, 159, 204, 249, 38, 83}},
      {{0x2d00, 0xdc5a, 0x8bb5, 0x3b0e, 0xea69, 0x99c2, 0x491c, 0xf876},
       {0xa7d0, 0x572a, 0x0685, 0xb5de, 0x6539, 0x1492, 0xc3ec, 0x7346},
       {0x22a0, 0xd1fa, 0x8155, 0x30ae, 0xe009, 0x8f62, 0x3ebc, 0xee16},
       {0x9d70, 0x4cca, 0xfc25, 0xab7e, 0x5ad9, 0x0a32, 0xb98c, 0x68e6},
       {0x1840, 0xc79a, 0x76f5, 0x264e, 0xd5a9, 0x8502, 0x345c, 0xe3b6},
       {0x9310, 0x426a, 0xf1c5, 0xa11e, 0x5079, 0xffd2, 0xaf2c, 0x5e86},
       {0x0de0, 0xbd3a, 0x6c95, 0x1bee, 0xcb49, 0x7aa2, 0x29fc, 0xd956},
       {0x88b0, 0x380a, 0xe765, 0x96be, 0x4619, 0xf572, 0xa4cc, 0x5426}},
      {{0xdc5a'2d00, 0x9063'8bb5, 0x446c'ea69, 0xf876'491c},
       {0xac7f'a7d0, 0x6089'0685, 0x1492'6539, 0xc89b'c3ec},
       {0x7ca5'22a0, 0x30ae'8155, 0xe4b7'e009, 0x98c1'3ebc},
       {0x4cca'9d70, 0x00d3'fc25, 0xb4dd'5ad9, 0x68e6'b98c},
       {0x1cf0'1840, 0xd0f9'76f5, 0x8502'd5a9, 0x390c'345c},
       {0xed15'9310, 0xa11e'f1c5, 0x5528'5079, 0x0931'af2c},
       {0xbd3b'0de0, 0x7144'6c95, 0x254d'cb49, 0xd957'29fc},
       {0x8d60'88b0, 0x4169'e765, 0xf573'4619, 0xa97c'a4cc}},
      {{0x3b0e'365f'dc5a'2d00, 0xf876'491c'446c'ea69},
       {0xb5de'5bda'ac7f'a7d0, 0x7346'6e97'1492'6539},
       {0x30ae'8155'7ca5'22a0, 0xee16'9411'e4b7'e009},
       {0xab7e'a6d0'4cca'9d70, 0x68e6'b98c'b4dd'5ad9},
       {0x264e'cc4b'1cf0'1840, 0xe3b6'df07'8502'd5a9},
       {0xa11e'f1c5'ed15'9310, 0x5e87'0482'5528'5079},
       {0x1bef'1740'bd3b'0de0, 0xd957'29fd'254d'cb49},
       {0x96bf'3cbb'8d60'88b0, 0x5427'4f77'f573'4619}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0xad00e457,  // vnmsub.vx v8, x1, v16, v0.t
      {{142, 15, 144, 17, 146, 19, 148, 21, 150, 23, 152, 25, 154, 27, 156, 29},
       {158, 31, 160, 33, 162, 35, 164, 37, 166, 39, 168, 41, 170, 43, 172, 45},
       {174, 47, 176, 49, 178, 51, 180, 53, 182, 55, 184, 57, 186, 59, 188, 61},
       {190, 63, 192, 65, 194, 67, 196, 69, 198, 71, 200, 73, 202, 75, 204, 77},
       {206, 79, 208, 81, 210, 83, 212, 85, 214, 87, 216, 89, 218, 91, 220, 93},
       {222, 95, 224, 97, 226, 99, 228, 101, 230, 103, 232, 105, 234, 107, 236, 109},
       {238, 111, 240, 113, 242, 115, 244, 117, 246, 119, 248, 121, 250, 123, 252, 125},
       {254, 127, 0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141}},
      {{0x648e, 0x6690, 0x6892, 0x6a94, 0x6c96, 0x6e98, 0x709a, 0x729c},
       {0x749e, 0x76a0, 0x78a2, 0x7aa4, 0x7ca6, 0x7ea8, 0x80aa, 0x82ac},
       {0x84ae, 0x86b0, 0x88b2, 0x8ab4, 0x8cb6, 0x8eb8, 0x90ba, 0x92bc},
       {0x94be, 0x96c0, 0x98c2, 0x9ac4, 0x9cc6, 0x9ec8, 0xa0ca, 0xa2cc},
       {0xa4ce, 0xa6d0, 0xa8d2, 0xaad4, 0xacd6, 0xaed8, 0xb0da, 0xb2dc},
       {0xb4de, 0xb6e0, 0xb8e2, 0xbae4, 0xbce6, 0xbee8, 0xc0ea, 0xc2ec},
       {0xc4ee, 0xc6f0, 0xc8f2, 0xcaf4, 0xccf6, 0xcef8, 0xd0fa, 0xd2fc},
       {0xd4fe, 0xd700, 0xd902, 0xdb04, 0xdd06, 0xdf08, 0xe10a, 0xe30c}},
      {{0x113b'648e, 0x153f'6892, 0x1943'6c96, 0x1d47'709a},
       {0x214b'749e, 0x254f'78a2, 0x2953'7ca6, 0x2d57'80aa},
       {0x315b'84ae, 0x355f'88b2, 0x3963'8cb6, 0x3d67'90ba},
       {0x416b'94be, 0x456f'98c2, 0x4973'9cc6, 0x4d77'a0ca},
       {0x517b'a4ce, 0x557f'a8d2, 0x5983'acd6, 0x5d87'b0da},
       {0x618b'b4de, 0x658f'b8e2, 0x6993'bce6, 0x6d97'c0ea},
       {0x719b'c4ee, 0x759f'c8f2, 0x79a3'ccf6, 0x7da7'd0fa},
       {0x81ab'd4fe, 0x85af'd902, 0x89b3'dd06, 0x8db7'e10a}},
      {{0x6a94'bde8'113b'648e, 0x729c'c5f0'1943'6c96},
       {0x7aa4'cdf8'214b'749e, 0x82ac'd600'2953'7ca6},
       {0x8ab4'de08'315b'84ae, 0x92bc'e610'3963'8cb6},
       {0x9ac4'ee18'416b'94be, 0xa2cc'f620'4973'9cc6},
       {0xaad4'fe28'517b'a4ce, 0xb2dd'0630'5983'acd6},
       {0xbae5'0e38'618b'b4de, 0xc2ed'1640'6993'bce6},
       {0xcaf5'1e48'719b'c4ee, 0xd2fd'2650'79a3'ccf6},
       {0xdb05'2e58'81ab'd4fe, 0xe30d'3660'89b3'dd06}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVminu) {
  TestVectorInstruction(
      0x110c0457,  // vminu.vv v8,v16,v24,v0.t
      {{0, 2, 2, 6, 4, 10, 6, 14, 8, 18, 10, 22, 12, 26, 14, 30},
       {16, 34, 18, 38, 20, 42, 22, 46, 24, 50, 26, 54, 28, 58, 30, 62},
       {32, 66, 34, 70, 36, 74, 38, 78, 40, 82, 42, 86, 44, 90, 46, 94},
       {48, 98, 50, 102, 52, 106, 54, 110, 56, 114, 58, 118, 60, 122, 62, 126},
       {64, 130, 66, 134, 68, 138, 70, 142, 72, 146, 74, 150, 76, 154, 78, 158},
       {80, 162, 82, 166, 84, 170, 86, 174, 88, 178, 90, 182, 92, 186, 94, 190},
       {96, 194, 98, 198, 100, 202, 102, 206, 104, 210, 106, 214, 108, 218, 110, 222},
       {112, 226, 114, 230, 116, 234, 118, 238, 120, 242, 122, 246, 124, 250, 126, 254}},
      {{0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc},
       {0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc}},
      {{0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
       {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
       {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8}},
      {{0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
       {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
       {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
       {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
       {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
       {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
       {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
       {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x1100c457,  // vminu.vx v8,v16,x1,v0.t
      {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
       {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
       {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 170, 44, 170, 46, 170},
       {48, 170, 50, 170, 52, 170, 54, 170, 56, 170, 58, 170, 60, 170, 62, 170},
       {64, 170, 66, 170, 68, 170, 70, 170, 72, 170, 74, 170, 76, 170, 78, 170},
       {80, 170, 82, 170, 84, 170, 86, 170, 88, 170, 90, 170, 92, 170, 94, 170},
       {96, 170, 98, 170, 100, 170, 102, 170, 104, 170, 106, 170, 108, 170, 110, 170},
       {112, 170, 114, 170, 116, 170, 118, 170, 120, 170, 122, 170, 124, 170, 126, 170}},
      {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
       {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
       {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa}},
      {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
       {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
       {0xa322'a120, 0xa726'a524, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa}},
      {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
       {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
       {0xa726'a524'a322'a120, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVmin) {
  TestVectorInstruction(
      0x150c0457,  // vmin.vv v8,v16,v24,v0.t
      {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
       {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
       {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 171, 44, 173, 46, 175},
       {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191},
       {128, 130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156, 158},
       {160, 162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188, 190},
       {192, 194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220, 222},
       {224, 226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252, 254}},
      {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
       {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
       {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
       {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc},
       {0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc}},
      {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
       {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
       {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
       {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
       {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
       {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8}},
      {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
       {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
       {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
       {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
       {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
       {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
       {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
       {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x1500c457,  // vmin.vx v8,v16,ra,v0.t
      {{170, 129, 170, 131, 170, 133, 170, 135, 170, 137, 170, 139, 170, 141, 170, 143},
       {170, 145, 170, 147, 170, 149, 170, 151, 170, 153, 170, 155, 170, 157, 170, 159},
       {170, 161, 170, 163, 170, 165, 170, 167, 170, 169, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170}},
      {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
       {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
       {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa}},
      {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
       {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
       {0xa322'a120, 0xa726'a524, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa}},
      {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
       {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
       {0xa726'a524'a322'a120, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVmaxu) {
  TestVectorInstruction(
      0x190c0457,  // vmaxu.vv v8,v16,v24,v0.t
      {{0, 129, 4, 131, 9, 133, 12, 135, 17, 137, 20, 139, 24, 141, 28, 143},
       {32, 145, 36, 147, 41, 149, 44, 151, 49, 153, 52, 155, 56, 157, 60, 159},
       {64, 161, 68, 163, 73, 165, 76, 167, 81, 169, 84, 171, 88, 173, 92, 175},
       {96, 177, 100, 179, 105, 181, 108, 183, 113, 185, 116, 187, 120, 189, 124, 191},
       {128, 193, 132, 195, 137, 197, 140, 199, 145, 201, 148, 203, 152, 205, 156, 207},
       {160, 209, 164, 211, 169, 213, 172, 215, 177, 217, 180, 219, 184, 221, 188, 223},
       {192, 225, 196, 227, 201, 229, 204, 231, 209, 233, 212, 235, 216, 237, 220, 239},
       {224, 241, 228, 243, 233, 245, 236, 247, 241, 249, 244, 251, 248, 253, 252, 255}},
      {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
       {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
       {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
       {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
       {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
       {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
       {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
       {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
      {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
       {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
       {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
       {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
       {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
       {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
       {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
       {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
      {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
       {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
       {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
       {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
       {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
       {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
       {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
       {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x1900c457,  // vmaxu.vx v8,v16,ra,v0.t
      {{170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 171, 170, 173, 170, 175},
       {170, 177, 170, 179, 170, 181, 170, 183, 170, 185, 170, 187, 170, 189, 170, 191},
       {170, 193, 170, 195, 170, 197, 170, 199, 170, 201, 170, 203, 170, 205, 170, 207},
       {170, 209, 170, 211, 170, 213, 170, 215, 170, 217, 170, 219, 170, 221, 170, 223},
       {170, 225, 170, 227, 170, 229, 170, 231, 170, 233, 170, 235, 170, 237, 170, 239},
       {170, 241, 170, 243, 170, 245, 170, 247, 170, 249, 170, 251, 170, 253, 170, 255}},
      {{0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xab2a, 0xad2c, 0xaf2e},
       {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
       {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
       {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
       {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
       {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
      {{0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xab2a'a928, 0xaf2e'ad2c},
       {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
       {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
       {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
       {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
       {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
      {{0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaf2e'ad2c'ab2a'a928},
       {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
       {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
       {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
       {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
       {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVmax) {
  TestVectorInstruction(
      0x1d0c0457,  // vmax.vv v8,v16,v24,v0.t
      {{0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62},
       {64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94},
       {96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126},
       {64, 193, 66, 195, 68, 197, 70, 199, 72, 201, 74, 203, 76, 205, 78, 207},
       {80, 209, 82, 211, 84, 213, 86, 215, 88, 217, 90, 219, 92, 221, 94, 223},
       {96, 225, 98, 227, 100, 229, 102, 231, 104, 233, 106, 235, 108, 237, 110, 239},
       {112, 241, 114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255}},
      {{0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
       {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
       {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
       {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
      {{0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
       {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
       {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
       {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
      {{0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
       {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
       {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
       {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
       {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
       {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
       {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
       {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x1d00c457,  // vmax.vx v8,v16,ra,v0.t
      {{0, 170, 2, 170, 4, 170, 6, 170, 8, 170, 10, 170, 12, 170, 14, 170},
       {16, 170, 18, 170, 20, 170, 22, 170, 24, 170, 26, 170, 28, 170, 30, 170},
       {32, 170, 34, 170, 36, 170, 38, 170, 40, 170, 42, 171, 44, 173, 46, 175},
       {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191},
       {64, 193, 66, 195, 68, 197, 70, 199, 72, 201, 74, 203, 76, 205, 78, 207},
       {80, 209, 82, 211, 84, 213, 86, 215, 88, 217, 90, 219, 92, 221, 94, 223},
       {96, 225, 98, 227, 100, 229, 102, 231, 104, 233, 106, 235, 108, 237, 110, 239},
       {112, 241, 114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255}},
      {{0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xab2a, 0xad2c, 0xaf2e},
       {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
       {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
       {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
       {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
       {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
      {{0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xab2a'a928, 0xaf2e'ad2c},
       {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
       {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
       {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
       {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
       {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
      {{0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaf2e'ad2c'ab2a'a928},
       {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
       {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
       {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
       {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
       {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVredsum) {
  TestVectorReductionInstruction(
      0x10c2457,  // vredsum.vs v8,v16,v24,v0.t
      /* expected_result_vd0_int8 */
      {242, 228, 200, 144, 0 /* unused */, 2, 12, 57},
      /* expected_result_vd0_int16 */
      {0x0172, 0x82e4, 0x88c8, 0xa090, 0x0000 /* unused */, 0x8300, 0x8904, 0xa119},
      /* expected_result_vd0_int32 */
      {0xcb42'b932, 0x9403'71e4, 0xa706'64c8, 0xd312'5090,
       0x0000'0000 /* unused */, 0x8906'8300, 0x8906'8300, 0x9712'8d09},
      /* expected_result_vd0_int64 */
      {0xb32e'a925'9f1a'9511, 0x1f97'0d86'fb72'e962, 0x0b928'970a'74e4'52c4,
       0xef4e'ad14'6aca'2888, 0x0000'0000'0000'000 /* unused */,
       0x9512'8f0d'8906'8300, 0x9512'8f0d'8906'8300, 0x9512'8f0d'8906'8300},
      /* expected_result_vd0_with_mask_int8 */
      {151, 104, 222, 75, 0 /* unused */, 0, 10, 34},
      /* expected_result_vd0_with_mask_int16 */
      {0xcf45, 0xc22f, 0x79d0, 0x98bf, 0x0000 /* unused */, 0x8300, 0x8300, 0x9b15},
      /* expected_result_vd0_with_mask_int32 */
      {0xbd36'af29, 0x299f'138a, 0x1984'ef5c, 0x9cf4'4aa1,
       0x0000'0000 /* unused */, 0x8906'8300, 0x08906'8300, 0x8906'8300},
      /* expected_result_vd0_with_mask_int64 */
      {0x9512'8f0d'8906'8300, 0x17a'f36e'e55e'd751, 0xde53'c83f'b227'9c13,
       0xc833'9e0e'73df'49b5, 0x0000'0000'0000'0000 /* unused */,
       0x9512'8f0d'8906'8300, 0x9512'8f0d'8906'8300, 0x9512'8f0d'8906'8300},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVredand) {
  TestVectorReductionInstruction(
      0x50c2457,  // vredand.vs v8,v16,v24,v0.t
      /* expected_result_vd0_int8 */
      {0, 0, 0, 0, 0, 0, 0, 0},
      /* expected_result_vd0_int16 */
      {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
      /* expected_result_vd0_int32 */
      {0x0200'0000, 0x0200'0000, 0x0200'0000, 0x0200'0000, 0x0, 0x0200'0000, 0x0200'0000, 0x0200'0000},
      /* expected_result_vd0_int64 */
      {0x0604'0000'0200'0000, 0x0604'0000'0200'0000, 0x0604'0000'0200'0000, 0x0604'0000'0200'0000, 0x0,
       0x0604'0000'0200'0000, 0x0604'0000'0200'0000, 0x0604'0000'0200'0000},
      /* expected_result_vd0_with_mask_int8 */
      {0, 0, 0, 0, 0, 0, 0, 0},
      /* expected_result_vd0_with_mask_int16 */
      {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
      /* expected_result_vd0_with_mask_int32 */
      {0x2000000, 0x2000000, 0x2000000, 0x2000000, 0x0, 0x2000000, 0x2000000, 0x2000000},
      /* expected_result_vd0_with_mask_int64 */
      {0x0604'0000'0200'0000, 0x0604'0000'0200'0000, 0x0604'0000'0200'0000, 0x0604'0000'0200'0000, 0x0,
       0x0604'0000'0200'0000, 0x0604'0000'0200'0000, 0x0604'0000'0200'0000},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVredor) {
  TestVectorReductionInstruction(
      0x90c2457,  // vredor.vs v8,v16,v24,v0.t
      /* expected_result_vd0_int8 */
      {31, 63, 127, 255, 0, 2, 6, 15},
      /* expected_result_vd0_int16 */
      {0x9f1d, 0xbf3d, 0xff7d, 0xfffd, 0x0, 0x8300, 0x8704, 0x8f0d},
      /* expected_result_vd0_int32 */
      {0x9f1e'9b19, 0xbf3e'bb39, 0xff7e'fb79, 0xfffe'fbf9, 0x0, 0x8706'8300, 0x8706'8300, 0x8f0e'8b09},
      /* expected_result_vd0_int64 */
      {0x9f1e'9f1d'9716'9311, 0xbf3e'bf3d'b736'b331, 0xff7e'ff7d'f776'f371, 0xfffefffdf7f6f3f1, 0x0,
       0x8f0e'8f0d'8706'8300, 0x8f0e'8f0d'8706'8300, 0x8f0e'8f0d'8706'8300},
      /* expected_result_vd0_with_mask_int8 */
      {31, 63, 127, 255, 0, 0, 6, 14},
      /* expected_result_vd0_with_mask_int16 */
      {0x9f1d, 0xbf3d, 0xff7d, 0xfffd, 0x0, 0x8300, 0x8300, 0x8f0d},
      /* expected_result_vd0_with_mask_int32 */
      {0x9f1e'9b19, 0xbf3e'bb39, 0xff7e'fb79, 0xfffe'fbf9, 0x0, 0x8706'8300, 0x8706'8300, 0x8706'8300},
      /* expected_result_vd0_with_mask_int64 */
      {0x8f0e'8f0d'8706'8300, 0xbf3e'bf3d'b736'b331, 0xff7e'ff7d'f776'f371, 0xfffe'fffd'f7f6'f3f1, 0x0,
       0x8f0e'8f0d'8706'8300, 0x8f0e'8f0d'8706'8300, 0x8f0e'8f0d'8706'8300},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVredxor) {
  TestVectorReductionInstruction(
      0xd0c2457,  // vredxor.vs v8,v16,v24,v0.t
      /* expected_result_vd0_int8 */
      {0, 0, 0, 0, 0, 2, 0, 1},
      /* expected_result_vd0_int16 */
      {0x8100, 0x8100, 0x8100, 0x8100, 0x0, 0x8300, 0x8504, 0x8101},
      /* expected_result_vd0_int32 */
      {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100, 0x0, 0x8506'8300, 0x8506'8300, 0x8b0a'8909},
      /* expected_result_vd0_int64 */
      {0x9716'9515'9312'9111, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x0,
       0x890a'8f0d'8506'8300, 0x890a'8f0d'8506'8300, 0x890a'8f0d'8506'8300},
      /* expected_result_vd0_with_mask_int8 */
      {31, 10, 6, 187, 0, 0, 2, 6},
      /* expected_result_vd0_with_mask_int16 */
      {0x8f0d, 0xbd3d, 0x9514, 0x8d0d, 0x0, 0x8300, 0x8300, 0x8705},
      /* expected_result_vd0_with_mask_int32 */
      {0x8d0e'8b09, 0x9d1e'9b18, 0xfb7a'f978, 0xab2a'a929, 0x0, 0x8506'8300, 0x8506'8300, 0x8506'8300},
      /* expected_result_vd0_with_mask_int64 */
      {0x890a'8f0d'8506'8300, 0x991a'9f1c'9516'9311, 0xb93a'bf3c'b536'b331, 0x77f6'75f5'73f2'71f1, 0x0,
       0x890a'8f0d'8506'8300, 0x890a'8f0d'8506'8300, 0x890a'8f0d'8506'8300},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVredminu) {
  TestVectorReductionInstruction(
      0x110c2457,  // vredminu.vs v8,v16,v24,v0.t
      /* expected_result_vd0_int8 */
      {0, 0, 0, 0, 0, 0, 0, 0},
      /* expected_result_vd0_int16 */
      {0x200, 0x200, 0x200, 0x200, 0x0, 0x200, 0x200, 0x200},
      /* expected_result_vd0_int32 */
      {0x0604'0200, 0x0604'0200, 0x0604'0200, 0x0604'0200, 0x0, 0x0604'0200, 0x0604'0200, 0x0604'0200},
      /* expected_result_vd0_int64 */
      {0x0e0c'0a09'0604'0200, 0x0e0c'0a09'0604'0200, 0x0e0c'0a09'0604'0200, 0x0e0c'0a09'0604'0200, 0x0,
       0x0e0c'0a09'0604'0200, 0x0e0c'0a09'0604'0200, 0x0e0c'0a09'0604'0200},
      /* expected_result_vd0_with_mask_int8 */
      {0, 0, 0, 0, 0, 0, 0, 0},
      /* expected_result_vd0_with_mask_int16 */
      {0x200, 0x200, 0x200, 0x200, 0x0, 0x200, 0x200, 0x200},
      /* expected_result_vd0_with_mask_int32 */
      {0x0604'0200, 0x0604'0200, 0x0604'0200, 0x0604'0200, 0x0, 0x0604'0200, 0x0604'0200, 0x0604'0200},
      /* expected_result_vd0_with_mask_int64 */
      {0x0e0c'0a09'0604'0200, 0x0e0c'0a09'0604'0200, 0x0e0c'0a09'0604'0200, 0x0e0c'0a09'0604'0200, 0x0,
       0x0e0c'0a09'0604'0200, 0x0e0c'0a09'0604'0200, 0x0e0c'0a09'0604'0200},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVredmin) {
  TestVectorReductionInstruction(
      0x150c2457,  // vredmin.vs v8,v16,v24,v0.t
      /* expected_result_vd0_int8 */
      {0, 0, 0, 128, 0, 0, 0, 0},
      /* expected_result_vd0_int16 */
      {0x8100, 0x8100, 0x8100, 0x8100, 0x0, 0x8100, 0x8100, 0x8100},
      /* expected_result_vd0_int32 */
      {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100, 0x0, 0x8302'8100, 0x8302'8100, 0x8302'8100},
      /* expected_result_vd0_int64 */
      {0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x0,
       0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
      /* expected_result_vd0_with_mask_int8 */
      {0, 0, 0, 128, 0, 0, 0, 0},
      /* expected_result_vd0_with_mask_int16 */
      {0x8100, 0x8100, 0x8100, 0x8100, 0x0, 0x8100, 0x8100, 0x8100},
      /* expected_result_vd0_with_mask_int32 */
      {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100, 0x0, 0x8302'8100, 0x8302'8100, 0x8302'8100},
      /* expected_result_vd0_with_mask_int64 */
      {0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x0,
       0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVredmaxu) {
  TestVectorReductionInstruction(
      0x190c2457,  // vredmaxu.vs v8,v16,v24,v0.t
      /* expected_result_vd0_int8 */
      {30, 62, 126, 254, 0, 2, 6, 14},
      /* expected_result_vd0_int16 */
      {0x8100, 0x8100, 0x8100, 0xfefc, 0x0, 0x8100, 0x8100, 0x8100},
      /* expected_result_vd0_int32 */
      {0x8302'8100, 0x8302'8100, 0x8302'8100, 0xfefc'faf8, 0x0, 0x8302'8100, 0x8302'8100, 0x8302'8100},
      /* expected_result_vd0_int64 */
      {0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0xfefc'faf8'f6f4'f2f1, 0x0,
       0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
      /* expected_result_vd0_with_mask_int8 */
      {30, 62, 126, 252, 0, 0, 6, 14},
      /* expected_result_vd0_with_mask_int16 */
      {0x8100, 0x8100, 0x8100, 0xfefc, 0x0, 0x8100, 0x8100, 0x8100},
      /* expected_result_vd0_with_mask_int32 */
      {0x8302'8100, 0x8302'8100, 0x8302'8100, 0xfefc'faf8, 0x0, 0x8302'8100, 0x8302'8100, 0x8302'8100},
      /* expected_result_vd0_with_mask_int64 */
      {0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0xfefc'faf8'f6f4'f2f1, 0x0,
       0x8706'8504'8302'8100, 0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVredmax) {
  TestVectorReductionInstruction(
      0x1d0c2457,  // vredmax.vs v8,v16,v24,v0.t
      /* expected_result_vd0_int8 */
      {30, 62, 126, 126, 0, 2, 6, 14},
      /* expected_result_vd0_int16 */
      {0x1e1c, 0x3e3c, 0x7e7c, 0x7e7c, 0x0, 0x200, 0x604, 0xe0c},
      /* expected_result_vd0_int32 */
      {0x1e1c1a18, 0x3e3c3a38, 0x7e7c7a78, 0x7e7c7a78, 0x0, 0x6040200, 0x6040200, 0xe0c0a09},
      /* expected_result_vd0_int64 */
      {0x1e1c1a1816141211, 0x3e3c3a3836343231, 0x7e7c7a7876747271, 0x7e7c7a7876747271, 0x0,
       0xe0c0a0906040200, 0xe0c0a0906040200, 0xe0c0a0906040200},
      /* expected_result_vd0_with_mask_int8 */
      {30, 62, 126, 126, 0, 0, 6, 14},
      /* expected_result_vd0_with_mask_int16 */
      {0x1e1c, 0x3e3c, 0x7e7c, 0x7e7c, 0x0, 0x200, 0x200, 0xe0c},
      /* expected_result_vd0_with_mask_int32 */
      {0x1e1c1a18, 0x3e3c3a38, 0x7e7c7a78, 0x7e7c7a78, 0x0, 0x6040200, 0x6040200, 0x6040200},
      /* expected_result_vd0_with_mask_int64 */
      {0xe0c0a0906040200, 0x3e3c3a3836343231, 0x7e7c7a7876747271, 0x7e7c7a7876747271, 0x0,
       0xe0c0a0906040200, 0xe0c0a0906040200, 0xe0c0a0906040200},
      kVectorCalculationsSource);
}

// Note that these expected test outputs for Vmerge are identical to those for Vmv. The difference
// between Vmerge and Vmv is captured in masking logic within TestVectorInstruction itself via the
// parameter expect_inactive_equals_vs2=true for Vmerge.
TEST_F(Riscv64InterpreterTest, TestVmerge) {
  TestVectorInstruction(
      0x5d0c0457,  // Vmerge.vvm v8, v16, v24, v0
      {{0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62},
       {64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94},
       {96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126},
       {128, 130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156, 158},
       {160, 162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188, 190},
       {192, 194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220, 222},
       {224, 226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252, 254}},
      {{0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc},
       {0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc}},
      {{0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
       {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
       {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8}},
      {{0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
       {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
       {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
       {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
       {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
       {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
       {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
       {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
      kVectorCalculationsSource,
      /*expect_inactive_equals_vs2=*/true);
  TestVectorInstruction(
      0x5d00c457,  // Vmerge.vxm v8, v16, x1, v0
      {{170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170}},
      {{0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa}},
      {{0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa}},
      {{0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa}},
      kVectorCalculationsSource,
      /*expect_inactive_equals_vs2=*/true);
  TestVectorInstruction(
      0x5d0ab457,  // Vmerge.vim v8, v16, -0xb, v0
      {{245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245}},
      {{0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5}},
      {{0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5}},
      {{0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5}},
      kVectorCalculationsSource,
      /*expect_inactive_equals_vs2=*/true);
}

TEST_F(Riscv64InterpreterTest, TestVmv) {
  TestVectorInstruction(
      0x5e0c0457,  // Vmv.v.v v8, v24
      {{0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62},
       {64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94},
       {96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126},
       {128, 130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156, 158},
       {160, 162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188, 190},
       {192, 194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220, 222},
       {224, 226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252, 254}},
      {{0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc},
       {0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc}},
      {{0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
       {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
       {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8}},
      {{0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
       {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
       {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
       {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
       {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
       {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
       {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
       {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x5e00c457,  // Vmv.v.x v8, x1
      {{170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170},
       {170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170}},
      {{0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa}},
      {{0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa, 0xaaaa'aaaa}},
      {{0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa},
       {0xaaaa'aaaa'aaaa'aaaa, 0xaaaa'aaaa'aaaa'aaaa}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x5e0ab457,  // Vmv.v.i v8, -0xb
      {{245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245},
       {245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245}},
      {{0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5}},
      {{0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5},
       {0xffff'fff5, 0xffff'fff5, 0xffff'fff5, 0xffff'fff5}},
      {{0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5},
       {0xffff'ffff'ffff'fff5, 0xffff'ffff'ffff'fff5}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVmul) {
  TestVectorInstruction(
      0x950c2457,  // vmul.vv v8, v16, v24, v0.t
       {{0, 2, 8, 18, 36, 50, 72, 98, 136, 162, 200, 242, 32, 82, 136, 194},
       {0, 66, 136, 210, 52, 114, 200, 34, 152, 226, 72, 178, 32, 146, 8, 130},
       {0, 130, 8, 146, 68, 178, 72, 226, 168, 34, 200, 114, 32, 210, 136, 66},
       {0, 194, 136, 82, 84, 242, 200, 162, 184, 98, 72, 50, 32, 18, 8, 2},
       {0, 2, 8, 18, 100, 50, 72, 98, 200, 162, 200, 242, 32, 82, 136, 194},
       {0, 66, 136, 210, 116, 114, 200, 34, 216, 226, 72, 178, 32, 146, 8, 130},
       {0, 130, 8, 146, 132, 178, 72, 226, 232, 34, 200, 114, 32, 210, 136, 66},
       {0, 194, 136, 82, 148, 242, 200, 162, 248, 98, 72, 50, 32, 18, 8, 2}},
      {{0x0000, 0x1808, 0xd524, 0xa848, 0xa988, 0xb8c8, 0x7120, 0x4988},
       {0x4200, 0x5a88, 0x2834, 0xebc8, 0xfd98, 0xfd48, 0xb620, 0x8f08},
       {0x8800, 0xa108, 0x7f44, 0x3348, 0x55a8, 0x45c8, 0xff20, 0xd888},
       {0xd200, 0xeb88, 0xda54, 0x7ec8, 0xb1b8, 0x9248, 0x4c20, 0x2608},
       {0x2000, 0x3a08, 0x3964, 0xce48, 0x11c8, 0xe2c8, 0x9d20, 0x7788},
       {0x7200, 0x8c88, 0x9c74, 0x21c8, 0x75d8, 0x3748, 0xf220, 0xcd08},
       {0xc800, 0xe308, 0x0384, 0x7948, 0xdde8, 0x8fc8, 0x4b20, 0x2688},
       {0x2200, 0x3d88, 0x6e94, 0xd4c8, 0x49f8, 0xec48, 0xa820, 0x8408}},
      {{0x0902'0000, 0x749c'd524, 0x5df5'a988, 0xb900'7120},
       {0x9fd6'4200, 0x1e83'2834, 0x0add'fd98, 0x58da'b620},
       {0x42b2'8800, 0xd471'7f44, 0xc3ce'55a8, 0x04bc'ff20},
       {0xf196'd200, 0x9667'da54, 0x88c6'b1b8, 0xbca7'4c20},
       {0xac83'2000, 0x6466'3964, 0x59c7'11c8, 0x8099'9d20},
       {0x7377'7200, 0x3e6c'9c74, 0x36cf'75d8, 0x5093'f220},
       {0x4673'c800, 0x247b'0384, 0x1fdf'dde8, 0x2c96'4b20},
       {0x2578'2200, 0x1691'6e94, 0x14f8'49f8, 0x14a0'a820}},
      {{0xfc4e'ad16'0902'0000, 0xa697'acf5'5df5'a988},
       {0x4fde'a9cf'9fd6'4200, 0x0833'b3b7'0add'fd98},
       {0xbf86'ba99'42b2'8800, 0x85e7'ce88'c3ce'55a8},
       {0x4b46'df72'f196'd200, 0x1fb3'fd6a'88c6'b1b8},
       {0xf31f'185c'ac83'2000, 0xd598'405c'59c7'11c8},
       {0xb70f'6556'7377'7200, 0xa794'975e'36cf'75d8},
       {0x9717'c660'4673'c800, 0x95a9'0270'1fdf'dde8},
       {0x9338'3b7a'2578'2200, 0x9fd5'8192'14f8'49f8}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x9500e457,  // vmul.vx v8, v16, x1, v0.t
      {{0, 170, 84, 254, 168, 82, 252, 166, 80, 250, 164, 78, 248, 162, 76, 246},
       {160, 74, 244, 158, 72, 242, 156, 70, 240, 154, 68, 238, 152, 66, 236, 150},
       {64, 234, 148, 62, 232, 146, 60, 230, 144, 58, 228, 142, 56, 226, 140, 54},
       {224, 138, 52, 222, 136, 50, 220, 134, 48, 218, 132, 46, 216, 130, 44, 214},
       {128, 42, 212, 126, 40, 210, 124, 38, 208, 122, 36, 206, 120, 34, 204, 118},
       {32, 202, 116, 30, 200, 114, 28, 198, 112, 26, 196, 110, 24, 194, 108, 22},
       {192, 106, 20, 190, 104, 18, 188, 102, 16, 186, 100, 14, 184, 98, 12, 182},
       {96, 10, 180, 94, 8, 178, 92, 6, 176, 90, 4, 174, 88, 2, 172, 86}},
      {{0xaa00, 0x5354, 0xfca8, 0xa5fc, 0x4f50, 0xf8a4, 0xa1f8, 0x4b4c},
       {0xf4a0, 0x9df4, 0x4748, 0xf09c, 0x99f0, 0x4344, 0xec98, 0x95ec},
       {0x3f40, 0xe894, 0x91e8, 0x3b3c, 0xe490, 0x8de4, 0x3738, 0xe08c},
       {0x89e0, 0x3334, 0xdc88, 0x85dc, 0x2f30, 0xd884, 0x81d8, 0x2b2c},
       {0xd480, 0x7dd4, 0x2728, 0xd07c, 0x79d0, 0x2324, 0xcc78, 0x75cc},
       {0x1f20, 0xc874, 0x71c8, 0x1b1c, 0xc470, 0x6dc4, 0x1718, 0xc06c},
       {0x69c0, 0x1314, 0xbc68, 0x65bc, 0x0f10, 0xb864, 0x61b8, 0x0b0c},
       {0xb460, 0x5db4, 0x0708, 0xb05c, 0x59b0, 0x0304, 0xac58, 0x55ac}},
      {{0x5353'aa00, 0xfb50'fca8, 0xa34e'4f50, 0x4b4b'a1f8},
       {0xf348'f4a0, 0x9b46'4748, 0x4343'99f0, 0xeb40'ec98},
       {0x933e'3f40, 0x3b3b'91e8, 0xe338'e490, 0x8b36'3738},
       {0x3333'89e0, 0xdb30'dc88, 0x832e'2f30, 0x2b2b'81d8},
       {0xd328'd480, 0x7b26'2728, 0x2323'79d0, 0xcb20'cc78},
       {0x731e'1f20, 0x1b1b'71c8, 0xc318'c470, 0x6b16'1718},
       {0x1313'69c0, 0xbb10'bc68, 0x630e'0f10, 0x0b0b'61b8},
       {0xb308'b460, 0x5b06'0708, 0x0303'59b0, 0xab00'ac58}},
      {{0xa5fb'a752'5353'aa00, 0x4b4b'a1f7'a34e'4f50},
       {0xf09b'9c9c'f348'f4a0, 0x95eb'9742'4343'99f0},
       {0x3b3b'91e7'933e'3f40, 0xe08b'8c8c'e338'e490},
       {0x85db'8732'3333'89e0, 0x2b2b'81d7'832e'2f30},
       {0xd07b'7c7c'd328'd480, 0x75cb'7722'2323'79d0},
       {0x1b1b'71c7'731e'1f20, 0xc06b'6c6c'c318'c470},
       {0x65bb'6712'1313'69c0, 0x0b0b'61b7'630e'0f10},
       {0xb05b'5c5c'b308'b460, 0x55ab'5702'0303'59b0}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVmulh) {
  TestVectorInstruction(
      0x9d0c2457,  // vmulh.vv v8, v16, v24, v0.t
      {{0, 255, 0, 253, 0, 251, 0, 249, 0, 247, 0, 245, 1, 244, 1, 242},
       {2, 241, 2, 239, 3, 238, 3, 237, 4, 235, 5, 234, 6, 233, 7, 232},
       {8, 231, 9, 230, 10, 229, 11, 228, 12, 228, 13, 227, 15, 226, 16, 226},
       {18, 225, 19, 225, 21, 224, 22, 224, 24, 224, 26, 224, 28, 224, 30, 224},
       {224, 31, 224, 29, 224, 27, 224, 25, 224, 23, 224, 21, 225, 20, 225, 18},
       {226, 17, 226, 15, 227, 14, 227, 13, 228, 11, 229, 10, 230, 9, 231, 8},
       {232, 7, 233, 6, 234, 5, 235, 4, 236, 4, 237, 3, 239, 2, 240, 2},
       {242, 1, 243, 1, 245, 0, 246, 0, 248, 0, 250, 0, 252, 0, 254, 0}},
      {{0xff02, 0xfd10, 0xfb2d, 0xf95c, 0xf79a, 0xf5e9, 0xf448, 0xf2b7},
       {0xf136, 0xefc5, 0xee64, 0xed13, 0xebd2, 0xeaa2, 0xe982, 0xe872},
       {0xe772, 0xe682, 0xe5a2, 0xe4d3, 0xe413, 0xe364, 0xe2c4, 0xe235},
       {0xe1b6, 0xe147, 0xe0e8, 0xe09a, 0xe05b, 0xe02d, 0xe00f, 0xe001},
       {0x1ec3, 0x1cd3, 0x1af3, 0x1923, 0x1764, 0x15b4, 0x1415, 0x1286},
       {0x1107, 0x0f98, 0x0e39, 0x0ceb, 0x0bac, 0x0a7e, 0x095f, 0x0851},
       {0x0753, 0x0665, 0x0588, 0x04ba, 0x03fc, 0x034f, 0x02b2, 0x0225},
       {0x01a8, 0x013b, 0x00de, 0x0091, 0x0055, 0x0028, 0x000c, 0x0000}},
      {{0xfd10'1a16, 0xf95c'aad6, 0xf5e9'bc58, 0xf2b7'4e9b},
       {0xefc5'619f, 0xed13'f564, 0xeaa3'09ea, 0xe872'9f31},
       {0xe682'b539, 0xe4d3'4c01, 0xe364'638b, 0xe235'fbd7},
       {0xe148'14e2, 0xe09a'aeaf, 0xe02d'c93d, 0xe001'648c},
       {0x1cd2'bf5c, 0x1923'5829, 0x15b4'71b7, 0x1286'0c06},
       {0x0f98'2716, 0x0cea'c2e7, 0x0a7d'df79, 0x0851'7ccc},
       {0x0665'9ae0, 0x04ba'39b5, 0x034f'594b, 0x0224'f9a2},
       {0x013b'1aba, 0x0091'bc93, 0x0028'df2d, 0x0000'8288}},
      {{0xf95c'aad6'78f5'63b8, 0xf2b7'4e9b'bf9d'55cb},
       {0xed13'f564'2968'6900, 0xe872'9f31'6a0c'5913},
       {0xe4d3'4c01'edf3'8a67, 0xe235'fbd7'2893'787a},
       {0xe09a'aeaf'c696'c7ef, 0xe001'648c'fb32'b402},
       {0x1923'5828'f00f'6056, 0x1286'0c06'169f'4261},
       {0x0cea'c2e6'e0d2'c60e, 0x0851'7ccc'015e'a619},
       {0x04ba'39b4'e5ae'47e6, 0x0224'f9a2'0036'25f1},
       {0x0091'bc92'fea1'e5de, 0x0000'8288'1325'c1e9}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x9d00e457,  // vmulh.vx v8, v16, x1, v0.t
      {{0, 42, 255, 41, 254, 41, 253, 40, 253, 39, 252, 39, 251, 38, 251, 37},
       {250, 37, 249, 36, 249, 35, 248, 35, 247, 34, 247, 33, 246, 33, 245, 32},
       {245, 31, 244, 31, 243, 30, 243, 29, 242, 29, 241, 28, 241, 27, 240, 27},
       {239, 26, 239, 25, 238, 25, 237, 24, 237, 23, 236, 23, 235, 22, 235, 21},
       {234, 21, 233, 20, 233, 19, 232, 19, 231, 18, 231, 17, 230, 17, 229, 16},
       {229, 15, 228, 15, 227, 14, 227, 13, 226, 13, 225, 12, 225, 11, 224, 11},
       {223, 10, 223, 9, 222, 9, 221, 8, 221, 7, 220, 7, 219, 6, 219, 5},
       {218, 5, 217, 4, 217, 3, 216, 3, 215, 2, 215, 1, 214, 1, 213, 0}},
      {{0x2a55, 0x29aa, 0x28fe, 0x2853, 0x27a8, 0x26fc, 0x2651, 0x25a6},
       {0x24fa, 0x244f, 0x23a4, 0x22f8, 0x224d, 0x21a2, 0x20f6, 0x204b},
       {0x1fa0, 0x1ef4, 0x1e49, 0x1d9e, 0x1cf2, 0x1c47, 0x1b9c, 0x1af0},
       {0x1a45, 0x199a, 0x18ee, 0x1843, 0x1798, 0x16ec, 0x1641, 0x1596},
       {0x14ea, 0x143f, 0x1394, 0x12e8, 0x123d, 0x1192, 0x10e6, 0x103b},  // NOTYPO
       {0x0f90, 0x0ee4, 0x0e39, 0x0d8e, 0x0ce2, 0x0c37, 0x0b8c, 0x0ae0},
       {0x0a35, 0x098a, 0x08de, 0x0833, 0x0788, 0x06dc, 0x0631, 0x0586},
       {0x04da, 0x042f, 0x0384, 0x02d8, 0x022d, 0x0182, 0x00d6, 0x002b}},
      {{0x29a9'd500, 0x2853'28fe, 0x26fc'7cfd, 0x25a5'd0fc},
       {0x244f'24fa, 0x22f8'78f9, 0x21a1'ccf8, 0x204b'20f6},
       {0x1ef4'74f5, 0x1d9d'c8f4, 0x1c47'1cf2, 0x1af0'70f1},
       {0x1999'c4f0, 0x1843'18ee, 0x16ec'6ced, 0x1595'c0ec},
       {0x143f'14ea, 0x12e8'68e9, 0x1191'bce8, 0x103b'10e6},  // NOTYPO
       {0x0ee4'64e5, 0x0d8d'b8e4, 0x0c37'0ce2, 0x0ae0'60e1},
       {0x0989'b4e0, 0x0833'08de, 0x06dc'5cdd, 0x0585'b0dc},
       {0x042f'04da, 0x02d8'58d9, 0x0181'acd8, 0x002b'00d6}},
      {{0x2853'28fe'7eff'2a55, 0x25a5'd0fb'd1a7'27a8},
       {0x22f8'78f9'244f'24fa, 0x204b'20f6'76f7'224d},
       {0x1d9d'c8f3'c99f'1fa0, 0x1af0'70f1'1c47'1cf2},
       {0x1843'18ee'6eef'1a45, 0x1595'c0eb'c197'1798},
       {0x12e8'68e9'143f'14ea, 0x103b'10e6'66e7'123d},  // NOTYPO
       {0x0d8d'b8e3'b98f'0f90, 0x0ae0'60e1'0c37'0ce2},
       {0x0833'08de'5edf'0a35, 0x0585'b0db'b187'0788},
       {0x02d8'58d9'042f'04da, 0x002b'00d6'56d7'022d}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVmulhu) {
  TestVectorInstruction(
      0x910c2457,  // vmulhu.vv v8, v16, v24, v0.t
      {{0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 1, 14, 1, 16},
       {2, 19, 2, 21, 3, 24, 3, 27, 4, 29, 5, 32, 6, 35, 7, 38},
       {8, 41, 9, 44, 10, 47, 11, 50, 12, 54, 13, 57, 15, 60, 16, 64},
       {18, 67, 19, 71, 21, 74, 22, 78, 24, 82, 26, 86, 28, 90, 30, 94},
       {32, 98, 34, 102, 36, 106, 38, 110, 40, 114, 42, 118, 45, 123, 47, 127},
       {50, 132, 52, 136, 55, 141, 57, 146, 60, 150, 63, 155, 66, 160, 69, 165},
       {72, 170, 75, 175, 78, 180, 81, 185, 84, 191, 87, 196, 91, 201, 94, 207},
       {98, 212, 101, 218, 105, 223, 108, 229, 112, 235, 116, 241, 120, 247, 124, 253}},
      {{0x0102, 0x0314, 0x0536, 0x0768, 0x09ab, 0x0bfd, 0x0e60, 0x10d3},
       {0x1356, 0x15e9, 0x188d, 0x1b3f, 0x1e03, 0x20d6, 0x23ba, 0x26ae},
       {0x29b2, 0x2cc6, 0x2feb, 0x331f, 0x3664, 0x39b8, 0x3d1c, 0x4091},
       {0x4416, 0x47ab, 0x4b51, 0x4f06, 0x52cc, 0x56a1, 0x5a87, 0x5e7d},
       {0x6283, 0x6699, 0x6ac0, 0x6ef5, 0x733d, 0x7792, 0x7bf9, 0x8070},
       {0x84f7, 0x898e, 0x8e36, 0x92ed, 0x97b5, 0x9c8c, 0xa173, 0xa66b},
       {0xab73, 0xb08b, 0xb5b5, 0xbaec, 0xc035, 0xc58d, 0xcaf6, 0xd06f},
       {0xd5f8, 0xdb91, 0xe13b, 0xe6f3, 0xecbe, 0xf296, 0xf880, 0xfe7a}},
      {{0x0314'1c16, 0x0768'b4df, 0x0bfd'ce69, 0x10d3'68b3},
       {0x15e9'83bf, 0x1b40'1f8d, 0x20d7'3c1b, 0x26ae'd969},
       {0x2cc6'f779, 0x331f'964a, 0x39b8'b5dc, 0x4092'562f},
       {0x47ac'7742, 0x4f07'1918, 0x56a2'3bae, 0x5e7d'df04},
       {0x669a'031c, 0x6ef6'a7f6, 0x7793'cd90, 0x8071'73ea},
       {0x898f'9b06, 0x92ee'42e4, 0x9c8d'6b82, 0xa66d'14e0},
       {0xb08d'3f00, 0xbaed'e9e2, 0xc58f'1584, 0xd070'c1e6},
       {0xdb92'ef0a, 0xe6f5'9cf0, 0xf298'cb96, 0xfe7c'7afc}},
      {{0x0768'b4df'7ef9'65b8, 0x10d3'68b3'd5b1'67dc},
       {0x1b40'1f8d'4f8c'8b20, 0x26ae'd969'a040'8b44},
       {0x331f'964b'3437'cca7, 0x4092'562f'7ee7'cacb},
       {0x4f07'1919'2cfb'2a4f, 0x5e7d'df05'71a7'2673},
       {0x6ef6'a7f7'39d6'a416, 0x8071'73eb'787e'9e3a},
       {0x92ee'42e5'5aca'39fe, 0xa66d'14e1'936e'3222},
       {0xbaed'e9e3'8fd5'ec06, 0xd070'c1e7'c275'e22a},
       {0xe6f5'9cf1'd8f9'ba2e, 0xfe7c'7afe'0595'ae52}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x9100e457,  // vmulhu.vx v8, v16, x1, v0.t
      {{0, 85, 1, 86, 2, 88, 3, 89, 5, 90, 6, 92, 7, 93, 9, 94},
       {10, 96, 11, 97, 13, 98, 14, 100, 15, 101, 17, 102, 18, 104, 19, 105},
       {21, 106, 22, 108, 23, 109, 25, 110, 26, 112, 27, 113, 29, 114, 30, 116},
       {31, 117, 33, 118, 34, 120, 35, 121, 37, 122, 38, 124, 39, 125, 41, 126},
       {42, 128, 43, 129, 45, 130, 46, 132, 47, 133, 49, 134, 50, 136, 51, 137},
       {53, 138, 54, 140, 55, 141, 57, 142, 58, 144, 59, 145, 61, 146, 62, 148},
       {63, 149, 65, 150, 66, 152, 67, 153, 69, 154, 70, 156, 71, 157, 73, 158},
       {74, 160, 75, 161, 77, 162, 78, 164, 79, 165, 81, 166, 82, 168, 83, 169}},
      {{0x55ff, 0x5756, 0x58ac, 0x5a03, 0x5b5a, 0x5cb0, 0x5e07, 0x5f5e},
       {0x60b4, 0x620b, 0x6362, 0x64b8, 0x660f, 0x6766, 0x68bc, 0x6a13},
       {0x6b6a, 0x6cc0, 0x6e17, 0x6f6e, 0x70c4, 0x721b, 0x7372, 0x74c8},
       {0x761f, 0x7776, 0x78cc, 0x7a23, 0x7b7a, 0x7cd0, 0x7e27, 0x7f7e},
       {0x80d4, 0x822b, 0x8382, 0x84d8, 0x862f, 0x8786, 0x88dc, 0x8a33},
       {0x8b8a, 0x8ce0, 0x8e37, 0x8f8e, 0x90e4, 0x923b, 0x9392, 0x94e8},
       {0x963f, 0x9796, 0x98ec, 0x9a43, 0x9b9a, 0x9cf0, 0x9e47, 0x9f9e},
       {0xa0f4, 0xa24b, 0xa3a2, 0xa4f8, 0xa64f, 0xa7a6, 0xa8fc, 0xaa53}},
      {{0x5757'00aa, 0x5a04'58ac, 0x5cb1'b0af, 0x5f5f'08b2},
       {0x620c'60b4, 0x64b9'b8b7, 0x6767'10ba, 0x6a14'68bc},
       {0x6cc1'c0bf, 0x6f6f'18c2, 0x721c'70c4, 0x74c9'c8c7},
       {0x7777'20ca, 0x7a24'78cc, 0x7cd1'd0cf, 0x7f7f'28d2},
       {0x822c'80d4, 0x84d9'd8d7, 0x8787'30da, 0x8a34'88dc},
       {0x8ce1'e0df, 0x8f8f'38e2, 0x923c'90e4, 0x94e9'e8e7},
       {0x9797'40ea, 0x9a44'98ec, 0x9cf1'f0ef, 0x9f9f'48f2},
       {0xa24c'a0f4, 0xa4f9'f8f7, 0xa7a7'50fa, 0xaa54'a8fc}},
      {{0x5a04'58ad'acac'55ff, 0x5f5f'08b3'075c'5b5a},
       {0x64b9'b8b8'620c'60b4, 0x6a14'68bd'bcbc'660f},
       {0x6f6f'18c3'176c'6b6a, 0x74c9'c8c8'721c'70c4},
       {0x7a24'78cd'cccc'761f, 0x7f7f'28d3'277c'7b7a},
       {0x84d9'd8d8'822c'80d4, 0x8a34'88dd'dcdc'862f},
       {0x8f8f'38e3'378c'8b8a, 0x94e9'e8e8'923c'90e4},
       {0x9a44'98ed'ecec'963f, 0x9f9f'48f3'479c'9b9a},
       {0xa4f9'f8f8'a24c'a0f4, 0xaa54'a8fd'fcfc'a64f}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVmulhsu) {
  TestVectorInstruction(
      0x990c2457,  // vmulhsu.vv v8, v16, v24, v0.t
      {{0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 1, 14, 1, 16},
       {2, 19, 2, 21, 3, 24, 3, 27, 4, 29, 5, 32, 6, 35, 7, 38},
       {8, 41, 9, 44, 10, 47, 11, 50, 12, 54, 13, 57, 15, 60, 16, 64},
       {18, 67, 19, 71, 21, 74, 22, 78, 24, 82, 26, 86, 28, 90, 30, 94},
       {224, 161, 224, 163, 224, 165, 224, 167, 224, 169, 224, 171, 225, 174, 225, 176},
       {226, 179, 226, 181, 227, 184, 227, 187, 228, 189, 229, 192, 230, 195, 231, 198},
       {232, 201, 233, 204, 234, 207, 235, 210, 236, 214, 237, 217, 239, 220, 240, 224},
       {242, 227, 243, 231, 245, 234, 246, 238, 248, 242, 250, 246, 252, 250, 254, 254}},
      {{0x0102, 0x0314, 0x0536, 0x0768, 0x09ab, 0x0bfd, 0x0e60, 0x10d3},
       {0x1356, 0x15e9, 0x188d, 0x1b3f, 0x1e03, 0x20d6, 0x23ba, 0x26ae},
       {0x29b2, 0x2cc6, 0x2feb, 0x331f, 0x3664, 0x39b8, 0x3d1c, 0x4091},
       {0x4416, 0x47ab, 0x4b51, 0x4f06, 0x52cc, 0x56a1, 0x5a87, 0x5e7d},
       {0xa143, 0xa357, 0xa57c, 0xa7af, 0xa9f5, 0xac48, 0xaead, 0xb122},
       {0xb3a7, 0xb63c, 0xb8e2, 0xbb97, 0xbe5d, 0xc132, 0xc417, 0xc70d},
       {0xca13, 0xcd29, 0xd051, 0xd386, 0xd6cd, 0xda23, 0xdd8a, 0xe101},
       {0xe488, 0xe81f, 0xebc7, 0xef7d, 0xf346, 0xf71c, 0xfb04, 0xfefc}},
      {{0x0314'1c16, 0x0768'b4df, 0x0bfd'ce69, 0x10d3'68b3},
       {0x15e9'83bf, 0x1b40'1f8d, 0x20d7'3c1b, 0x26ae'd969},
       {0x2cc6'f779, 0x331f'964a, 0x39b8'b5dc, 0x4092'562f},
       {0x47ac'7742, 0x4f07'1918, 0x56a2'3bae, 0x5e7d'df04},
       {0xa357'41dc, 0xa7af'e2b2, 0xac49'0448, 0xb122'a69e},
       {0xb63c'c9b6, 0xbb97'6d90, 0xc132'922a, 0xc70e'3784},
       {0xcd2a'5da0, 0xd387'047e, 0xda24'2c1c, 0xe101'd47a},
       {0xe81f'fd9a, 0xef7e'a77c, 0xf71d'd21e, 0xfefd'7d80}},
      {{0x0768'b4df'7ef9'65b8, 0x10d3'68b3'd5b1'67dc},
       {0x1b40'1f8d'4f8c'8b20, 0x26ae'd969'a040'8b44},
       {0x331f'964b'3437'cca7, 0x4092'562f'7ee7'cacb},
       {0x4f07'1919'2cfb'2a4f, 0x5e7d'df05'71a7'2673},
       {0xa7af'e2b2'7693'e2d6, 0xb122'a69e'ad33'd4f2},
       {0xbb97'6d90'8777'68ae, 0xc70e'3784'b813'58ca},
       {0xd387'047e'ac73'0aa6, 0xe101'd47a'd70a'f8c2},
       {0xef7e'a77c'e586'c8be, 0xfefd'7d81'0a1a'b4da}},
      kVectorCalculationsSource);
  TestVectorInstruction(
      0x9900e457,  // vmulhsu.vx v8, v16, x1, v0.t
      {{0, 212, 255, 211, 254, 211, 253, 210, 253, 209, 252, 209, 251, 208, 251, 207},
       {250, 207, 249, 206, 249, 205, 248, 205, 247, 204, 247, 203, 246, 203, 245, 202},
       {245, 201, 244, 201, 243, 200, 243, 199, 242, 199, 241, 198, 241, 197, 240, 197},
       {239, 196, 239, 195, 238, 195, 237, 194, 237, 193, 236, 193, 235, 192, 235, 191},
       {234, 191, 233, 190, 233, 189, 232, 189, 231, 188, 231, 187, 230, 187, 229, 186},
       {229, 185, 228, 185, 227, 184, 227, 183, 226, 183, 225, 182, 225, 181, 224, 181},
       {223, 180, 223, 179, 222, 179, 221, 178, 221, 177, 220, 177, 219, 176, 219, 175},
       {218, 175, 217, 174, 217, 173, 216, 173, 215, 172, 215, 171, 214, 171, 213, 170}},
      {{0xd4ff, 0xd454, 0xd3a8, 0xd2fd, 0xd252, 0xd1a6, 0xd0fb, 0xd050},
       {0xcfa4, 0xcef9, 0xce4e, 0xcda2, 0xccf7, 0xcc4c, 0xcba0, 0xcaf5},
       {0xca4a, 0xc99e, 0xc8f3, 0xc848, 0xc79c, 0xc6f1, 0xc646, 0xc59a},
       {0xc4ef, 0xc444, 0xc398, 0xc2ed, 0xc242, 0xc196, 0xc0eb, 0xc040},
       {0xbf94, 0xbee9, 0xbe3e, 0xbd92, 0xbce7, 0xbc3c, 0xbb90, 0xbae5},
       {0xba3a, 0xb98e, 0xb8e3, 0xb838, 0xb78c, 0xb6e1, 0xb636, 0xb58a},
       {0xb4df, 0xb434, 0xb388, 0xb2dd, 0xb232, 0xb186, 0xb0db, 0xb030},
       {0xaf84, 0xaed9, 0xae2e, 0xad82, 0xacd7, 0xac2c, 0xab80, 0xaad5}},
      {{0xd454'7faa, 0xd2fd'd3a8, 0xd1a7'27a7, 0xd050'7ba6},
       {0xcef9'cfa4, 0xcda3'23a3, 0xcc4c'77a2, 0xcaf5'cba0},
       {0xc99f'1f9f, 0xc848'739e, 0xc6f1'c79c, 0xc59b'1b9b},
       {0xc444'6f9a, 0xc2ed'c398, 0xc197'1797, 0xc040'6b96},
       {0xbee9'bf94, 0xbd93'1393, 0xbc3c'6792, 0xbae5'bb90},
       {0xb98f'0f8f, 0xb838'638e, 0xb6e1'b78c, 0xb58b'0b8b},
       {0xb434'5f8a, 0xb2dd'b388, 0xb187'0787, 0xb030'5b86},
       {0xaed9'af84, 0xad83'0383, 0xac2c'5782, 0xaad5'ab80}},
      {{0xd2fd'd3a9'29a9'd4ff, 0xd050'7ba6'7c51'd252},
       {0xcda3'23a3'cef9'cfa4, 0xcaf5'cba1'21a1'ccf7},
       {0xc848'739e'7449'ca4a, 0xc59b'1b9b'c6f1'c79c},
       {0xc2ed'c399'1999'c4ef, 0xc040'6b96'6c41'c242},
       {0xbd93'1393'bee9'bf94, 0xbae5'bb91'1191'bce7},
       {0xb838'638e'6439'ba3a, 0xb58b'0b8b'b6e1'b78c},
       {0xb2dd'b389'0989'b4df, 0xb030'5b86'5c31'b232},
       {0xad83'0383'aed9'af84, 0xaad5'ab81'0181'acd7}},
      kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVcpopm) {
  TestVₓmₓsInstruction(
      0x410820d7,  // vcpop.m x1, v16, v0.t
      { 0, /* default value when vl=0 */
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  2,
        2,  3,  3,  3,  3,  3,  3,  3,  4,  5,  5,  5,  5,  5,  5,  6,
        6,  6,  7,  7,  7,  7,  7,  7,  8,  8,  9,  9,  9,  9,  9, 10,
       10, 11, 12, 12, 12, 12, 12, 12, 13, 14, 15, 15, 15, 15, 15, 16,
       16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19, 20,
       20, 21, 21, 22, 22, 22, 22, 22, 23, 24, 24, 25, 25, 25, 25, 26,
       26, 26, 27, 28, 28, 28, 28, 28, 29, 29, 30, 31, 31, 31, 31, 32,
       32, 33, 34, 35, 35, 35, 35, 35, 36, 37, 38, 39, 39, 39, 39, 40},
      { 0, /* default value when vl=0 */
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  2,
        2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  5,
        5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,
        8,  8,  9,  9,  9,  9,  9,  9, 10, 10, 11, 11, 11, 11, 11, 12,
       12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 14,
       14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 17, 17, 17, 17, 18,
       18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 21, 21, 21, 21, 21, 21,
       21, 22, 23, 23, 23, 23, 23, 23, 23, 24, 24, 25, 25, 25, 25, 25},
      kVectorCalculationsSource[0]);
}

TEST_F(Riscv64InterpreterTest, TestVfirstm) {
  TestVₓmₓsInstruction(
      0x4108a0d7,  // vfirst.m x1, v16, v0.t
      { [0 ... 8] = ~0ULL,
        [9 ... 128] = 9 },
      { [0 ... 8] = ~0ULL,
        [9 ... 128] = 9 },
      kVectorCalculationsSource[0]);
}

}  // namespace

}  // namespace berberis
