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

#include <malloc.h>  // memalign
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>  // copy_n, fill_n
#include <cfenv>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <memory>

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"
#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/riscv64/interpreter.h"
#include "berberis/intrinsics/guest_cpu_flags.h"       // GuestModeFromHostRounding
#include "berberis/intrinsics/guest_rounding_modes.h"  // ScopedRoundingMode
#include "berberis/intrinsics/simd_register.h"
#include "berberis/intrinsics/vector_intrinsics.h"
#include "berberis/runtime_primitives/memory_region_reservation.h"
#include "faulty_memory_accesses.h"

namespace berberis {

namespace {

#if defined(__i386__)
constexpr size_t kRegIP = REG_EIP;
#elif defined(__x86_64__)
constexpr size_t kRegIP = REG_RIP;
#else
#error "Unsupported arch"
#endif

bool sighandler_called = false;

void FaultHandler(int /* sig */, siginfo_t* /* info */, void* ctx) {
  ucontext_t* ucontext = reinterpret_cast<ucontext_t*>(ctx);
  static_assert(sizeof(void*) == sizeof(greg_t), "Unsupported type sizes");
  void* fault_addr = reinterpret_cast<void*>(ucontext->uc_mcontext.gregs[kRegIP]);
  void* recovery_addr = FindFaultyMemoryAccessRecoveryAddrForTesting(fault_addr);
  sighandler_called = true;
  CHECK(recovery_addr);
  ucontext->uc_mcontext.gregs[kRegIP] = reinterpret_cast<greg_t>(recovery_addr);
}

class ScopedFaultySigaction {
 public:
  ScopedFaultySigaction() {
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = FaultHandler;
    CHECK_EQ(sigaction(SIGSEGV, &sa, &old_sa_), 0);
  }

  ~ScopedFaultySigaction() { CHECK_EQ(sigaction(SIGSEGV, &old_sa_, nullptr), 0); }

 private:
  struct sigaction old_sa_;
};

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
  template <typename ElementType>
  void TestFPExceptions(uint32_t insn_bytes) {
    // Install the following arguments: NaN, 1.0, 1.0, NaN.
    // Then try to mask out NaNs with vstart, vl, and mask.
    state_.cpu.v[24] = state_.cpu.v[16] = state_.cpu.v[8] =
        SIMD128Register{__v2du{0x7ff4'0000'0000'0000, 0x3ff0'0000'0000'0000}}.Get<__uint128_t>();
    state_.cpu.v[25] = state_.cpu.v[17] = state_.cpu.v[9] =
        SIMD128Register{__v2du{0x3ff0'0000'0000'0000, 0x7ff4'0000'0000'0000}}.Get<__uint128_t>();
    state_.cpu.v[26] = state_.cpu.v[18] = state_.cpu.v[10] = SIMD128Register{
        __v4su{
            0x7fa0'0000, 0x3f80'0000, 0x3f80'0000, 0x7fa0'0000}}.Get<__uint128_t>();
    state_.cpu.f[1] = 0xffff'ffff'3f80'0000;
    state_.cpu.f[2] = 0x3ff0'0000'0000'0000;
    ;
    state_.cpu.vtype = (BitUtilLog2(sizeof(ElementType)) << 3) | /*vlmul=*/1;
    // First clear exceptions and execute the instruction with vstart = 0, vl = 4.
    // This should produce FE_INVALID with most floating point instructions.
    EXPECT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
    state_.cpu.vstart = 0;
    state_.cpu.vl = 4;
    state_.cpu.v[0] = 0b1111;
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_TRUE(bool(fetestexcept(FE_ALL_EXCEPT)));
    // Mask NaNs using vstart and vl.
    EXPECT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
    state_.cpu.vstart = 1;
    state_.cpu.vl = 3;
    state_.cpu.v[0] = 0b1111;
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_FALSE(bool(fetestexcept(FE_ALL_EXCEPT)));
    // Mask NaNs using mask.
    EXPECT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
    state_.cpu.vstart = 0;
    state_.cpu.vl = 4;
    state_.cpu.v[0] = 0b0110;
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_FALSE(bool(fetestexcept(FE_ALL_EXCEPT)));
  }

  template <size_t kNFfields>
  void TestVlXreXX(uint32_t insn_bytes) {
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

  template <typename ElementType, size_t kNFfields, size_t kLmul>
  void VlxsegXeiXX(
      uint32_t insn_bytes,
      const ElementType::BaseType (&expected_results)[kNFfields * kLmul][static_cast<int>(
          sizeof(SIMD128Register) / sizeof(ElementType))]) {
    VlxsegXeiXX<ElementType, UInt8, kNFfields, kLmul>(insn_bytes, expected_results);
    VlxsegXeiXX<ElementType, UInt8, kNFfields, kLmul>(insn_bytes | 0x8000000, expected_results);
    VlxsegXeiXX<ElementType, UInt16, kNFfields, kLmul>(insn_bytes | 0x5000, expected_results);
    VlxsegXeiXX<ElementType, UInt16, kNFfields, kLmul>(insn_bytes | 0x8005000, expected_results);
    VlxsegXeiXX<ElementType, UInt32, kNFfields, kLmul>(insn_bytes | 0x6000, expected_results);
    VlxsegXeiXX<ElementType, UInt32, kNFfields, kLmul>(insn_bytes | 0x8006000, expected_results);
    VlxsegXeiXX<ElementType, UInt64, kNFfields, kLmul>(insn_bytes | 0x7000, expected_results);
    VlxsegXeiXX<ElementType, UInt64, kNFfields, kLmul>(insn_bytes | 0x8007000, expected_results);
  }

  template <typename DataElementType, typename IndexElementType, size_t kNFfields, size_t kLmul>
  void VlxsegXeiXX(
      uint32_t insn_bytes,
      const DataElementType::BaseType (&expected_results)[kNFfields * kLmul][static_cast<int>(
          sizeof(SIMD128Register) / sizeof(DataElementType))]) {
    constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(IndexElementType);
    constexpr size_t kTotalElements =
        sizeof(SIMD128Register) / sizeof(DataElementType) * kLmul * kNFfields;
    // If we need more indexes than may fit into 8 vector registers then such operation is
    // impossible on RISC-V and we should skip that combination.
    if constexpr (sizeof(IndexElementType) * kTotalElements <= sizeof(SIMD128Register) * 8) {
      TestVectorLoad<DataElementType, kNFfields, kLmul>(insn_bytes, expected_results, [this] {
        for (size_t reg_no = 0; reg_no < AlignUp<kElementsCount>(kTotalElements) / kElementsCount;
             ++reg_no) {
          SIMD128Register index_register;
          for (size_t index = 0; index < kElementsCount; ++index) {
            index_register.Set(IndexElementType{static_cast<typename IndexElementType::BaseType>(
                                   kPermutedIndexes[index + reg_no * kElementsCount] *
                                   sizeof(DataElementType) * kNFfields)},
                               index);
          }
          state_.cpu.v[16 + reg_no] = index_register.Get<__uint128_t>();
        }
      });
    }
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul>
  void TestVlsegXeXX(
      uint32_t insn_bytes,
      const ElementType::BaseType (
          &expected_results)[kNFfields * kLmul][static_cast<int>(16 / sizeof(ElementType))]) {
    TestVectorLoad<ElementType, kNFfields, kLmul>(insn_bytes, expected_results, [] {});
    // Turn Vector Unit-Stride Segment Loads and Stores into Fault-Only-First Load.
    TestVectorLoad<ElementType, kNFfields, kLmul>(insn_bytes | 0x1000000, expected_results, [] {});
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul>
  void TestVlssegXeXX(
      uint32_t insn_bytes,
      uint64_t stride,
      const ElementType::BaseType (
          &expected_results)[kNFfields * kLmul][static_cast<int>(16 / sizeof(ElementType))]) {
    TestVectorLoad<ElementType, kNFfields, kLmul>(
        insn_bytes, expected_results, [stride, this] { SetXReg<2>(state_.cpu, stride); });
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul, typename ExtraCPUInitType>
  void TestVectorLoad(
      uint32_t insn_bytes,
      const ElementType::BaseType (
          &expected_results)[kNFfields * kLmul][static_cast<int>(16 / sizeof(ElementType))],
      ExtraCPUInitType ExtraCPUInit) {
    constexpr int kElementsCount = static_cast<int>(16 / sizeof(ElementType));
    const auto kUndisturbedValue = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
    for (uint8_t vstart = 0; vstart <= kElementsCount * kLmul; ++vstart) {
      for (uint8_t vl = 0; vl <= kElementsCount * kLmul; ++vl) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          // Handle three masking cases:
          //   no masking (vma == 0), agnostic (vma == 1), undisturbed (vma == 2)
          for (uint8_t vma = 0; vma < 3; ++vma) {
            state_.cpu.vtype = (vma & 1) << 7 | (vta << 6) |
                               (BitUtilLog2(sizeof(ElementType)) << 3) | BitUtilLog2(kLmul);
            state_.cpu.vstart = vstart;
            state_.cpu.vl = vl;
            uint32_t insn_bytes_with_vm = insn_bytes;
            // If masking is supposed to be disabled then we need to set vm bit (#25).
            if (!vma) {
              insn_bytes_with_vm |= (1 << 25);
            }
            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes_with_vm);
            SetXReg<1>(state_.cpu, ToGuestAddr(&kVectorCalculationsSource));
            state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
            for (size_t index = 0; index < 8; index++) {
              state_.cpu.v[8 + index] = kUndisturbedValue;
            }
            ExtraCPUInit();
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
            EXPECT_EQ(state_.cpu.vstart, 0);
            EXPECT_EQ(state_.cpu.vl, vl);
            for (size_t field = 0; field < kNFfields; field++) {
              for (size_t index = 0; index < kLmul; index++) {
                for (size_t element = 0; element < kElementsCount; ++element) {
                  ElementType expected_element;
                  if (vstart >= vl) {
                    // When vstart ⩾ vl, there are no body elements, and no elements are updated in
                    // any destinationvector register group, including that no tail elements are
                    // updated with agnostic values.
                    expected_element =
                        SIMD128Register{kUndisturbedResult}.Get<ElementType>(element);
                  } else if (element + index * kElementsCount < std::min(vstart, vl)) {
                    // Elements before vstart are undisturbed if vstart is less than vl.
                    expected_element =
                        SIMD128Register{kUndisturbedResult}.Get<ElementType>(element);
                  } else if (element + index * kElementsCount >= vl) {
                    // Element after vl have to be processed with vta policy.
                    if (vta == 1) {
                      expected_element = ~ElementType{0};
                    } else {
                      expected_element =
                          SIMD128Register{kUndisturbedResult}.Get<ElementType>(element);
                    }
                  } else if (vma &&
                             (~(state_.cpu.v[0]) >> (element + index * kElementsCount)) & 1) {
                    // If element is inactive it's processed with vma policy.
                    if (vma == 1) {
                      expected_element = ~ElementType{0};
                    } else {
                      expected_element =
                          SIMD128Register{kUndisturbedResult}.Get<ElementType>(element);
                    }
                  } else {
                    expected_element =
                        ElementType{expected_results[index + field * kLmul][element]};
                  }
                  EXPECT_EQ(
                      SIMD128Register{state_.cpu.v[8 + index + field * kLmul]}.Get<ElementType>(
                          element),
                      expected_element);
                }
              }
            }
            for (size_t index = kNFfields * kLmul; index < 8; index++) {
              EXPECT_EQ(state_.cpu.v[8 + index], kUndisturbedValue);
            }
          }
        }
      }
    }
  }

  void TestVlm(uint32_t insn_bytes, __v16qu expected_results) {
    const auto kUndisturbedValue = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
    // Vlm.v is special form of normal vector load which mostly ignores vtype.
    // The only bit that it honors is vill: https://github.com/riscv/riscv-v-spec/pull/877
    // Verify that changes to vtype don't affect the execution (but vstart and vl do).
    for (uint8_t sew = 0; sew < 4; ++sew) {
      for (uint8_t vlmul = 0; vlmul < 4; ++vlmul) {
        const uint8_t kElementsCount = (16 >> sew) << vlmul;
        for (uint8_t vstart = 0; vstart <= kElementsCount; ++vstart) {
          for (uint8_t vl = 0; vl <= kElementsCount; ++vl) {
            const uint8_t kVlmVl = AlignUp<CHAR_BIT>(vl) / CHAR_BIT;
            for (uint8_t vta = 0; vta < 2; ++vta) {
              for (uint8_t vma = 0; vma < 2; ++vma) {
                state_.cpu.vtype = (vma << 7) | (vta << 6) | (sew << 3) | vlmul;
                state_.cpu.vstart = vstart;
                state_.cpu.vl = vl;
                state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
                SetXReg<1>(state_.cpu, ToGuestAddr(&kVectorCalculationsSource));
                state_.cpu.v[8] = kUndisturbedValue;
                EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
                EXPECT_EQ(state_.cpu.vstart, 0);
                EXPECT_EQ(state_.cpu.vl, vl);
                for (size_t element = 0; element < 16; ++element) {
                  UInt8 expected_element;
                  if (element < vstart || vstart >= kVlmVl) {
                    expected_element = SIMD128Register{kUndisturbedResult}.Get<UInt8>(element);
                  } else if (element >= kVlmVl) {
                    expected_element = ~UInt8{0};
                  } else {
                    expected_element = UInt8{expected_results[element]};
                  }
                  EXPECT_EQ(SIMD128Register{state_.cpu.v[8]}.Get<UInt8>(element), expected_element);
                }
              }
            }
          }
        }
      }
    }
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul, size_t kResultSize>
  void VsxsegXeiXX(uint32_t insn_bytes, const uint64_t (&expected_results)[kResultSize]) {
    VsxsegXeiXX<ElementType, UInt8, kNFfields, kLmul>(insn_bytes, expected_results);
    VsxsegXeiXX<ElementType, UInt8, kNFfields, kLmul>(insn_bytes | 0x8000000, expected_results);
    VsxsegXeiXX<ElementType, UInt16, kNFfields, kLmul>(insn_bytes | 0x5000, expected_results);
    VsxsegXeiXX<ElementType, UInt16, kNFfields, kLmul>(insn_bytes | 0x8005000, expected_results);
    VsxsegXeiXX<ElementType, UInt32, kNFfields, kLmul>(insn_bytes | 0x6000, expected_results);
    VsxsegXeiXX<ElementType, UInt32, kNFfields, kLmul>(insn_bytes | 0x8006000, expected_results);
    VsxsegXeiXX<ElementType, UInt64, kNFfields, kLmul>(insn_bytes | 0x7000, expected_results);
    VsxsegXeiXX<ElementType, UInt64, kNFfields, kLmul>(insn_bytes | 0x8007000, expected_results);
  }

  template <typename DataElementType,
            typename IndexElementType,
            size_t kNFfields,
            size_t kLmul,
            size_t kResultSize>
  void VsxsegXeiXX(uint32_t insn_bytes, const uint64_t (&expected_results)[kResultSize]) {
    constexpr size_t kElementsCount = sizeof(SIMD128Register) / sizeof(IndexElementType);
    constexpr size_t kTotalElements =
        sizeof(SIMD128Register) / sizeof(DataElementType) * kLmul * kNFfields;
    // If we need more indexes than may fit into 8 vector registers then such operation is
    // impossible on RISC-V and we should skip that combination.
    if constexpr (sizeof(IndexElementType) * kTotalElements <= sizeof(SIMD128Register) * 8) {
      TestVectorStore<DataElementType, kNFfields, kLmul>(insn_bytes, expected_results, [this] {
        for (size_t reg_no = 0; reg_no < AlignUp<kElementsCount>(kTotalElements) / kElementsCount;
             ++reg_no) {
          SIMD128Register index_register;
          for (size_t index = 0; index < kElementsCount; ++index) {
            index_register.Set(IndexElementType{static_cast<typename IndexElementType::BaseType>(
                                   kPermutedIndexes[index + reg_no * kElementsCount] *
                                   sizeof(DataElementType) * kNFfields)},
                               index);
          }
          state_.cpu.v[16 + reg_no] = index_register.Get<__uint128_t>();
        }
      });
    }
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul, size_t kResultSize>
  void TestVssegXeXX(uint32_t insn_bytes, const uint64_t (&expected_results)[kResultSize]) {
    TestVectorStore<ElementType, kNFfields, kLmul>(insn_bytes, expected_results, [] {});
  }

  template <typename ElementType, size_t kNFfields, size_t kLmul, size_t kResultSize>
  void TestVsssegXeXX(uint32_t insn_bytes,
                      uint64_t stride,
                      const uint64_t (&expected_results)[kResultSize]) {
    TestVectorStore<ElementType, kNFfields, kLmul>(
        insn_bytes, expected_results, [stride, this] { SetXReg<2>(state_.cpu, stride); });
  }

  template <typename ElementType,
            size_t kNFfields,
            size_t kLmul,
            size_t kResultSize,
            typename ExtraCPUInitType>
  void TestVectorStore(uint32_t insn_bytes,
                       const uint64_t (&expected_results)[kResultSize],
                       ExtraCPUInitType ExtraCPUInit) {
    constexpr size_t kElementsCount = 16 / sizeof(ElementType);
    const SIMD128Register kUndisturbedValue = SIMD128Register{kUndisturbedResult};
    state_.cpu.vtype = (BitUtilLog2(sizeof(ElementType)) << 3) | BitUtilLog2(kLmul);
    state_.cpu.vstart = 0;
    state_.cpu.vl = kElementsCount * kLmul;
    // First verify that store works with no inactive elements and no masking.
    uint32_t insn_bytes_with_vm = insn_bytes | (1 << 25);
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes_with_vm);
    SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
    for (size_t index = 0; index < 8; ++index) {
      state_.cpu.v[8 + index] =
          SIMD128Register{kVectorCalculationsSource[index]}.Get<__uint128_t>();
    }
    for (uint64_t& element : store_area_) {
      element = kUndisturbedResult[0];
    }
    ExtraCPUInit();
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    for (size_t index = 0; index < std::size(store_area_); ++index) {
      if (index >= std::size(expected_results)) {
        EXPECT_EQ(store_area_[index], static_cast<uint64_t>(kUndisturbedResult[0]));
      } else {
        EXPECT_EQ(store_area_[index], expected_results[index]);
      }
    }
    // Most vector instruction apply masks to *target* operands which means we may apply the exact
    // same, well-tested logic to verify how they work. But store instructions apply masks to *data
    // source* operand which makes generation of expected target problematic (especially for
    // complicated store with segments and strides and/or indexes). To sidestep the issue we are
    // first testing that full version with all elements active works (above) and then reuse it to
    // verify that `vstart`, `vl` 𝖺𝗇𝖽 `mask` operands work as expected. This wouldn't work if some
    // elements are overlapping, but these instructions, while technically permitted, already are
    // allowed to produce many possible different results, thus we are not testing them.
    for (uint8_t vstart = 0; vstart <= kElementsCount * kLmul; ++vstart) {
      for (uint8_t vl = 0; vl <= kElementsCount * kLmul; ++vl) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          // Handle three masking cases:
          //   no masking (vma == 0), agnostic (vma == 1), undisturbed (vma == 2)
          // Note: vta and vma settings are ignored by store instructions, we are just verifying
          // this is what actually happens.
          for (uint8_t vma = 0; vma < 3; ++vma) {
            // Use instruction in the mode tested above to generate expected results.
            state_.cpu.vtype = (BitUtilLog2(sizeof(ElementType)) << 3) | BitUtilLog2(kLmul);
            state_.cpu.vstart = 0;
            state_.cpu.vl = kElementsCount * kLmul;
            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes_with_vm);
            decltype(store_area_) expected_store_area;
            SetXReg<1>(state_.cpu, ToGuestAddr(&expected_store_area));
            for (size_t index = 0; index < kLmul; ++index) {
              for (size_t field = 0; field < kNFfields; ++field) {
                SIMD128Register register_value =
                    SIMD128Register{kVectorCalculationsSource[index + field * kLmul]};
                if (vma) {
                  auto ApplyMask = [&register_value, &kUndisturbedValue, index](auto mask) {
                    register_value =
                        (register_value & mask[index]) | (kUndisturbedValue & ~mask[index]);
                  };
                  if constexpr (sizeof(ElementType) == sizeof(Int8)) {
                    ApplyMask(kMaskInt8);
                  } else if constexpr (sizeof(ElementType) == sizeof(Int16)) {
                    ApplyMask(kMaskInt16);
                  } else if constexpr (sizeof(ElementType) == sizeof(Int32)) {
                    ApplyMask(kMaskInt32);
                  } else if constexpr (sizeof(ElementType) == sizeof(Int64)) {
                    ApplyMask(kMaskInt64);
                  } else {
                    static_assert(kDependentTypeFalse<ElementType>);
                  }
                }
                if (vstart > kElementsCount * index) {
                  for (size_t prefix_id = 0;
                       prefix_id < std::min(kElementsCount, vstart - kElementsCount * index);
                       ++prefix_id) {
                    register_value.Set(kUndisturbedValue.Get<ElementType>(0), prefix_id);
                  }
                }
                if (vl <= kElementsCount * index) {
                  register_value = kUndisturbedValue.Get<__uint128_t>();
                } else if (vl < kElementsCount * (index + 1)) {
                  for (size_t suffix_id = vl - kElementsCount * index; suffix_id < kElementsCount;
                       ++suffix_id) {
                    register_value.Set(kUndisturbedValue.Get<ElementType>(0), suffix_id);
                  }
                }
                state_.cpu.v[8 + index + field * kLmul] = register_value.Get<__uint128_t>();
              }
            }
            for (uint64_t& element : expected_store_area) {
              element = kUndisturbedResult[0];
            }
            ExtraCPUInit();
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
            // Now execute instruction with mode that we want to actually test.
            state_.cpu.vtype = (vma & 1) << 7 | (vta << 6) |
                               (BitUtilLog2(sizeof(ElementType)) << 3) | BitUtilLog2(kLmul);
            state_.cpu.vstart = vstart;
            state_.cpu.vl = vl;
            uint32_t insn_bytes_with_vm = insn_bytes;
            // If masking is supposed to be disabled then we need to set vm bit (#25).
            if (!vma) {
              insn_bytes_with_vm |= (1 << 25);
            }
            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes_with_vm);
            SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
            state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
            for (size_t index = 0; index < 8; ++index) {
              state_.cpu.v[8 + index] =
                  SIMD128Register{kVectorCalculationsSource[index]}.Get<__uint128_t>();
            }
            for (uint64_t& element : store_area_) {
              element = kUndisturbedResult[0];
            }
            ExtraCPUInit();
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
            for (size_t index = 0; index < std::size(store_area_); ++index) {
              EXPECT_EQ(store_area_[index], expected_store_area[index]);
            }
          }
        }
      }
    }
  }

  void TestVsm(uint32_t insn_bytes, __v16qu expected_results) {
    const auto kUndisturbedValue = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
    // Vlm.v is special form of normal vector load which mostly ignores vtype.
    // The only bit that it honors is vill: https://github.com/riscv/riscv-v-spec/pull/877
    // Verify that changes to vtype don't affect the execution (but vstart and vl do).
    for (uint8_t sew = 0; sew < 4; ++sew) {
      for (uint8_t vlmul = 0; vlmul < 4; ++vlmul) {
        const uint8_t kElementsCount = (16 >> sew) << vlmul;
        for (uint8_t vstart = 0; vstart <= kElementsCount; ++vstart) {
          for (uint8_t vl = 0; vl <= kElementsCount; ++vl) {
            const uint8_t kVlmVl = AlignUp<CHAR_BIT>(vl) / CHAR_BIT;
            for (uint8_t vta = 0; vta < 2; ++vta) {
              for (uint8_t vma = 0; vma < 2; ++vma) {
                state_.cpu.vtype = (vma << 7) | (vta << 6) | (sew << 3) | vlmul;
                state_.cpu.vstart = vstart;
                state_.cpu.vl = vl;
                state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
                SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
                store_area_[0] = kUndisturbedResult[0];
                store_area_[1] = kUndisturbedResult[1];
                state_.cpu.v[8] = SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>();
                EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
                EXPECT_EQ(state_.cpu.vstart, 0);
                EXPECT_EQ(state_.cpu.vl, vl);
                SIMD128Register memory_result =
                    SIMD128Register{__v2du{store_area_[0], store_area_[1]}};
                for (size_t element = 0; element < 16; ++element) {
                  UInt8 expected_element;
                  if (element < vstart || element >= kVlmVl) {
                    expected_element = SIMD128Register{kUndisturbedResult}.Get<UInt8>(element);
                  } else {
                    expected_element = UInt8{expected_results[element]};
                  }
                  EXPECT_EQ(memory_result.Get<UInt8>(element), expected_element);
                }
              }
            }
          }
        }
      }
    }
  }

  // Vector instructions.
  void TestVleXXff(uint32_t insn_bytes,
                   uint8_t loadable_bytes,
                   uint8_t vsew,
                   uint8_t expected_vl,
                   bool fail_on_first = false) {
    ScopedFaultySigaction scoped_sa;
    sighandler_called = false;
    char* buffer;
    const size_t kPageSize = sysconf(_SC_PAGE_SIZE);
    buffer = (char*)memalign(kPageSize, 2 * kPageSize);
    mprotect(buffer, kPageSize * 2, PROT_WRITE);
    char* p = buffer + kPageSize - loadable_bytes;
    std::memcpy(p, &kVectorCalculationsSource[0], 128);
    mprotect(buffer + kPageSize, kPageSize, PROT_NONE);

    auto [vlmax, vtype] = intrinsics::Vsetvl(~0ULL, (vsew << 3) | 3);
    state_.cpu.vstart = 0;
    state_.cpu.vl = vlmax;
    state_.cpu.vtype = vtype;
    insn_bytes = insn_bytes | (1 << 25);
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    SetXReg<1>(state_.cpu, ToGuestAddr(p));

    if (fail_on_first) {
      EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr));
    } else {
      EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    }
    if (loadable_bytes < 128) {
      EXPECT_TRUE(sighandler_called);
    } else {
      EXPECT_FALSE(sighandler_called);
    }
    EXPECT_EQ(state_.cpu.vl, expected_vl);
  }

  // Vector instructions.
  void TestVleXX(uint32_t insn_bytes,
                 const __v16qu (&expected_result_int8)[8],
                 const __v8hu (&expected_result_int16)[8],
                 const __v4su (&expected_result_int32)[8],
                 const __v2du (&expected_result_int64)[8],
                 uint8_t veew,
                 const __v2du (&source)[16]) {
    auto Verify = [this, &source](uint32_t insn_bytes,
                                  uint8_t vsew,
                                  uint8_t veew,
                                  uint8_t vlmul_max,
                                  const auto& expected_result,
                                  auto mask) {
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (uint8_t vlmul = 0; vlmul < vlmul_max; ++vlmul) {
        int vemul = SignExtend<3>(vlmul);
        vemul += vsew;  // Multiply by SEW.
        vemul -= veew;  // Divide by EEW.
        if (vemul < -3 || vemul > 3) {
          // Incompatible vlmul
          continue;
        }

        for (uint8_t vta = 0; vta < 2; ++vta) {
          for (uint8_t vma = 0; vma < 2; ++vma) {
            auto [vlmax, vtype] =
                intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
            if (vlmax == 0) {
              continue;
            }
            // Test vstart/vl changes with only with vemul == 2 (4 registers)
            if (vemul == 2) {
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
            SetXReg<1>(state_.cpu, ToGuestAddr(&source));
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
            // Values for inactive elements (i.e. corresponding mask bit is 0).
            const size_t n = std::size(source);
            __m128i expected_inactive[n];
            std::fill_n(expected_inactive, n, (vma ? kAgnosticResult : kUndisturbedResult));
            if (vemul >= 0) {
              for (size_t index = 0; index < 1 << vemul; ++index) {
                if (index == 0 && vemul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            SIMD128Register{
                                (kUndisturbedResult & kFractionMaskInt8[3]) |
                                (expected_result[index] & mask[index] & ~kFractionMaskInt8[3]) |
                                (expected_inactive[index] & ~mask[index] & ~kFractionMaskInt8[3])}
                                .Get<__uint128_t>());
                } else if (index == 2 && vemul == 2) {
                  EXPECT_EQ(
                      state_.cpu.v[8 + index],
                      SIMD128Register{
                          (expected_result[index] & mask[index] & kFractionMaskInt8[3]) |
                          (expected_inactive[index] & ~mask[index] & kFractionMaskInt8[3]) |
                          ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[3])}
                          .Get<__uint128_t>());
                } else if (index == 3 && vemul == 2 && vta) {
                  EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kAgnosticResult});
                } else if (index == 3 && vemul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kUndisturbedResult});
                } else {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            SIMD128Register{(expected_result[index] & mask[index]) |
                                            (expected_inactive[index] & ~mask[index])}
                                .Get<__uint128_t>());
                }
              }

            } else {
              EXPECT_EQ(state_.cpu.v[8],
                        SIMD128Register{
                            (expected_result[0] & mask[0] & kFractionMaskInt8[(vemul + 4)]) |
                            (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[(vemul + 4)]) |
                            ((vta ? kAgnosticResult : kUndisturbedResult) &
                             ~kFractionMaskInt8[(vemul + 4)])}
                            .Get<__uint128_t>());
            }
          }
        }
      }
    };
    switch (veew) {
      case 0:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int8, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int8, kMaskInt8);
        break;
      case 1:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int16, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int16, kMaskInt16);
        break;
      case 2:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int32, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int32, kMaskInt32);
        break;
      case 3:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int64, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int64, kMaskInt64);
        break;
      default:
        break;
    }
  }

  // Vector instructions.
  void TestVseXX(uint32_t insn_bytes,
                 const __v16qu (&expected_result_int8)[8],
                 const __v8hu (&expected_result_int16)[8],
                 const __v4su (&expected_result_int32)[8],
                 const __v2du (&expected_result_int64)[8],
                 uint8_t veew,
                 const __v2du (&source)[16]) {
    auto Verify = [this, &source](uint32_t insn_bytes,
                                  uint8_t vsew,
                                  uint8_t veew,
                                  uint8_t vlmul_max,
                                  const auto& expected_result,
                                  auto mask) {
      SIMD128Register expected_result_in_register;
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (uint8_t vlmul = 0; vlmul < vlmul_max; ++vlmul) {
        int vemul = SignExtend<3>(vlmul);
        vemul += vsew;  // Multiply by SEW.
        vemul -= veew;  // Divide by EEW.
        if (vemul < -3 || vemul > 3) {
          // Incompatible vlmul
          continue;
        }

        for (uint8_t vma = 0; vma < 2; ++vma) {
          auto [vlmax, vtype] = intrinsics::Vsetvl(~0ULL, (vma << 7) | (vsew << 3) | vlmul);
          if (vlmax == 0) {
            continue;
          }
          // Test vstart/vl changes with only with vemul == 2 (4 registers)
          if (vemul == 2) {
            state_.cpu.vstart = vlmax / 8;
            state_.cpu.vl = (vlmax * 5) / 8;
          } else {
            state_.cpu.vstart = 0;
            state_.cpu.vl = vlmax;
          }
          state_.cpu.vtype = vtype;

          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          SetXReg<1>(state_.cpu, ToGuestAddr(&store_area_));
          for (size_t index = 0; index < 8; index++) {
            state_.cpu.v[8 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
            store_area_[index * 2] = kUndisturbedResult[0];
            store_area_[index * 2 + 1] = kUndisturbedResult[1];
          }
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
          // Values for inactive elements (i.e. corresponding mask bit is 0).
          const size_t n = std::size(source);
          __m128i expected_inactive[n];
          std::fill_n(expected_inactive, n, kUndisturbedResult);
          if (vemul >= 0) {
            for (size_t index = 0; index < 1 << vemul; ++index) {
              if (index == 0 && vemul == 2) {
                expected_result_in_register = SIMD128Register{
                    (kUndisturbedResult & kFractionMaskInt8[3]) |
                    (expected_result[index] & mask[index] & ~kFractionMaskInt8[3]) |
                    (expected_inactive[index] & ~mask[index] & ~kFractionMaskInt8[3])};
              } else if (index == 2 && vemul == 2) {
                expected_result_in_register = SIMD128Register{
                    (expected_result[index] & mask[index] & kFractionMaskInt8[3]) |
                    (expected_inactive[index] & ~mask[index] & kFractionMaskInt8[3]) |
                    ((kUndisturbedResult) & ~kFractionMaskInt8[3])};
              } else if (index == 3 && vemul == 2) {
                expected_result_in_register = SIMD128Register{kUndisturbedResult};
              } else {
                expected_result_in_register =
                    SIMD128Register{(expected_result[index] & mask[index]) |
                                    (expected_inactive[index] & ~mask[index])};
              }

              EXPECT_EQ(store_area_[index * 2], expected_result_in_register.Get<uint64_t>(0));
              EXPECT_EQ(store_area_[index * 2 + 1], expected_result_in_register.Get<uint64_t>(1));
            }

          } else {
            expected_result_in_register =
                SIMD128Register{(expected_result[0] & mask[0] & kFractionMaskInt8[(vemul + 4)]) |
                                (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[(vemul + 4)]) |
                                ((kUndisturbedResult) & ~kFractionMaskInt8[(vemul + 4)])};
            EXPECT_EQ(store_area_[0], expected_result_in_register.Get<uint64_t>(0));
            EXPECT_EQ(store_area_[1], expected_result_in_register.Get<uint64_t>(1));
          }
        }
      }
    };
    switch (veew) {
      case 0:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int8, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int8, kMaskInt8);
        break;
      case 1:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int16, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int16, kMaskInt16);
        break;
      case 2:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int32, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int32, kMaskInt32);
        break;
      case 3:
        Verify(insn_bytes | (1 << 25), veew, veew, 8, expected_result_int64, kNoMask);
        Verify(insn_bytes, veew, veew, 8, expected_result_int64, kMaskInt64);
        break;
      default:
        break;
    }
  }

  template <int kNFfields>
  void TestVmvXr(uint32_t insn_bytes) {
    TestVmvXr<Int8, kNFfields>(insn_bytes);
    TestVmvXr<Int16, kNFfields>(insn_bytes);
    TestVmvXr<Int32, kNFfields>(insn_bytes);
    TestVmvXr<Int64, kNFfields>(insn_bytes);
  }

  template <typename ElementType, int kNFfields>
  void TestVmvXr(uint32_t insn_bytes) {
    // Note that VmvXr actually DOES depend on vtype, contrary to what RISC-V V 1.0 manual says:
    //   https://github.com/riscv/riscv-v-spec/pull/872
    state_.cpu.vtype = BitUtilLog2(sizeof(ElementType)) << 3;
    state_.cpu.vl = 0;
    constexpr int kElementsCount = static_cast<int>(sizeof(SIMD128Register) / sizeof(ElementType));
    for (int vstart = 0; vstart <= kElementsCount * kNFfields; ++vstart) {
      state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
      state_.cpu.vstart = vstart;
      for (size_t index = 0; index < 16; ++index) {
        state_.cpu.v[8 + index] =
            SIMD128Register{kVectorComparisonSource[index]}.Get<__uint128_t>();
      }
      EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
      for (int index = 0; index < 8; ++index) {
        SIMD128Register expected_state{kVectorComparisonSource[index]};
        SIMD128Register source_value{kVectorComparisonSource[index + 8]};
        if ((vstart < kElementsCount * kNFfields) && index < kNFfields) {
          // The usual property that no elements are written if vstart >= vl does not apply to these
          // instructions. Instead, no elements are written if vstart >= evl.
          for (int element_index = 0; element_index < kElementsCount; ++element_index) {
            if (element_index + index * kElementsCount >= vstart) {
              expected_state.Set(source_value.Get<ElementType>(element_index), element_index);
            }
          }
        }
        EXPECT_EQ(state_.cpu.v[8 + index], expected_state.Get<__uint128_t>());
      }
      EXPECT_EQ(state_.cpu.vstart, 0);
    }
  }

  template <typename ElementType>
  void TestVfmvfs(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.vtype = BitUtilLog2(sizeof(ElementType)) << 3;
    state_.cpu.vstart = 0;
    state_.cpu.vl = 0;
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    state_.cpu.v[8] = SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>();
    SetFReg<1>(state_.cpu, 0x5555'5555'5555'5555);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(GetFReg<1>(state_.cpu), expected_result);
  }

  template <typename ElementType>
  void TestVfmvsf(uint32_t insn_bytes, uint64_t boxed_value, ElementType unboxed_value) {
    for (uint8_t vstart = 0; vstart < 2; ++vstart) {
      for (uint8_t vl = 0; vl < 2; ++vl) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          state_.cpu.vtype = (vta << 6) | (BitUtilLog2(sizeof(ElementType)) << 3);
          state_.cpu.vstart = vstart;
          state_.cpu.vl = vl;
          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          state_.cpu.v[8] = SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>();
          SetFReg<1>(state_.cpu, boxed_value);
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
          if (vstart == 0 && vl != 0) {
            SIMD128Register expected_result =
                vta ? ~SIMD128Register{} : SIMD128Register{kVectorCalculationsSource[0]};
            expected_result.Set<ElementType>(unboxed_value, 0);
            EXPECT_EQ(state_.cpu.v[8], expected_result);
          } else {
            EXPECT_EQ(state_.cpu.v[8],
                      SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>());
          }
        }
      }
    }
  }

  template <typename ElementType>
  void TestVmvsx(uint32_t insn_bytes) {
    for (uint8_t vstart = 0; vstart < 2; ++vstart) {
      for (uint8_t vl = 0; vl < 2; ++vl) {
        for (uint8_t vta = 0; vta < 2; ++vta) {
          state_.cpu.vtype = (vta << 6) | (BitUtilLog2(sizeof(ElementType)) << 3);
          state_.cpu.vstart = vstart;
          state_.cpu.vl = vl;
          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          state_.cpu.v[8] = SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>();
          SetXReg<1>(state_.cpu, 0x5555'5555'5555'5555);
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
          if (vstart == 0 && vl != 0) {
            SIMD128Register expected_result =
                vta ? ~SIMD128Register{} : SIMD128Register{kVectorCalculationsSource[0]};
            expected_result.Set<ElementType>(MaybeTruncateTo<ElementType>(0x5555'5555'5555'5555),
                                             0);
            EXPECT_EQ(state_.cpu.v[8], expected_result);
          } else {
            EXPECT_EQ(state_.cpu.v[8],
                      SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>());
          }
        }
      }
    }
  }

  template <typename ElementType>
  void TestVmvxs(uint32_t insn_bytes, uint64_t expected_result) {
    state_.cpu.vtype = BitUtilLog2(sizeof(ElementType)) << 3;
    state_.cpu.vstart = 0;
    state_.cpu.vl = 0;
    state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
    state_.cpu.v[8] = SIMD128Register{kVectorCalculationsSource[0]}.Get<__uint128_t>();
    SetXReg<1>(state_.cpu, 0x5555'5555'5555'5555);
    EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));
    EXPECT_EQ(GetXReg<1>(state_.cpu), expected_result);
  }

  template <size_t kNFfields>
  void TestVsX(uint32_t insn_bytes) {
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

  void TestVectorRegisterGather(uint32_t insn_bytes,
                                const __v16qu (&expected_result_int8)[8],
                                const __v8hu (&expected_result_int16)[8],
                                const __v4su (&expected_result_int32)[8],
                                const __v2du (&expected_result_int64)[8],
                                const uint8_t vlmul,
                                const __v2du (&source)[16]) {
    auto Verify = [this, &source, &vlmul](
                      uint32_t insn_bytes, uint8_t vsew, const auto& expected_result, auto mask) {
      // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
      // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (size_t index = 0; index < 8; ++index) {
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

          // Make sure indexes in src2 fall within vlmax.
          uint64_t src2_mask{0};
          const size_t kElementSize = 8 << vsew;
          const uint64_t kIndexMask = (1 << BitUtilLog2(vlmax)) - 1;
          for (uint8_t index = 0; index < 64 / kElementSize; index++) {
            src2_mask |= kIndexMask << (kElementSize * index);
          }
          for (size_t index = 0; index < 8; ++index) {
            __v2du masked_register = {source[8 + index][0] & src2_mask, source[8 + index][1]};
            state_.cpu.v[24 + index] = SIMD128Register{masked_register}.Get<__uint128_t>();
          }
          SetXReg<1>(state_.cpu, 0xaaaa'aaaa'aaaa'aaaaULL & kIndexMask);

          // To make tests quick enough we don't test vstart and vl change with small register
          // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of
          // first
          // register and half of last register.
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
          const size_t n = std::size(source) * 2;
          __m128i expected_inactive[n];
          // For most instructions, follow basic inactive processing rules based on vma flag.
          std::fill_n(expected_inactive, n, (vma ? kAgnosticResult : kUndisturbedResult));

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
                SIMD128Register{
                    (expected_result[0] & mask[0] & kFractionMaskInt8[vlmul - 4]) |
                    (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[vlmul - 4]) |
                    ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[vlmul - 4])}
                    .Get<__uint128_t>());
          }

          if (vlmul == 2) {
            // Every vector instruction must set vstart to 0, but shouldn't touch vl.
            EXPECT_EQ(state_.cpu.vstart, 0);
            EXPECT_EQ(state_.cpu.vl, (vlmax * 5) / 8);
          }
        }
      }
    };
    Verify(insn_bytes, 0, expected_result_int8, kMaskInt8);
    Verify(insn_bytes | (1 << 25), 0, expected_result_int8, kNoMask);
    Verify(insn_bytes, 1, expected_result_int16, kMaskInt16);
    Verify(insn_bytes | (1 << 25), 1, expected_result_int16, kNoMask);
    Verify(insn_bytes, 2, expected_result_int32, kMaskInt32);
    Verify(insn_bytes | (1 << 25), 2, expected_result_int32, kNoMask);
    Verify(insn_bytes, 3, expected_result_int64, kMaskInt64);
    Verify(insn_bytes | (1 << 25), 3, expected_result_int64, kNoMask);
  }

  void TestVectorFloatInstruction(uint32_t insn_bytes,
                                  const uint32_t (&expected_result_int32)[8][4],
                                  const uint64_t (&expected_result_int64)[8][2],
                                  const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kDefault>(
        insn_bytes, source, expected_result_int32, expected_result_int64);
  }

  void TestVectorInstruction(uint32_t insn_bytes,
                             const uint8_t (&expected_result_int8)[8][16],
                             const uint16_t (&expected_result_int16)[8][8],
                             const uint32_t (&expected_result_int32)[8][4],
                             const uint64_t (&expected_result_int64)[8][2],
                             const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kInteger, TestVectorInstructionMode::kDefault>(
        insn_bytes,
        source,
        expected_result_int8,
        expected_result_int16,
        expected_result_int32,
        expected_result_int64);
  }

  void TestVectorMergeFloatInstruction(uint32_t insn_bytes,
                                       const uint32_t (&expected_result_int32)[8][4],
                                       const uint64_t (&expected_result_int64)[8][2],
                                       const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kVMerge>(
        insn_bytes, source, expected_result_int32, expected_result_int64);
  }

  void TestVectorMergeInstruction(uint32_t insn_bytes,
                                  const uint8_t (&expected_result_int8)[8][16],
                                  const uint16_t (&expected_result_int16)[8][8],
                                  const uint32_t (&expected_result_int32)[8][4],
                                  const uint64_t (&expected_result_int64)[8][2],
                                  const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kInteger, TestVectorInstructionMode::kVMerge>(
        insn_bytes,
        source,
        expected_result_int8,
        expected_result_int16,
        expected_result_int32,
        expected_result_int64);
  }

  void TestNarrowingVectorInstruction(uint32_t insn_bytes,
                                      const uint8_t (&expected_result_int8)[4][16],
                                      const uint16_t (&expected_result_int16)[4][8],
                                      const uint32_t (&expected_result_int32)[4][4],
                                      const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kInteger,
                          TestVectorInstructionMode::kNarrowing>(
        insn_bytes, source, expected_result_int8, expected_result_int16, expected_result_int32);
  }

  void TestWideningVectorFloatInstruction(uint32_t insn_bytes,
                                          const uint64_t (&expected_result_int64)[8][2],
                                          const __v2du (&source)[16],
                                          __m128i dst_result = kUndisturbedResult) {
    TestVectorInstructionInternal<TestVectorInstructionKind::kFloat,
                                  TestVectorInstructionMode::kWidening>(
        insn_bytes, dst_result, source, expected_result_int64);
  }

  void TestWideningVectorFloatInstruction(uint32_t insn_bytes,
                                          const uint32_t (&expected_result_int32)[8][4],
                                          const uint64_t (&expected_result_int64)[8][2],
                                          const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kFloat, TestVectorInstructionMode::kWidening>(
        insn_bytes, source, expected_result_int32, expected_result_int64);
  }

  void TestWideningVectorInstruction(uint32_t insn_bytes,
                                     const uint16_t (&expected_result_int16)[8][8],
                                     const uint32_t (&expected_result_int32)[8][4],
                                     const uint64_t (&expected_result_int64)[8][2],
                                     const __v2du (&source)[16]) {
    TestVectorInstruction<TestVectorInstructionKind::kInteger,
                          TestVectorInstructionMode::kWidening>(
        insn_bytes, source, expected_result_int16, expected_result_int32, expected_result_int64);
  }

  enum class TestVectorInstructionKind { kInteger, kFloat };
  enum class TestVectorInstructionMode { kDefault, kWidening, kNarrowing, kVMerge };

  template <TestVectorInstructionKind kTestVectorInstructionKind,
            TestVectorInstructionMode kTestVectorInstructionMode,
            typename... ElementType,
            size_t... kResultsCount,
            size_t... kElementCount>
  void TestVectorInstruction(
      uint32_t insn_bytes,
      const __v2du (&source)[16],
      const ElementType (&... expected_result)[kResultsCount][kElementCount]) {
    TestVectorInstructionInternal<kTestVectorInstructionKind, kTestVectorInstructionMode>(
        insn_bytes, kUndisturbedResult, source, expected_result...);
  }

  template <TestVectorInstructionKind kTestVectorInstructionKind,
            TestVectorInstructionMode kTestVectorInstructionMode,
            typename... ElementType,
            size_t... kResultsCount,
            size_t... kElementCount>
  void TestVectorInstructionInternal(
      uint32_t insn_bytes,
      __m128i dst_result,
      const __v2du (&source)[16],
      const ElementType (&... expected_result)[kResultsCount][kElementCount]) {
    auto Verify = [this, &source, dst_result](uint32_t insn_bytes,
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
      if (kTestVectorInstructionKind == TestVectorInstructionKind::kInteger) {
        // Set x1 for vx instructions.
        SetXReg<1>(state_.cpu, 0xaaaa'aaaa'aaaa'aaaa);
      } else {
        // We only support Float32/Float64 for float instructions, but there are conversion
        // instructions that work with double width floats.
        // These instructions never use float registers though and thus we don't need to store
        // anything into f1 register, if they are used.
        // For Float32/Float64 case we load 5.625 of the appropriate type into f1.
        ASSERT_LE(vsew, 3);
        if (vsew == 2) {
          SetFReg<1>(state_.cpu, 0xffff'ffff'40b4'0000);  // float 5.625
        } else if (vsew == 3) {
          SetFReg<1>(state_.cpu, 0x4016'8000'0000'0000);  // double 5.625
        }
      }
      for (uint8_t vlmul = 0; vlmul < vlmul_max; ++vlmul) {
        if constexpr (kTestVectorInstructionMode == TestVectorInstructionMode::kNarrowing ||
                      kTestVectorInstructionMode == TestVectorInstructionMode::kWidening) {
          // Incompatible vlmul for narrowing.
          if (vlmul == 3) {
            continue;
          }
        }
        for (uint8_t vta = 0; vta < 2; ++vta) {
          for (uint8_t vma = 0; vma < 2; ++vma) {
            auto [vlmax, vtype] =
                intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
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
            if (emul == 2) {
              state_.cpu.vstart = vlmax / 8;
              state_.cpu.vl = (vlmax * 5) / 8;
            } else {
              state_.cpu.vstart = 0;
              state_.cpu.vl = vlmax;
            }
            state_.cpu.vtype = vtype;

            // Set expected_result vector registers into 0b01010101… pattern.
            for (size_t index = 0; index < 8; ++index) {
              state_.cpu.v[8 + index] = SIMD128Register{dst_result}.Get<__uint128_t>();
            }

            state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
            EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

            // Values for inactive elements (i.e. corresponding mask bit is 0).
            __m128i expected_inactive[8];
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
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            ((dst_result & kFractionMaskInt8[3]) |
                             (SIMD128Register{expected_result[index]} & mask[index] &
                              ~kFractionMaskInt8[3]) |
                             (expected_inactive[index] & ~mask[index] & ~kFractionMaskInt8[3]))
                                .template Get<__uint128_t>());
                } else if (index == 2 && emul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            ((SIMD128Register{expected_result[index]} & mask[index] &
                              kFractionMaskInt8[3]) |
                             (expected_inactive[index] & ~mask[index] & kFractionMaskInt8[3]) |
                             ((vta ? kAgnosticResult : dst_result) & ~kFractionMaskInt8[3]))
                                .template Get<__uint128_t>());
                } else if (index == 3 && emul == 2 && vta) {
                  EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kAgnosticResult});
                } else if (index == 3 && emul == 2) {
                  EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{dst_result});
                } else {
                  EXPECT_EQ(state_.cpu.v[8 + index],
                            ((SIMD128Register{expected_result[index]} & mask[index]) |
                             ((expected_inactive[index] & ~mask[index])))
                                .template Get<__uint128_t>());
                }
              }
            } else {
              EXPECT_EQ(
                  state_.cpu.v[8],
                  ((SIMD128Register{expected_result[0]} & mask[0] & kFractionMaskInt8[emul - 4]) |
                   (expected_inactive[0] & ~mask[0] & kFractionMaskInt8[emul - 4]) |
                   ((vta ? kAgnosticResult : dst_result) & ~kFractionMaskInt8[emul - 4]))
                      .template Get<__uint128_t>());
            }

            if (emul == 2) {
              // Every vector instruction must set vstart to 0, but shouldn't touch vl.
              EXPECT_EQ(state_.cpu.vstart, 0);
              EXPECT_EQ(state_.cpu.vl, (vlmax * 5) / 8);
            }
          }
        }
      }
    };

    // Some instructions don't support use of mask register, but in these instructions bit
    // #25 is set.  This function doesn't support these. Verify that vm bit is not set.
    EXPECT_EQ(insn_bytes & (1 << 25), 0U);
    // Every insruction is tested with vm bit not set (and mask register used) and with vm bit
    // set (and mask register is not used).
    ((Verify(insn_bytes,
             BitUtilLog2(sizeof(ElementType)) -
                 (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening),
             8,
             expected_result,
             MaskForElem<ElementType>()),
      Verify((insn_bytes &
              ~(0x01f00000 * (kTestVectorInstructionMode == TestVectorInstructionMode::kVMerge))) |
                 (1 << 25),
             BitUtilLog2(sizeof(ElementType)) -
                 (kTestVectorInstructionMode == TestVectorInstructionMode::kWidening),
             8,
             expected_result,
             kNoMask)),
     ...);
  }

  void TestExtendingVectorInstruction(uint32_t insn_bytes,
                                      const __v8hu (&expected_result_int16)[8],
                                      const __v4su (&expected_result_int32)[8],
                                      const __v2du (&expected_result_int64)[8],
                                      const __v2du (&source)[16],
                                      const uint8_t factor) {
    auto Verify = [this, &source, &factor](uint32_t insn_bytes,
                                           uint8_t vsew,
                                           uint8_t vlmul_max,
                                           const auto& expected_result,
                                           auto mask) {
      CHECK((factor == 2) || (factor == 4) || (factor == 8));
      // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
      // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (uint8_t index = 0; index < (8 / factor); ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }
      for (uint8_t vlmul = 0; vlmul < vlmul_max; ++vlmul) {
        if (vlmul == 3) {
          continue;
        }
        for (uint8_t vta = 0; vta < 2; ++vta) {
          for (uint8_t vma = 0; vma < 2; ++vma) {
            auto [vlmax, vtype] =
                intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
            // Incompatible vsew and vlmax. Skip it.
            if (vlmax == 0) {
              continue;
            }
            // To make tests quick enough we don't test vstart and vl change with small register
            // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of
            // first
            // register and half of last register.
            // Don't use vlmul == 3 because that one may not be supported if instruction widens
            // the result.
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
            const size_t n = std::size(source) * 2;
            __m128i expected_inactive[n];
            // For most instructions, follow basic inactive processing rules based on vma flag.
            std::fill_n(expected_inactive, n, (vma ? kAgnosticResult : kUndisturbedResult));

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

    if (factor == 2) {
      Verify(insn_bytes, 1, 8, expected_result_int16, kMaskInt16);
      Verify(insn_bytes | (1 << 25), 1, 8, expected_result_int16, kNoMask);
    }
    if (factor == 2 || factor == 4) {
      Verify(insn_bytes, 2, 8, expected_result_int32, kMaskInt32);
      Verify(insn_bytes | (1 << 25), 2, 8, expected_result_int32, kNoMask);
    }
    Verify(insn_bytes, 3, 8, expected_result_int64, kMaskInt64);
    Verify(insn_bytes | (1 << 25), 3, 8, expected_result_int64, kNoMask);
  }

  void TestVectorMaskInstruction(uint8_t max_vstart,
                                 intrinsics::InactiveProcessing vma,
                                 uint32_t insn_bytes,
                                 const __v2du expected_result) {
    // Mask instructions don't look on vtype directly, but they still require valid one because it
    // affects vlmax;
    auto [vlmax, vtype] = intrinsics::Vsetvl(~0ULL, 3 | (static_cast<uint8_t>(vma) << 7));
    // We need mask with a few bits set for Vmsₓf instructions.  Inverse of normal kMask works.
    const __uint128_t mask = SIMD128Register{~kMask}.Get<__uint128_t>();
    const __uint128_t undisturbed = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
    const __uint128_t src1 = SIMD128Register{kVectorCalculationsSourceLegacy[0]}.Get<__uint128_t>();
    const __uint128_t src2 = SIMD128Register{kVectorCalculationsSourceLegacy[8]}.Get<__uint128_t>();
    const __uint128_t expected = SIMD128Register{expected_result}.Get<__uint128_t>();
    state_.cpu.vtype = vtype;
    for (uint8_t vl = 0; vl <= vlmax; ++vl) {
      state_.cpu.vl = vl;
      for (uint8_t vstart = 0; vstart <= max_vstart; ++vstart) {
        state_.cpu.vstart = vstart;
        // Set expected_result vector registers into 0b01010101… pattern.
        state_.cpu.v[0] = mask;
        state_.cpu.v[8] = undisturbed;
        state_.cpu.v[16] = src1;
        state_.cpu.v[24] = src2;

        state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
        EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

        for (uint8_t bit_pos = 0; bit_pos < 128; ++bit_pos) {
          __uint128_t bit = __uint128_t{1} << bit_pos;
          // When vstart ⩾ vl, there are no body elements, and no elements are updated in any
          // destinationvector register group, including that no tail elements are updated with
          // agnostic values.
          if (bit_pos < vstart || vstart >= vl) {
            EXPECT_EQ(state_.cpu.v[8] & bit, undisturbed & bit);
          } else if (bit_pos >= vl) {
            EXPECT_EQ(state_.cpu.v[8] & bit, bit);
          } else {
            EXPECT_EQ(state_.cpu.v[8] & bit, expected & bit);
          }
        }
      }
    }
  }

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
      static_assert(kDependentTypeFalse<ElementType>);
    }
  }

  void TestVectorMaskTargetInstruction(uint32_t insn_bytes,
                                       const uint32_t expected_result_int32,
                                       const uint16_t expected_result_int64,
                                       const __v2du (&source)[16]) {
    TestVectorMaskTargetInstruction(
        insn_bytes, source, expected_result_int32, expected_result_int64);
  }

  void TestVectorMaskTargetInstruction(uint32_t insn_bytes,
                                       const uint8_t (&expected_result_int8)[16],
                                       const uint64_t expected_result_int16,
                                       const uint32_t expected_result_int32,
                                       const uint16_t expected_result_int64,
                                       const __v2du (&source)[16]) {
    TestVectorMaskTargetInstruction(insn_bytes,
                                    source,
                                    expected_result_int8,
                                    expected_result_int16,
                                    expected_result_int32,
                                    expected_result_int64);
  }

  template <typename... ExpectedResultType>
  void TestVectorMaskTargetInstruction(uint32_t insn_bytes,
                                       const __v2du (&source)[16],
                                       const ExpectedResultType(&... expected_result)) {
    auto Verify = [this, &source](
                      uint32_t insn_bytes, uint8_t vsew, const auto& expected_result, auto mask) {
      // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
      // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (size_t index = 0; index < std::size(source); ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }
      // Set x1 for vx instructions.
      SetXReg<1>(state_.cpu, 0xaaaa'aaaa'aaaa'aaaa);
      // Set f1 for vf instructions.
      if (vsew == 2) {
        SetFReg<1>(state_.cpu, 0xffff'ffff'40b4'0000);  // float 5.625
      } else if (vsew == 3) {
        SetFReg<1>(state_.cpu, 0x4016'8000'0000'0000);  // double 5.625
      }
      for (uint8_t vlmul = 0; vlmul < 8; ++vlmul) {
        for (uint8_t vta = 0; vta < 2; ++vta) {  // vta should be ignored but we test both values!
          for (uint8_t vma = 0; vma < 2; ++vma) {
            auto [vlmax, vtype] =
                intrinsics::Vsetvl(~0ULL, (vma << 7) | (vta << 6) | (vsew << 3) | vlmul);
            // Incompatible vsew and vlmax. Skip it.
            if (vlmax == 0) {
              continue;
            }

            // To make tests quick enough we don't test vstart and vl change with small register
            // sets. Only with vlmul == 2 (4 registers) we set vstart and vl to skip half of first
            // register, last register and half of next-to last register.
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

            SIMD128Register expected_result_in_register(expected_result);
            if (vma == 0) {
              expected_result_in_register = (expected_result_in_register & SIMD128Register{mask}) |
                                            (kUndisturbedResult & ~SIMD128Register{mask});
            } else {
              expected_result_in_register = expected_result_in_register | ~SIMD128Register{mask};
            }
            // Mask registers are always processing tail like vta is set.
            if (vlmax != 128)
              expected_result_in_register |= std::get<0>(
                  intrinsics::MakeBitmaskFromVl((vlmul == 2) ? (vlmax * 5) / 8 : vlmax));
            if (vlmul == 2) {
              const auto [start_mask] = intrinsics::MakeBitmaskFromVl(vlmax / 8);
              expected_result_in_register = (SIMD128Register{kUndisturbedResult} & ~start_mask) |
                                            (expected_result_in_register & start_mask);
            }
            EXPECT_EQ(state_.cpu.v[8], expected_result_in_register);

            if (vlmul == 2) {
              // Every vector instruction must set vstart to 0, but shouldn't touch vl.
              EXPECT_EQ(state_.cpu.vstart, 0);
              EXPECT_EQ(state_.cpu.vl, (vlmax * 5) / 8);
            }
          }
        }
      }
    };

    ((Verify(insn_bytes,
             BitUtilLog2(sizeof(SIMD128Register) / sizeof(ExpectedResultType)),
             expected_result,
             kMask),
      Verify(insn_bytes | (1 << 25),
             BitUtilLog2(sizeof(SIMD128Register) / sizeof(ExpectedResultType)),
             expected_result,
             kNoMask[0])),
     ...);
  }

  void TestVXmXXsInstruction(uint32_t insn_bytes,
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
                                      const uint32_t (&expected_result_vd0_int32)[8],
                                      const uint64_t (&expected_result_vd0_int64)[8],
                                      const uint32_t (&expected_result_vd0_with_mask_int32)[8],
                                      const uint64_t (&expected_result_vd0_with_mask_int64)[8],
                                      const __v2du (&source)[16]) {
    TestVectorReductionInstruction(
        insn_bytes,
        source,
        std::tuple<const uint32_t(&)[8], const uint32_t(&)[8]>{expected_result_vd0_int32,
                                                               expected_result_vd0_with_mask_int32},
        std::tuple<const uint64_t(&)[8], const uint64_t(&)[8]>{
            expected_result_vd0_int64, expected_result_vd0_with_mask_int64});
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
    TestVectorReductionInstruction(
        insn_bytes,
        source,
        std::tuple<const uint8_t(&)[8], const uint8_t(&)[8]>{expected_result_vd0_int8,
                                                             expected_result_vd0_with_mask_int8},
        std::tuple<const uint16_t(&)[8], const uint16_t(&)[8]>{expected_result_vd0_int16,
                                                               expected_result_vd0_with_mask_int16},
        std::tuple<const uint32_t(&)[8], const uint32_t(&)[8]>{expected_result_vd0_int32,
                                                               expected_result_vd0_with_mask_int32},
        std::tuple<const uint64_t(&)[8], const uint64_t(&)[8]>{
            expected_result_vd0_int64, expected_result_vd0_with_mask_int64});
  }

  template <typename... ExpectedResultType>
  void TestVectorReductionInstruction(
      uint32_t insn_bytes,
      const __v2du (&source)[16],
      std::tuple<const ExpectedResultType (&)[8],
                 const ExpectedResultType (&)[8]>... expected_result) {
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
      ((Verify(insn_bytes,
               BitUtilLog2(sizeof(ExpectedResultType)),
               vlmul,
               std::get<1>(expected_result)[vlmul]),
        Verify(insn_bytes | (1 << 25),
               BitUtilLog2(sizeof(ExpectedResultType)),
               vlmul,
               std::get<0>(expected_result)[vlmul])),
       ...);
    }
  }

  void TestVectorFloatPermutationInstruction(uint32_t insn_bytes,
                                             const uint32_t (&expected_result_int32)[8][4],
                                             const uint64_t (&expected_result_int64)[8][2],
                                             const __v2du (&source)[16],
                                             uint8_t vlmul,
                                             uint64_t skip = 0,
                                             bool ignore_vma_for_last = false,
                                             bool last_elem_is_f1 = false) {
    TestVectorPermutationInstruction<TestVectorInstructionKind::kFloat>(insn_bytes,
                                                                        source,
                                                                        vlmul,
                                                                        skip,
                                                                        ignore_vma_for_last,
                                                                        last_elem_is_f1,
                                                                        /* regx1 */ 0x0,
                                                                        expected_result_int32,
                                                                        expected_result_int64);
  }

  void TestVectorPermutationInstruction(uint32_t insn_bytes,
                                        const uint8_t (&expected_result_int8)[8][16],
                                        const uint16_t (&expected_result_int16)[8][8],
                                        const uint32_t (&expected_result_int32)[8][4],
                                        const uint64_t (&expected_result_int64)[8][2],
                                        const __v2du (&source)[16],
                                        uint8_t vlmul,
                                        uint64_t regx1 = 0x0,
                                        uint64_t skip = 0,
                                        bool ignore_vma_for_last = false,
                                        bool last_elem_is_x1 = false) {
    TestVectorPermutationInstruction<TestVectorInstructionKind::kInteger>(insn_bytes,
                                                                          source,
                                                                          vlmul,
                                                                          skip,
                                                                          ignore_vma_for_last,
                                                                          last_elem_is_x1,
                                                                          regx1,
                                                                          expected_result_int8,
                                                                          expected_result_int16,
                                                                          expected_result_int32,
                                                                          expected_result_int64);
  }

  // Unlike regular arithmetic instructions, the result of a permutation
  // instruction depends also on vlmul.  Also, the vslideup specs mention that
  // the destination vector remains unchanged the first |offset| elements (in
  // effect, the offset acts akin to vstart), in those cases skip can be used
  // to specify how many elements' mask will be skipped (counting from the
  // beginning, should be the same as the offset).
  //
  // If |ignore_vma_for_last| is true, an inactive element at vl-1 will be
  // treated as if vma=0 (Undisturbed).
  // If |last_elem_is_reg1| is true, the last element of the vector in
  // expected_result (that is, at vl-1) will be expected to be the same as
  // |regx1| when VL < VMAX and said element is active.
  template <TestVectorInstructionKind kTestVectorInstructionKind,
            typename... ElementType,
            size_t... kResultsCount,
            size_t... kElementCount>
  void TestVectorPermutationInstruction(
      uint32_t insn_bytes,
      const __v2du (&source)[16],
      uint8_t vlmul,
      uint64_t skip,
      bool ignore_vma_for_last,
      bool last_elem_is_reg1,
      uint64_t regx1,
      const ElementType (&... expected_result)[kResultsCount][kElementCount]) {
    auto Verify = [this, &source, vlmul, regx1, skip, ignore_vma_for_last, last_elem_is_reg1](
                      uint32_t insn_bytes,
                      uint8_t vsew,
                      const auto& expected_result_raw,
                      auto mask) {
      // Mask register is, unconditionally, v0, and we need 8, 16, or 24 to handle full 8-registers
      // inputs thus we use v8..v15 for destination and place sources into v16..v23 and v24..v31.
      state_.cpu.v[0] = SIMD128Register{kMask}.Get<__uint128_t>();
      for (size_t index = 0; index < std::size(source); ++index) {
        state_.cpu.v[16 + index] = SIMD128Register{source[index]}.Get<__uint128_t>();
      }

      if constexpr (kTestVectorInstructionKind == TestVectorInstructionKind::kFloat) {
        UNUSED(regx1);
        // We only support Float32/Float64 for float instructions, but there are conversion
        // instructions that work with double width floats.
        // These instructions never use float registers though and thus we don't need to store
        // anything into f1 register, if they are used.
        // For Float32/Float64 case we load 5.625 of the appropriate type into f1.
        ASSERT_LE(vsew, 3);
        if (vsew == 2) {
          SetFReg<1>(state_.cpu, 0xffff'ffff'40b4'0000);  // float 5.625
        } else if (vsew == 3) {
          SetFReg<1>(state_.cpu, 0x4016'8000'0000'0000);  // double 5.625
        }
      } else {
        // Set x1 for vx instructions.
        SetXReg<1>(state_.cpu, regx1);
      }

      const size_t kElementSize = 1 << vsew;
      size_t num_regs = 1 << vlmul;
      if (vlmul > 3) {
        num_regs = 1;
      }
      // Values for which the mask is not applied due to being before the offset when doing
      // vslideup.
      SIMD128Register skip_mask[num_regs];
      int64_t toskip = skip;
      for (size_t index = 0; index < num_regs && toskip > 0; ++index) {
        size_t skip_bits = toskip * kElementSize * 8;
        skip_mask[index] =
            ~std::get<0>(intrinsics::MakeBitmaskFromVl(skip_bits > 128 ? 128 : skip_bits));
        toskip -= 16 / kElementSize;
      }

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
          // register, last register and half of next-to last register.
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

          // Set dst vector registers into 0b01010101… pattern.
          for (size_t index = 0; index < 8; ++index) {
            state_.cpu.v[8 + index] = SIMD128Register{kUndisturbedResult}.Get<__uint128_t>();
          }

          state_.cpu.insn_addr = ToGuestAddr(&insn_bytes);
          EXPECT_TRUE(RunOneInstruction(&state_, state_.cpu.insn_addr + 4));

          const size_t n = std::size(source);
          // Values for inactive elements (i.e. corresponding mask bit is 0).
          __m128i expected_inactive[n];
          // For most instructions, follow basic inactive processing rules based on vma flag.
          std::fill_n(expected_inactive, n, (vma ? kAgnosticResult : kUndisturbedResult));

          const size_t kElementsPerRegister = 16 / kElementSize;
          const size_t last_reg = (state_.cpu.vl - 1) / kElementsPerRegister;
          const size_t last_elem = (state_.cpu.vl - 1) % kElementsPerRegister;
          const auto [mask_for_vl] = intrinsics::MakeBitmaskFromVl(last_elem * kElementSize * 8);
          if (vma && ignore_vma_for_last) {
            // Set expected value for inactive element at vl-1 to Undisturbed.
            expected_inactive[last_reg] =
                ((expected_inactive[last_reg] & ~mask_for_vl) | (kUndisturbedResult & mask_for_vl))
                    .Get<__m128i>();
          }

          SIMD128Register expected_result[std::size(expected_result_raw)];
          for (size_t index = 0; index < std::size(expected_result_raw); ++index) {
            expected_result[index] = SIMD128Register{expected_result_raw[index]};
          }

          if (vlmul == 2 && last_elem_is_reg1) {
            switch (kElementSize) {
              case 1:
                expected_result[last_reg].template Set<uint8_t>(
                    static_cast<uint8_t>(GetXReg<1>(state_.cpu)), last_elem);
                break;
              case 2:
                expected_result[last_reg].template Set<uint16_t>(
                    static_cast<uint16_t>(GetXReg<1>(state_.cpu)), last_elem);
                break;
              case 4:
                if constexpr (kTestVectorInstructionKind == TestVectorInstructionKind::kFloat) {
                  expected_result[last_reg].template Set<uint32_t>(
                      static_cast<uint32_t>(GetFReg<1>(state_.cpu)), last_elem);
                } else {
                  expected_result[last_reg].template Set<uint32_t>(
                      static_cast<uint32_t>(GetXReg<1>(state_.cpu)), last_elem);
                }
                break;
              case 8:
                if constexpr (kTestVectorInstructionKind == TestVectorInstructionKind::kFloat) {
                  expected_result[last_reg].template Set<uint64_t>(
                      static_cast<uint64_t>(GetFReg<1>(state_.cpu)), last_elem);
                } else {
                  expected_result[last_reg].template Set<uint64_t>(
                      static_cast<uint64_t>(GetXReg<1>(state_.cpu)), last_elem);
                }
                break;
              default:
                FAIL() << "Element size is " << kElementSize;
            }
          }

          if (vlmul < 4) {
            for (size_t index = 0; index < num_regs; ++index) {
              if (index == 0 && vlmul == 2) {
                EXPECT_EQ(
                    state_.cpu.v[8 + index],
                    SIMD128Register{(kUndisturbedResult & kFractionMaskInt8[3]) |
                                    (expected_result[index] & (mask[index] | skip_mask[index]) &
                                     ~kFractionMaskInt8[3]) |
                                    (expected_inactive[index] & ~mask[index] & ~skip_mask[index] &
                                     ~kFractionMaskInt8[3])}
                        .Get<__uint128_t>());
              } else if (index == 2 && vlmul == 2) {
                EXPECT_EQ(
                    state_.cpu.v[8 + index],
                    SIMD128Register{
                        (expected_result[index] & (mask[index] | skip_mask[index]) &
                         kFractionMaskInt8[3]) |
                        (expected_inactive[index] & ~mask[index] & ~skip_mask[index] &
                         kFractionMaskInt8[3]) |
                        ((vta ? kAgnosticResult : kUndisturbedResult) & ~kFractionMaskInt8[3])}
                        .Get<__uint128_t>());
              } else if (index == 3 && vlmul == 2 && vta) {
                EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kAgnosticResult});
              } else if (index == 3 && vlmul == 2) {
                EXPECT_EQ(state_.cpu.v[8 + index], SIMD128Register{kUndisturbedResult});
              } else {
                EXPECT_EQ(
                    state_.cpu.v[8 + index],
                    SIMD128Register{(expected_result[index] & (mask[index] | skip_mask[index])) |
                                    (expected_inactive[index] & ~(mask[index] | skip_mask[index]))}
                        .Get<__uint128_t>());
              }
            }
          } else {
            __uint128_t v8 = state_.cpu.v[8];
            SIMD128Register affected_part{expected_result[0] &
                                          (mask[0] & kFractionMaskInt8[vlmul - 4] | skip_mask[0])};
            SIMD128Register masked_part{expected_inactive[0] & ~mask[0] & ~skip_mask[0] &
                                        kFractionMaskInt8[vlmul - 4]};
            SIMD128Register tail_part{(vta ? kAgnosticResult : kUndisturbedResult) &
                                      ~kFractionMaskInt8[vlmul - 4]};

            EXPECT_EQ(v8, (affected_part | masked_part | tail_part).Get<__uint128_t>());
          }

          if (vlmul == 2) {
            // Every vector instruction must set vstart to 0, but shouldn't touch vl.
            EXPECT_EQ(state_.cpu.vstart, 0);
            EXPECT_EQ(state_.cpu.vl, (vlmax * 5) / 8);
          }
        }
      }
    };

    // Test with and without masking enabled.
    (Verify(
         insn_bytes, BitUtilLog2(sizeof(ElementType)), expected_result, MaskForElem<ElementType>()),
     ...);
    (Verify(insn_bytes | (1 << 25), BitUtilLog2(sizeof(ElementType)), expected_result, kNoMask),
     ...);
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

      {0x9e0c'9a09'9604'9200, 0x8e1c'8a18'8614'8211},
      {0xbe2c'ba29'b624'b220, 0xae3c'aa38'a634'a231},
      {0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251},
      {0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271},
      {0x1e8c'1a89'1684'1280, 0x0e9c'0a98'0694'0291},
      {0x3eac'3aa9'36a4'32a0, 0x2ebc'2ab8'26b4'22b1},
      {0x5ecc'5ac9'56c4'52c0, 0x4edc'4ad8'46d4'42d1},
      {0x7eec'7ae9'76e4'72e0, 0x6efc'6af8'66f4'62f1},
  };

  static constexpr __v2du kVectorCalculationsSourceLegacy[16] = {
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
      {0xf005'f005'f005'f005, 0xffff'ffff'4040'4040},
      {0xffff'ffff'40b4'40b4, 0xffff'ffff'40b4'0000},
      {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
      {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
      {0xfff4'fff4'fff4'fff4, 0xfff6'fff6'fff6'fff6},
      {0xfff8'fff8'fff4'fff4, 0xfff5'fff5'fff5'fff5},
      {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
      {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9},

      {0xf005'f005'f005'f005, 0xffff'ffff'4040'4040},
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
  static constexpr __v16qu kFractionMaskInt8[5] = {
      {255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},        // Half of ⅛ reg = ¹⁄₁₆
      {255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},      // Half of ¼ reg = ⅛
      {255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // Half of ½ reg = ¼
      {255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0},  // Half of full reg = ½
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},  // Full reg
  };
  // Agnostic result is -1 on RISC-V, not 0.
  static constexpr __m128i kAgnosticResult = {-1, -1};
  // Undisturbed result is put in registers v8, v9, …, v15 and is expected to get read back.
  static constexpr __m128i kUndisturbedResult = {0x5555'5555'5555'5555, 0x5555'5555'5555'5555};
  // Note: permutation of indexes here is not entirely random. First 32 indexes are limited to 31
  // maximum and first 64 indexes are limited to 63. That way we can guarantee that 8byte elements
  // and 4byte elements wouldn't need to access area outside of our 256-byte buffer.
  static constexpr uint8_t kPermutedIndexes[128] = {
      1,   0,   3,  2,   7,   5,   4,   6,   9,   14,  15,  11,  13,  12, 8,  10,  30,  31,  17,
      22,  18,  26, 25,  19,  29,  28,  16,  21,  27,  24,  20,  23,  44, 50, 52,  34,  61,  38,
      54,  43,  42, 63,  57,  40,  36,  46,  39,  47,  35,  41,  62,  59, 60, 51,  55,  53,  33,
      32,  58,  49, 56,  37,  45,  48,  124, 92,  78,  101, 114, 89,  75, 64, 98,  112, 111, 118,
      121, 102, 73, 105, 109, 68,  103, 72,  110, 79,  119, 96,  123, 85, 90, 126, 66,  69,  120,
      97,  113, 76, 100, 67,  125, 117, 65,  84,  104, 122, 71,  81,  99, 70, 91,  86,  115, 127,
      77,  107, 74, 93,  80,  106, 87,  94,  83,  95,  116, 108, 82,  88};

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

  // FenceI explicitly not supported.
}

TEST_F(Riscv64InterpreterTest, SyscallWrite) {
  const char message[] = "Hello";
  // Prepare a pipe to write to.
  int pipefd[2];
  ASSERT_EQ(0, pipe(pipefd));

  // Only ecall instruction needs guest thread, since it involves pending signals manipulations.
  std::unique_ptr<GuestThread, decltype(&GuestThread::Destroy)> guest_thread(
      GuestThread::CreateForTest(&state_), GuestThread::Destroy);
  state_.thread = guest_thread.get();

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

TEST_F(Riscv64InterpreterTest, TestFPExceptions) {
  // Keep the same sort as Section 19 "Vector Instruction Listing".
  TestFPExceptions<intrinsics::Float32>(0x012d1557);  // Vfadd.vv v10, v18, v26, v0.t
  TestFPExceptions<intrinsics::Float64>(0x010c1457);  // Vfadd.vv v8, v16, v24, v0.t
  TestFPExceptions<intrinsics::Float32>(0x0120d557);  // Vfadd.vf v10, v18, f1, v0.t
  TestFPExceptions<intrinsics::Float64>(0x01015457);  // Vfadd.vf v8, v16, f2, v0.t
  TestFPExceptions<intrinsics::Float32>(0x092d1557);  // Vfsub.vv v10, v18, v26, v0.t
  TestFPExceptions<intrinsics::Float64>(0x090c1457);  // Vfsub.vv v8, v16, v24, v0.t
  TestFPExceptions<intrinsics::Float32>(0x0920d557);  // Vfsub.vf v10, v18, f1, v0.t
  TestFPExceptions<intrinsics::Float64>(0x09015457);  // Vfsub.vf v8, v16, f2, v0.t
  TestFPExceptions<intrinsics::Float32>(0x812d1557);  // Vfdiv.vv v10, v18, v26, v0.t
  TestFPExceptions<intrinsics::Float64>(0x810c1457);  // Vfdiv.vv v8, v16, v24, v0.t
  TestFPExceptions<intrinsics::Float32>(0x8120d557);  // Vfdiv.vf v10, v18, f1, v0.t
  TestFPExceptions<intrinsics::Float64>(0x81015457);  // Vfdiv.vf v8, v16, f2, v0.t
  TestFPExceptions<intrinsics::Float32>(0x912d1557);  // Vfmul.vv v10, v18, v26, v0.t
  TestFPExceptions<intrinsics::Float64>(0x910c1457);  // Vfmul.vv v8, v16, v24, v0.t
  TestFPExceptions<intrinsics::Float32>(0x9120d557);  // Vfmul.vf v10, v18, f1, v0.t
  TestFPExceptions<intrinsics::Float64>(0x91015457);  // Vfmul.vf v8, v16, f2, v0.t
  TestFPExceptions<intrinsics::Float32>(0x9d20d557);  // Vfrsub.vf v10, v18, f1, v0.t
  TestFPExceptions<intrinsics::Float64>(0x9d015457);  // Vfrsub.vf v8, v16, f2, v0.t
}

TEST_F(Riscv64InterpreterTest, TestVlXreXX) {
  TestVlXreXX<1>(0x2808407);   // vl1re8.v v8, (x1)
  TestVlXreXX<2>(0x22808407);  // vl2re8.v v8, (x1)
  TestVlXreXX<4>(0x62808407);  // vl4re8.v v8, (x1)
  TestVlXreXX<8>(0xe2808407);  // vl8re8.v v8, (x1)

  TestVlXreXX<1>(0x280d407);   // vl1re16.v v8, (x1)
  TestVlXreXX<2>(0x2280d407);  // vl2re16.v v8, (x1)
  TestVlXreXX<4>(0x6280d407);  // vl4re16.v v8, (x1)
  TestVlXreXX<8>(0xe280d407);  // vl8re16.v v8, (x1)

  TestVlXreXX<1>(0x280e407);   // vl1re32.v v8, (x1)
  TestVlXreXX<2>(0x2280e407);  // vl2re32.v v8, (x1)
  TestVlXreXX<4>(0x6280e407);  // vl4re32.v v8, (x1)
  TestVlXreXX<8>(0xe280e407);  // vl8re32.v v8, (x1)

  TestVlXreXX<1>(0x280f407);   // vl1re64.v v8, (x1)
  TestVlXreXX<2>(0x2280f407);  // vl2re64.v v8, (x1)
  TestVlXreXX<4>(0x6280f407);  // vl4re64.v v8, (x1)
  TestVlXreXX<8>(0xe280f407);  // vl8re64.v v8, (x1)
}

TEST_F(Riscv64InterpreterTest, TestVmXr) {
  TestVmvXr<1>(0x9f003457);  // Vmv1r.v v8, v16
  TestVmvXr<2>(0x9f00b457);  // Vmv2r.v v8, v16
  TestVmvXr<4>(0x9f01b457);  // Vmv4r.v v8, v16
  TestVmvXr<8>(0x9f03b457);  // Vmv8r.v v8, v16
}

TEST_F(Riscv64InterpreterTest, TestVfrsqrt7) {
  TestVectorFloatInstruction(0x4d821457,  // Vfrsqrt7.v v8, v24, v0.t
                             {{0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000, 0x7fc0'0000},
                              {0x53fb'8000, 0x4ff4'8000, 0x5bed'8000, 0x57e7'8000},
                              {0x43e2'0000, 0x3fdc'8000, 0x4bd7'8000, 0x47d3'0000},
                              {0x33ce'8000, 0x2fca'8000, 0x3bc6'8000, 0x37c3'0000},
                              {0x23bf'8000, 0x1fbc'8000, 0x2bb9'0000, 0x27b6'8000}},
                             {{0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x7ff8'0000'0000'0000, 0x7ff8'0000'0000'0000},
                              {0x50a1'1000'0000'0000, 0x5898'3000'0000'0000},
                              {0x4091'1000'0000'0000, 0x4888'2000'0000'0000},
                              {0x3081'0000'0000'0000, 0x3878'1000'0000'0000},
                              {0x2071'0000'0000'0000, 0x2868'0000'0000'0000}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfmvfs) {
  TestVfmvfs<intrinsics::Float32>(0x428010d7, 0xffff'ffff'8302'8100);  // Vfmv.f.s f1, v8
  TestVfmvfs<intrinsics::Float64>(0x428010d7, 0x8706'8504'8302'8100);  // Vfmv.f.s f1, v8
}

TEST_F(Riscv64InterpreterTest, TestVfmvsf) {
  TestVfmvsf<intrinsics::Float32>(0x4200d457,  // Vfmv.s.f v8, f1
                                  0xffff'ffff'40b4'0000,
                                  intrinsics::Float32{5.625f});
  TestVfmvsf<intrinsics::Float64>(0x4200d457,  // Vfmv.s.f v8, f1
                                  0x4016'8000'0000'0000,
                                  intrinsics::Float64{5.625});
}

TEST_F(Riscv64InterpreterTest, TestVmvsx) {
  TestVmvsx<Int8>(0x4200e457);   // Vmv.s.x v8, x1
  TestVmvsx<Int16>(0x4200e457);  // Vmv.s.x v8, x1
  TestVmvsx<Int32>(0x4200e457);  // Vmv.s.x v8, x1
  TestVmvsx<Int64>(0x4200e457);  // Vmv.s.x v8, x1
}

TEST_F(Riscv64InterpreterTest, TestVmvxs) {
  TestVmvxs<Int8>(0x428020d7, 0);                       // Vmv.x.s x1, v8
  TestVmvxs<Int16>(0x428020d7, 0xffff'ffff'ffff'8100);  // Vmv.x.s x1, v8
  TestVmvxs<Int32>(0x428020d7, 0xffff'ffff'8302'8100);  // Vmv.x.s x1, v8
  TestVmvxs<Int64>(0x428020d7, 0x8706'8504'8302'8100);  // Vmv.x.s x1, v8
}

TEST_F(Riscv64InterpreterTest, TestVsX) {
  TestVsX<1>(0x2808427);   // vs1r.v v8, (x1)
  TestVsX<2>(0x22808427);  // vs2r.v v8, (x1)
  TestVsX<4>(0x62808427);  // vs4r.v v8, (x1)
  TestVsX<8>(0xe2808427);  // vs8r.v v8, (x1)
}

TEST_F(Riscv64InterpreterTest, TestVlxsegXeiXX) {
  VlxsegXeiXX<UInt8, 1, 1>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                           {{129, 0, 131, 2, 135, 133, 4, 6, 137, 14, 143, 139, 141, 12, 8, 10}});
  VlxsegXeiXX<UInt8, 1, 2>(
      0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
      {{129, 0, 131, 2, 135, 133, 4, 6, 137, 14, 143, 139, 141, 12, 8, 10},
       {30, 159, 145, 22, 18, 26, 153, 147, 157, 28, 16, 149, 155, 24, 20, 151}});
  VlxsegXeiXX<UInt8, 1, 4>(
      0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
      {{129, 0, 131, 2, 135, 133, 4, 6, 137, 14, 143, 139, 141, 12, 8, 10},
       {30, 159, 145, 22, 18, 26, 153, 147, 157, 28, 16, 149, 155, 24, 20, 151},
       {44, 50, 52, 34, 189, 38, 54, 171, 42, 191, 185, 40, 36, 46, 167, 175},
       {163, 169, 62, 187, 60, 179, 183, 181, 161, 32, 58, 177, 56, 165, 173, 48}});
  VlxsegXeiXX<UInt8, 1, 8>(
      0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
      {{129, 0, 131, 2, 135, 133, 4, 6, 137, 14, 143, 139, 141, 12, 8, 10},
       {30, 159, 145, 22, 18, 26, 153, 147, 157, 28, 16, 149, 155, 24, 20, 151},
       {44, 50, 52, 34, 189, 38, 54, 171, 42, 191, 185, 40, 36, 46, 167, 175},
       {163, 169, 62, 187, 60, 179, 183, 181, 161, 32, 58, 177, 56, 165, 173, 48},
       {124, 92, 78, 229, 114, 217, 203, 64, 98, 112, 239, 118, 249, 102, 201, 233},
       {237, 68, 231, 72, 110, 207, 247, 96, 251, 213, 90, 126, 66, 197, 120, 225},
       {241, 76, 100, 195, 253, 245, 193, 84, 104, 122, 199, 209, 227, 70, 219, 86},
       {243, 255, 205, 235, 74, 221, 80, 106, 215, 94, 211, 223, 116, 108, 82, 88}});
  VlxsegXeiXX<UInt8, 2, 1>(
      0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
      {{2, 0, 6, 4, 14, 10, 8, 12, 18, 28, 30, 22, 26, 24, 16, 20},
       {131, 129, 135, 133, 143, 139, 137, 141, 147, 157, 159, 151, 155, 153, 145, 149}});
  VlxsegXeiXX<UInt8, 2, 2>(
      0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
      {{2, 0, 6, 4, 14, 10, 8, 12, 18, 28, 30, 22, 26, 24, 16, 20},
       {60, 62, 34, 44, 36, 52, 50, 38, 58, 56, 32, 42, 54, 48, 40, 46},
       {131, 129, 135, 133, 143, 139, 137, 141, 147, 157, 159, 151, 155, 153, 145, 149},
       {189, 191, 163, 173, 165, 181, 179, 167, 187, 185, 161, 171, 183, 177, 169, 175}});
  VlxsegXeiXX<UInt8, 2, 4>(
      0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
      {{2, 0, 6, 4, 14, 10, 8, 12, 18, 28, 30, 22, 26, 24, 16, 20},
       {60, 62, 34, 44, 36, 52, 50, 38, 58, 56, 32, 42, 54, 48, 40, 46},
       {88, 100, 104, 68, 122, 76, 108, 86, 84, 126, 114, 80, 72, 92, 78, 94},
       {70, 82, 124, 118, 120, 102, 110, 106, 66, 64, 116, 98, 112, 74, 90, 96},
       {131, 129, 135, 133, 143, 139, 137, 141, 147, 157, 159, 151, 155, 153, 145, 149},
       {189, 191, 163, 173, 165, 181, 179, 167, 187, 185, 161, 171, 183, 177, 169, 175},
       {217, 229, 233, 197, 251, 205, 237, 215, 213, 255, 243, 209, 201, 221, 207, 223},
       {199, 211, 253, 247, 249, 231, 239, 235, 195, 193, 245, 227, 241, 203, 219, 225}});
  VlxsegXeiXX<UInt8, 3, 1>(
      0x45008407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
      {{131, 0, 137, 6, 149, 143, 12, 18, 155, 42, 173, 161, 167, 36, 24, 30},
       {4, 129, 10, 135, 22, 16, 141, 147, 28, 171, 46, 34, 40, 165, 153, 159},
       {133, 2, 139, 8, 151, 145, 14, 20, 157, 44, 175, 163, 169, 38, 26, 32}});
  VlxsegXeiXX<UInt8, 3, 2>(
      0x45008407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
      {{131, 0, 137, 6, 149, 143, 12, 18, 155, 42, 173, 161, 167, 36, 24, 30},
       {90, 221, 179, 66, 54, 78, 203, 185, 215, 84, 48, 191, 209, 72, 60, 197},
       {4, 129, 10, 135, 22, 16, 141, 147, 28, 171, 46, 34, 40, 165, 153, 159},
       {219, 94, 52, 195, 183, 207, 76, 58, 88, 213, 177, 64, 82, 201, 189, 70},
       {133, 2, 139, 8, 151, 145, 14, 20, 157, 44, 175, 163, 169, 38, 26, 32},
       {92, 223, 181, 68, 56, 80, 205, 187, 217, 86, 50, 193, 211, 74, 62, 199}});
  VlxsegXeiXX<UInt8, 4, 1>(
      0x65008407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
      {{4, 0, 12, 8, 28, 20, 16, 24, 36, 56, 60, 44, 52, 48, 32, 40},
       {133, 129, 141, 137, 157, 149, 145, 153, 165, 185, 189, 173, 181, 177, 161, 169},
       {6, 2, 14, 10, 30, 22, 18, 26, 38, 58, 62, 46, 54, 50, 34, 42},
       {135, 131, 143, 139, 159, 151, 147, 155, 167, 187, 191, 175, 183, 179, 163, 171}});
  VlxsegXeiXX<UInt8, 4, 2>(
      0x65008407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
      {{4, 0, 12, 8, 28, 20, 16, 24, 36, 56, 60, 44, 52, 48, 32, 40},
       {120, 124, 68, 88, 72, 104, 100, 76, 116, 112, 64, 84, 108, 96, 80, 92},
       {133, 129, 141, 137, 157, 149, 145, 153, 165, 185, 189, 173, 181, 177, 161, 169},
       {249, 253, 197, 217, 201, 233, 229, 205, 245, 241, 193, 213, 237, 225, 209, 221},
       {6, 2, 14, 10, 30, 22, 18, 26, 38, 58, 62, 46, 54, 50, 34, 42},
       {122, 126, 70, 90, 74, 106, 102, 78, 118, 114, 66, 86, 110, 98, 82, 94},
       {135, 131, 143, 139, 159, 151, 147, 155, 167, 187, 191, 175, 183, 179, 163, 171},
       {251, 255, 199, 219, 203, 235, 231, 207, 247, 243, 195, 215, 239, 227, 211, 223}});
  VlxsegXeiXX<UInt8, 5, 1>(
      0x85008407,  // Vluxseg5ei8.v v8, (x1), v16, v0.t
      {{133, 0, 143, 10, 163, 153, 20, 30, 173, 70, 203, 183, 193, 60, 40, 50},
       {6, 129, 16, 139, 36, 26, 149, 159, 46, 199, 76, 56, 66, 189, 169, 179},
       {135, 2, 145, 12, 165, 155, 22, 32, 175, 72, 205, 185, 195, 62, 42, 52},
       {8, 131, 18, 141, 38, 28, 151, 161, 48, 201, 78, 58, 68, 191, 171, 181},
       {137, 4, 147, 14, 167, 157, 24, 34, 177, 74, 207, 187, 197, 64, 44, 54}});
  VlxsegXeiXX<UInt8, 6, 1>(
      0xa5008407,  // Vluxseg6ei8.v v8, (x1), v16, v0.t
      {{6, 0, 18, 12, 42, 30, 24, 36, 54, 84, 90, 66, 78, 72, 48, 60},
       {135, 129, 147, 141, 171, 159, 153, 165, 183, 213, 219, 195, 207, 201, 177, 189},
       {8, 2, 20, 14, 44, 32, 26, 38, 56, 86, 92, 68, 80, 74, 50, 62},
       {137, 131, 149, 143, 173, 161, 155, 167, 185, 215, 221, 197, 209, 203, 179, 191},
       {10, 4, 22, 16, 46, 34, 28, 40, 58, 88, 94, 70, 82, 76, 52, 64},
       {139, 133, 151, 145, 175, 163, 157, 169, 187, 217, 223, 199, 211, 205, 181, 193}});
  VlxsegXeiXX<UInt8, 7, 1>(
      0xc5008407,  // Vluxseg7ei8.v v8, (x1), v16, v0.t
      {{135, 0, 149, 14, 177, 163, 28, 42, 191, 98, 233, 205, 219, 84, 56, 70},
       {8, 129, 22, 143, 50, 36, 157, 171, 64, 227, 106, 78, 92, 213, 185, 199},
       {137, 2, 151, 16, 179, 165, 30, 44, 193, 100, 235, 207, 221, 86, 58, 72},
       {10, 131, 24, 145, 52, 38, 159, 173, 66, 229, 108, 80, 94, 215, 187, 201},
       {139, 4, 153, 18, 181, 167, 32, 46, 195, 102, 237, 209, 223, 88, 60, 74},
       {12, 133, 26, 147, 54, 40, 161, 175, 68, 231, 110, 82, 96, 217, 189, 203},
       {141, 6, 155, 20, 183, 169, 34, 48, 197, 104, 239, 211, 225, 90, 62, 76}});
  VlxsegXeiXX<UInt8, 8, 1>(
      0xe5008407,  // Vluxseg8ei8.v v8, (x1), v16, v0.t
      {{8, 0, 24, 16, 56, 40, 32, 48, 72, 112, 120, 88, 104, 96, 64, 80},
       {137, 129, 153, 145, 185, 169, 161, 177, 201, 241, 249, 217, 233, 225, 193, 209},
       {10, 2, 26, 18, 58, 42, 34, 50, 74, 114, 122, 90, 106, 98, 66, 82},
       {139, 131, 155, 147, 187, 171, 163, 179, 203, 243, 251, 219, 235, 227, 195, 211},
       {12, 4, 28, 20, 60, 44, 36, 52, 76, 116, 124, 92, 108, 100, 68, 84},
       {141, 133, 157, 149, 189, 173, 165, 181, 205, 245, 253, 221, 237, 229, 197, 213},
       {14, 6, 30, 22, 62, 46, 38, 54, 78, 118, 126, 94, 110, 102, 70, 86},
       {143, 135, 159, 151, 191, 175, 167, 183, 207, 247, 255, 223, 239, 231, 199, 215}});
  VlxsegXeiXX<UInt16, 1, 1>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8302, 0x8100, 0x8706, 0x8504, 0x8f0e, 0x8b0a, 0x8908, 0x8d0c}});
  VlxsegXeiXX<UInt16, 1, 2>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8302, 0x8100, 0x8706, 0x8504, 0x8f0e, 0x8b0a, 0x8908, 0x8d0c},
                             {0x9312, 0x9d1c, 0x9f1e, 0x9716, 0x9b1a, 0x9918, 0x9110, 0x9514}});
  VlxsegXeiXX<UInt16, 1, 4>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8302, 0x8100, 0x8706, 0x8504, 0x8f0e, 0x8b0a, 0x8908, 0x8d0c},
                             {0x9312, 0x9d1c, 0x9f1e, 0x9716, 0x9b1a, 0x9918, 0x9110, 0x9514},
                             {0xbd3c, 0xbf3e, 0xa322, 0xad2c, 0xa524, 0xb534, 0xb332, 0xa726},
                             {0xbb3a, 0xb938, 0xa120, 0xab2a, 0xb736, 0xb130, 0xa928, 0xaf2e}});
  VlxsegXeiXX<UInt16, 1, 8>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8302, 0x8100, 0x8706, 0x8504, 0x8f0e, 0x8b0a, 0x8908, 0x8d0c},
                             {0x9312, 0x9d1c, 0x9f1e, 0x9716, 0x9b1a, 0x9918, 0x9110, 0x9514},
                             {0xbd3c, 0xbf3e, 0xa322, 0xad2c, 0xa524, 0xb534, 0xb332, 0xa726},
                             {0xbb3a, 0xb938, 0xa120, 0xab2a, 0xb736, 0xb130, 0xa928, 0xaf2e},
                             {0xd958, 0xe564, 0xe968, 0xc544, 0xfb7a, 0xcd4c, 0xed6c, 0xd756},
                             {0xd554, 0xff7e, 0xf372, 0xd150, 0xc948, 0xdd5c, 0xcf4e, 0xdf5e},
                             {0xc746, 0xd352, 0xfd7c, 0xf776, 0xf978, 0xe766, 0xef6e, 0xeb6a},
                             {0xc342, 0xc140, 0xf574, 0xe362, 0xf170, 0xcb4a, 0xdb5a, 0xe160}});
  VlxsegXeiXX<UInt16, 2, 1>(0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8504, 0x8100, 0x8d0c, 0x8908, 0x9d1c, 0x9514, 0x9110, 0x9918},
                             {0x8706, 0x8302, 0x8f0e, 0x8b0a, 0x9f1e, 0x9716, 0x9312, 0x9b1a}});
  VlxsegXeiXX<UInt16, 2, 2>(0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8504, 0x8100, 0x8d0c, 0x8908, 0x9d1c, 0x9514, 0x9110, 0x9918},
                             {0xa524, 0xb938, 0xbd3c, 0xad2c, 0xb534, 0xb130, 0xa120, 0xa928},
                             {0x8706, 0x8302, 0x8f0e, 0x8b0a, 0x9f1e, 0x9716, 0x9312, 0x9b1a},
                             {0xa726, 0xbb3a, 0xbf3e, 0xaf2e, 0xb736, 0xb332, 0xa322, 0xab2a}});
  VlxsegXeiXX<UInt16, 2, 4>(0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8504, 0x8100, 0x8d0c, 0x8908, 0x9d1c, 0x9514, 0x9110, 0x9918},
                             {0xa524, 0xb938, 0xbd3c, 0xad2c, 0xb534, 0xb130, 0xa120, 0xa928},
                             {0xf978, 0xfd7c, 0xc544, 0xd958, 0xc948, 0xe968, 0xe564, 0xcd4c},
                             {0xf574, 0xf170, 0xc140, 0xd554, 0xed6c, 0xe160, 0xd150, 0xdd5c},
                             {0x8706, 0x8302, 0x8f0e, 0x8b0a, 0x9f1e, 0x9716, 0x9312, 0x9b1a},
                             {0xa726, 0xbb3a, 0xbf3e, 0xaf2e, 0xb736, 0xb332, 0xa322, 0xab2a},
                             {0xfb7a, 0xff7e, 0xc746, 0xdb5a, 0xcb4a, 0xeb6a, 0xe766, 0xcf4e},
                             {0xf776, 0xf372, 0xc342, 0xd756, 0xef6e, 0xe362, 0xd352, 0xdf5e}});
  VlxsegXeiXX<UInt16, 3, 1>(0x45008407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x8706, 0x8100, 0x9312, 0x8d0c, 0xab2a, 0x9f1e, 0x9918, 0xa524},
                             {0x8908, 0x8302, 0x9514, 0x8f0e, 0xad2c, 0xa120, 0x9b1a, 0xa726},
                             {0x8b0a, 0x8504, 0x9716, 0x9110, 0xaf2e, 0xa322, 0x9d1c, 0xa928}});
  VlxsegXeiXX<UInt16, 3, 2>(0x45008407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x8706, 0x8100, 0x9312, 0x8d0c, 0xab2a, 0x9f1e, 0x9918, 0xa524},
                             {0xb736, 0xd554, 0xdb5a, 0xc342, 0xcf4e, 0xc948, 0xb130, 0xbd3c},
                             {0x8908, 0x8302, 0x9514, 0x8f0e, 0xad2c, 0xa120, 0x9b1a, 0xa726},
                             {0xb938, 0xd756, 0xdd5c, 0xc544, 0xd150, 0xcb4a, 0xb332, 0xbf3e},
                             {0x8b0a, 0x8504, 0x9716, 0x9110, 0xaf2e, 0xa322, 0x9d1c, 0xa928},
                             {0xbb3a, 0xd958, 0xdf5e, 0xc746, 0xd352, 0xcd4c, 0xb534, 0xc140}});
  VlxsegXeiXX<UInt16, 4, 1>(0x65008407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0x8908, 0x8100, 0x9918, 0x9110, 0xb938, 0xa928, 0xa120, 0xb130},
                             {0x8b0a, 0x8302, 0x9b1a, 0x9312, 0xbb3a, 0xab2a, 0xa322, 0xb332},
                             {0x8d0c, 0x8504, 0x9d1c, 0x9514, 0xbd3c, 0xad2c, 0xa524, 0xb534},
                             {0x8f0e, 0x8706, 0x9f1e, 0x9716, 0xbf3e, 0xaf2e, 0xa726, 0xb736}});
  VlxsegXeiXX<UInt16, 4, 2>(0x65008407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0x8908, 0x8100, 0x9918, 0x9110, 0xb938, 0xa928, 0xa120, 0xb130},
                             {0xc948, 0xf170, 0xf978, 0xd958, 0xe968, 0xe160, 0xc140, 0xd150},
                             {0x8b0a, 0x8302, 0x9b1a, 0x9312, 0xbb3a, 0xab2a, 0xa322, 0xb332},
                             {0xcb4a, 0xf372, 0xfb7a, 0xdb5a, 0xeb6a, 0xe362, 0xc342, 0xd352},
                             {0x8d0c, 0x8504, 0x9d1c, 0x9514, 0xbd3c, 0xad2c, 0xa524, 0xb534},
                             {0xcd4c, 0xf574, 0xfd7c, 0xdd5c, 0xed6c, 0xe564, 0xc544, 0xd554},
                             {0x8f0e, 0x8706, 0x9f1e, 0x9716, 0xbf3e, 0xaf2e, 0xa726, 0xb736},
                             {0xcf4e, 0xf776, 0xff7e, 0xdf5e, 0xef6e, 0xe766, 0xc746, 0xd756}});
  VlxsegXeiXX<UInt16, 5, 1>(0x85008407,  // Vluxseg5ei8.v v8, (x1), v16, v0.t
                            {{0x8b0a, 0x8100, 0x9f1e, 0x9514, 0xc746, 0xb332, 0xa928, 0xbd3c},
                             {0x8d0c, 0x8302, 0xa120, 0x9716, 0xc948, 0xb534, 0xab2a, 0xbf3e},
                             {0x8f0e, 0x8504, 0xa322, 0x9918, 0xcb4a, 0xb736, 0xad2c, 0xc140},
                             {0x9110, 0x8706, 0xa524, 0x9b1a, 0xcd4c, 0xb938, 0xaf2e, 0xc342},
                             {0x9312, 0x8908, 0xa726, 0x9d1c, 0xcf4e, 0xbb3a, 0xb130, 0xc544}});
  VlxsegXeiXX<UInt16, 6, 1>(0xa5008407,  // Vluxseg6ei8.v v8, (x1), v16, v0.t
                            {{0x8d0c, 0x8100, 0xa524, 0x9918, 0xd554, 0xbd3c, 0xb130, 0xc948},
                             {0x8f0e, 0x8302, 0xa726, 0x9b1a, 0xd756, 0xbf3e, 0xb332, 0xcb4a},
                             {0x9110, 0x8504, 0xa928, 0x9d1c, 0xd958, 0xc140, 0xb534, 0xcd4c},
                             {0x9312, 0x8706, 0xab2a, 0x9f1e, 0xdb5a, 0xc342, 0xb736, 0xcf4e},
                             {0x9514, 0x8908, 0xad2c, 0xa120, 0xdd5c, 0xc544, 0xb938, 0xd150},
                             {0x9716, 0x8b0a, 0xaf2e, 0xa322, 0xdf5e, 0xc746, 0xbb3a, 0xd352}});
  VlxsegXeiXX<UInt16, 7, 1>(0xc5008407,  // Vluxseg7ei8.v v8, (x1), v16, v0.t
                            {{0x8f0e, 0x8100, 0xab2a, 0x9d1c, 0xe362, 0xc746, 0xb938, 0xd554},
                             {0x9110, 0x8302, 0xad2c, 0x9f1e, 0xe564, 0xc948, 0xbb3a, 0xd756},
                             {0x9312, 0x8504, 0xaf2e, 0xa120, 0xe766, 0xcb4a, 0xbd3c, 0xd958},
                             {0x9514, 0x8706, 0xb130, 0xa322, 0xe968, 0xcd4c, 0xbf3e, 0xdb5a},
                             {0x9716, 0x8908, 0xb332, 0xa524, 0xeb6a, 0xcf4e, 0xc140, 0xdd5c},
                             {0x9918, 0x8b0a, 0xb534, 0xa726, 0xed6c, 0xd150, 0xc342, 0xdf5e},
                             {0x9b1a, 0x8d0c, 0xb736, 0xa928, 0xef6e, 0xd352, 0xc544, 0xe160}});
  VlxsegXeiXX<UInt16, 8, 1>(0xe5008407,  // Vluxseg8ei8.v v8, (x1), v16, v0.t
                            {{0x9110, 0x8100, 0xb130, 0xa120, 0xf170, 0xd150, 0xc140, 0xe160},
                             {0x9312, 0x8302, 0xb332, 0xa322, 0xf372, 0xd352, 0xc342, 0xe362},
                             {0x9514, 0x8504, 0xb534, 0xa524, 0xf574, 0xd554, 0xc544, 0xe564},
                             {0x9716, 0x8706, 0xb736, 0xa726, 0xf776, 0xd756, 0xc746, 0xe766},
                             {0x9918, 0x8908, 0xb938, 0xa928, 0xf978, 0xd958, 0xc948, 0xe968},
                             {0x9b1a, 0x8b0a, 0xbb3a, 0xab2a, 0xfb7a, 0xdb5a, 0xcb4a, 0xeb6a},
                             {0x9d1c, 0x8d0c, 0xbd3c, 0xad2c, 0xfd7c, 0xdd5c, 0xcd4c, 0xed6c},
                             {0x9f1e, 0x8f0e, 0xbf3e, 0xaf2e, 0xff7e, 0xdf5e, 0xcf4e, 0xef6e}});
  VlxsegXeiXX<UInt32, 1, 1>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8706'8504, 0x8302'8100, 0x8f0e'8d0c, 0x8b0a'8908}});
  VlxsegXeiXX<UInt32, 1, 2>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8706'8504, 0x8302'8100, 0x8f0e'8d0c, 0x8b0a'8908},
                             {0x9f1e'9d1c, 0x9716'9514, 0x9312'9110, 0x9b1a'9918}});
  VlxsegXeiXX<UInt32, 1, 4>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8706'8504, 0x8302'8100, 0x8f0e'8d0c, 0x8b0a'8908},
                             {0x9f1e'9d1c, 0x9716'9514, 0x9312'9110, 0x9b1a'9918},
                             {0xa726'a524, 0xbb3a'b938, 0xbf3e'bd3c, 0xaf2e'ad2c},
                             {0xb736'b534, 0xb332'b130, 0xa322'a120, 0xab2a'a928}});
  VlxsegXeiXX<UInt32, 1, 8>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8706'8504, 0x8302'8100, 0x8f0e'8d0c, 0x8b0a'8908},
                             {0x9f1e'9d1c, 0x9716'9514, 0x9312'9110, 0x9b1a'9918},
                             {0xa726'a524, 0xbb3a'b938, 0xbf3e'bd3c, 0xaf2e'ad2c},
                             {0xb736'b534, 0xb332'b130, 0xa322'a120, 0xab2a'a928},
                             {0xfb7a'f978, 0xff7e'fd7c, 0xc746'c544, 0xdb5a'd958},
                             {0xcb4a'c948, 0xeb6a'e968, 0xe766'e564, 0xcf4e'cd4c},
                             {0xf776'f574, 0xf372'f170, 0xc342'c140, 0xd756'd554},
                             {0xef6e'ed6c, 0xe362'e160, 0xd352'd150, 0xdf5e'dd5c}});
  VlxsegXeiXX<UInt32, 2, 1>(0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8b0a'8908, 0x8302'8100, 0x9b1a'9918, 0x9312'9110},
                             {0x8f0e'8d0c, 0x8706'8504, 0x9f1e'9d1c, 0x9716'9514}});
  VlxsegXeiXX<UInt32, 2, 2>(0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8b0a'8908, 0x8302'8100, 0x9b1a'9918, 0x9312'9110},
                             {0xbb3a'b938, 0xab2a'a928, 0xa322'a120, 0xb332'b130},
                             {0x8f0e'8d0c, 0x8706'8504, 0x9f1e'9d1c, 0x9716'9514},
                             {0xbf3e'bd3c, 0xaf2e'ad2c, 0xa726'a524, 0xb736'b534}});
  VlxsegXeiXX<UInt32, 2, 4>(0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x8b0a'8908, 0x8302'8100, 0x9b1a'9918, 0x9312'9110},
                             {0xbb3a'b938, 0xab2a'a928, 0xa322'a120, 0xb332'b130},
                             {0xcb4a'c948, 0xf372'f170, 0xfb7a'f978, 0xdb5a'd958},
                             {0xeb6a'e968, 0xe362'e160, 0xc342'c140, 0xd352'd150},
                             {0x8f0e'8d0c, 0x8706'8504, 0x9f1e'9d1c, 0x9716'9514},
                             {0xbf3e'bd3c, 0xaf2e'ad2c, 0xa726'a524, 0xb736'b534},
                             {0xcf4e'cd4c, 0xf776'f574, 0xff7e'fd7c, 0xdf5e'dd5c},
                             {0xef6e'ed6c, 0xe766'e564, 0xc746'c544, 0xd756'd554}});
  VlxsegXeiXX<UInt32, 3, 1>(0x45008407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c, 0x8302'8100, 0xa726'a524, 0x9b1a'9918},
                             {0x9312'9110, 0x8706'8504, 0xab2a'a928, 0x9f1e'9d1c},
                             {0x9716'9514, 0x8b0a'8908, 0xaf2e'ad2c, 0xa322'a120}});
  VlxsegXeiXX<UInt32, 3, 2>(0x45008407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c, 0x8302'8100, 0xa726'a524, 0x9b1a'9918},
                             {0xd756'd554, 0xbf3e'bd3c, 0xb332'b130, 0xcb4a'c948},
                             {0x9312'9110, 0x8706'8504, 0xab2a'a928, 0x9f1e'9d1c},
                             {0xdb5a'd958, 0xc342'c140, 0xb736'b534, 0xcf4e'cd4c},
                             {0x9716'9514, 0x8b0a'8908, 0xaf2e'ad2c, 0xa322'a120},
                             {0xdf5e'dd5c, 0xc746'c544, 0xbb3a'b938, 0xd352'd150}});
  VlxsegXeiXX<UInt32, 4, 1>(0x65008407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0x9312'9110, 0x8302'8100, 0xb332'b130, 0xa322'a120},
                             {0x9716'9514, 0x8706'8504, 0xb736'b534, 0xa726'a524},
                             {0x9b1a'9918, 0x8b0a'8908, 0xbb3a'b938, 0xab2a'a928},
                             {0x9f1e'9d1c, 0x8f0e'8d0c, 0xbf3e'bd3c, 0xaf2e'ad2c}});
  VlxsegXeiXX<UInt32, 4, 2>(0x65008407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0x9312'9110, 0x8302'8100, 0xb332'b130, 0xa322'a120},
                             {0xf372'f170, 0xd352'd150, 0xc342'c140, 0xe362'e160},
                             {0x9716'9514, 0x8706'8504, 0xb736'b534, 0xa726'a524},
                             {0xf776'f574, 0xd756'd554, 0xc746'c544, 0xe766'e564},
                             {0x9b1a'9918, 0x8b0a'8908, 0xbb3a'b938, 0xab2a'a928},
                             {0xfb7a'f978, 0xdb5a'd958, 0xcb4a'c948, 0xeb6a'e968},
                             {0x9f1e'9d1c, 0x8f0e'8d0c, 0xbf3e'bd3c, 0xaf2e'ad2c},
                             {0xff7e'fd7c, 0xdf5e'dd5c, 0xcf4e'cd4c, 0xef6e'ed6c}});
  VlxsegXeiXX<UInt32, 5, 1>(0x85008407,  // Vluxseg5ei8.v v8, (x1), v16, v0.t
                            {{0x9716'9514, 0x8302'8100, 0xbf3e'bd3c, 0xab2a'a928},
                             {0x9b1a'9918, 0x8706'8504, 0xc342'c140, 0xaf2e'ad2c},
                             {0x9f1e'9d1c, 0x8b0a'8908, 0xc746'c544, 0xb332'b130},
                             {0xa322'a120, 0x8f0e'8d0c, 0xcb4a'c948, 0xb736'b534},
                             {0xa726'a524, 0x9312'9110, 0xcf4e'cd4c, 0xbb3a'b938}});
  VlxsegXeiXX<UInt32, 6, 1>(0xa5008407,  // Vluxseg6ei8.v v8, (x1), v16, v0.t
                            {{0x9b1a'9918, 0x8302'8100, 0xcb4a'c948, 0xb332'b130},
                             {0x9f1e'9d1c, 0x8706'8504, 0xcf4e'cd4c, 0xb736'b534},
                             {0xa322'a120, 0x8b0a'8908, 0xd352'd150, 0xbb3a'b938},
                             {0xa726'a524, 0x8f0e'8d0c, 0xd756'd554, 0xbf3e'bd3c},
                             {0xab2a'a928, 0x9312'9110, 0xdb5a'd958, 0xc342'c140},
                             {0xaf2e'ad2c, 0x9716'9514, 0xdf5e'dd5c, 0xc746'c544}});
  VlxsegXeiXX<UInt32, 7, 1>(0xc5008407,  // Vluxseg7ei8.v v8, (x1), v16, v0.t
                            {{0x9f1e'9d1c, 0x8302'8100, 0xd756'd554, 0xbb3a'b938},
                             {0xa322'a120, 0x8706'8504, 0xdb5a'd958, 0xbf3e'bd3c},
                             {0xa726'a524, 0x8b0a'8908, 0xdf5e'dd5c, 0xc342'c140},
                             {0xab2a'a928, 0x8f0e'8d0c, 0xe362'e160, 0xc746'c544},
                             {0xaf2e'ad2c, 0x9312'9110, 0xe766'e564, 0xcb4a'c948},
                             {0xb332'b130, 0x9716'9514, 0xeb6a'e968, 0xcf4e'cd4c},
                             {0xb736'b534, 0x9b1a'9918, 0xef6e'ed6c, 0xd352'd150}});
  VlxsegXeiXX<UInt32, 8, 1>(0xe5008407,  // Vluxseg8ei8.v v8, (x1), v16, v0.t
                            {{0xa322'a120, 0x8302'8100, 0xe362'e160, 0xc342'c140},
                             {0xa726'a524, 0x8706'8504, 0xe766'e564, 0xc746'c544},
                             {0xab2a'a928, 0x8b0a'8908, 0xeb6a'e968, 0xcb4a'c948},
                             {0xaf2e'ad2c, 0x8f0e'8d0c, 0xef6e'ed6c, 0xcf4e'cd4c},
                             {0xb332'b130, 0x9312'9110, 0xf372'f170, 0xd352'd150},
                             {0xb736'b534, 0x9716'9514, 0xf776'f574, 0xd756'd554},
                             {0xbb3a'b938, 0x9b1a'9918, 0xfb7a'f978, 0xdb5a'd958},
                             {0xbf3e'bd3c, 0x9f1e'9d1c, 0xff7e'fd7c, 0xdf5e'dd5c}});
  VlxsegXeiXX<UInt64, 1, 1>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100}});
  VlxsegXeiXX<UInt64, 1, 2>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100},
                             {0x9f1e'9d1c'9b1a'9918, 0x9716'9514'9312'9110}});
  VlxsegXeiXX<UInt64, 1, 4>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100},
                             {0x9f1e'9d1c'9b1a'9918, 0x9716'9514'9312'9110},
                             {0xbf3e'bd3c'bb3a'b938, 0xaf2e'ad2c'ab2a'a928},
                             {0xa726'a524'a322'a120, 0xb736'b534'b332'b130}});
  VlxsegXeiXX<UInt64, 1, 8>(0x05008407,  // Vluxei8.v v8, (x1), v16, v0.t
                            {{0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100},
                             {0x9f1e'9d1c'9b1a'9918, 0x9716'9514'9312'9110},
                             {0xbf3e'bd3c'bb3a'b938, 0xaf2e'ad2c'ab2a'a928},
                             {0xa726'a524'a322'a120, 0xb736'b534'b332'b130},
                             {0xcf4e'cd4c'cb4a'c948, 0xf776'f574'f372'f170},
                             {0xff7e'fd7c'fb7a'f978, 0xdf5e'dd5c'db5a'd958},
                             {0xef6e'ed6c'eb6a'e968, 0xe766'e564'e362'e160},
                             {0xc746'c544'c342'c140, 0xd756'd554'd352'd150}});
  VlxsegXeiXX<UInt64, 2, 1>(0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x9716'9514'9312'9110, 0x8706'8504'8302'8100},
                             {0x9f1e'9d1c'9b1a'9918, 0x8f0e'8d0c'8b0a'8908}});
  VlxsegXeiXX<UInt64, 2, 2>(0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x9716'9514'9312'9110, 0x8706'8504'8302'8100},
                             {0xb736'b534'b332'b130, 0xa726'a524'a322'a120},
                             {0x9f1e'9d1c'9b1a'9918, 0x8f0e'8d0c'8b0a'8908},
                             {0xbf3e'bd3c'bb3a'b938, 0xaf2e'ad2c'ab2a'a928}});
  VlxsegXeiXX<UInt64, 2, 4>(0x25008407,  // Vluxseg2ei8.v v8, (x1), v16, v0.t
                            {{0x9716'9514'9312'9110, 0x8706'8504'8302'8100},
                             {0xb736'b534'b332'b130, 0xa726'a524'a322'a120},
                             {0xf776'f574'f372'f170, 0xd756'd554'd352'd150},
                             {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                             {0x9f1e'9d1c'9b1a'9918, 0x8f0e'8d0c'8b0a'8908},
                             {0xbf3e'bd3c'bb3a'b938, 0xaf2e'ad2c'ab2a'a928},
                             {0xff7e'fd7c'fb7a'f978, 0xdf5e'dd5c'db5a'd958},
                             {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968}});
  VlxsegXeiXX<UInt64, 3, 1>(0x45008407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x9f1e'9d1c'9b1a'9918, 0x8706'8504'8302'8100},
                             {0xa726'a524'a322'a120, 0x8f0e'8d0c'8b0a'8908},
                             {0xaf2e'ad2c'ab2a'a928, 0x9716'9514'9312'9110}});
  VlxsegXeiXX<UInt64, 3, 2>(0x45008407,  // Vluxseg3ei8.v v8, (x1), v16, v0.t
                            {{0x9f1e'9d1c'9b1a'9918, 0x8706'8504'8302'8100},
                             {0xcf4e'cd4c'cb4a'c948, 0xb736'b534'b332'b130},
                             {0xa726'a524'a322'a120, 0x8f0e'8d0c'8b0a'8908},
                             {0xd756'd554'd352'd150, 0xbf3e'bd3c'bb3a'b938},
                             {0xaf2e'ad2c'ab2a'a928, 0x9716'9514'9312'9110},
                             {0xdf5e'dd5c'db5a'd958, 0xc746'c544'c342'c140}});
  VlxsegXeiXX<UInt64, 4, 1>(0x65008407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0xa726'a524'a322'a120, 0x8706'8504'8302'8100},
                             {0xaf2e'ad2c'ab2a'a928, 0x8f0e'8d0c'8b0a'8908},
                             {0xb736'b534'b332'b130, 0x9716'9514'9312'9110},
                             {0xbf3e'bd3c'bb3a'b938, 0x9f1e'9d1c'9b1a'9918}});
  VlxsegXeiXX<UInt64, 4, 2>(0x65008407,  // Vluxseg4ei8.v v8, (x1), v16, v0.t
                            {{0xa726'a524'a322'a120, 0x8706'8504'8302'8100},
                             {0xe766'e564'e362'e160, 0xc746'c544'c342'c140},
                             {0xaf2e'ad2c'ab2a'a928, 0x8f0e'8d0c'8b0a'8908},
                             {0xef6e'ed6c'eb6a'e968, 0xcf4e'cd4c'cb4a'c948},
                             {0xb736'b534'b332'b130, 0x9716'9514'9312'9110},
                             {0xf776'f574'f372'f170, 0xd756'd554'd352'd150},
                             {0xbf3e'bd3c'bb3a'b938, 0x9f1e'9d1c'9b1a'9918},
                             {0xff7e'fd7c'fb7a'f978, 0xdf5e'dd5c'db5a'd958}});
  VlxsegXeiXX<UInt64, 5, 1>(0x85008407,  // Vluxseg5ei8.v v8, (x1), v16, v0.t
                            {{0xaf2e'ad2c'ab2a'a928, 0x8706'8504'8302'8100},
                             {0xb736'b534'b332'b130, 0x8f0e'8d0c'8b0a'8908},
                             {0xbf3e'bd3c'bb3a'b938, 0x9716'9514'9312'9110},
                             {0xc746'c544'c342'c140, 0x9f1e'9d1c'9b1a'9918},
                             {0xcf4e'cd4c'cb4a'c948, 0xa726'a524'a322'a120}});
  VlxsegXeiXX<UInt64, 6, 1>(0xa5008407,  // Vluxseg6ei8.v v8, (x1), v16, v0.t
                            {{0xb736'b534'b332'b130, 0x8706'8504'8302'8100},
                             {0xbf3e'bd3c'bb3a'b938, 0x8f0e'8d0c'8b0a'8908},
                             {0xc746'c544'c342'c140, 0x9716'9514'9312'9110},
                             {0xcf4e'cd4c'cb4a'c948, 0x9f1e'9d1c'9b1a'9918},
                             {0xd756'd554'd352'd150, 0xa726'a524'a322'a120},
                             {0xdf5e'dd5c'db5a'd958, 0xaf2e'ad2c'ab2a'a928}});
  VlxsegXeiXX<UInt64, 7, 1>(0xc5008407,  // Vluxseg7ei8.v v8, (x1), v16, v0.t
                            {{0xbf3e'bd3c'bb3a'b938, 0x8706'8504'8302'8100},
                             {0xc746'c544'c342'c140, 0x8f0e'8d0c'8b0a'8908},
                             {0xcf4e'cd4c'cb4a'c948, 0x9716'9514'9312'9110},
                             {0xd756'd554'd352'd150, 0x9f1e'9d1c'9b1a'9918},
                             {0xdf5e'dd5c'db5a'd958, 0xa726'a524'a322'a120},
                             {0xe766'e564'e362'e160, 0xaf2e'ad2c'ab2a'a928},
                             {0xef6e'ed6c'eb6a'e968, 0xb736'b534'b332'b130}});
  VlxsegXeiXX<UInt64, 8, 1>(0xe5008407,  // Vluxseg8ei8.v v8, (x1), v16, v0.t
                            {{0xc746'c544'c342'c140, 0x8706'8504'8302'8100},
                             {0xcf4e'cd4c'cb4a'c948, 0x8f0e'8d0c'8b0a'8908},
                             {0xd756'd554'd352'd150, 0x9716'9514'9312'9110},
                             {0xdf5e'dd5c'db5a'd958, 0x9f1e'9d1c'9b1a'9918},
                             {0xe766'e564'e362'e160, 0xa726'a524'a322'a120},
                             {0xef6e'ed6c'eb6a'e968, 0xaf2e'ad2c'ab2a'a928},
                             {0xf776'f574'f372'f170, 0xb736'b534'b332'b130},
                             {0xff7e'fd7c'fb7a'f978, 0xbf3e'bd3c'bb3a'b938}});
}

TEST_F(Riscv64InterpreterTest, TestVlsegXeXX) {
  TestVlsegXeXX<UInt8, 1, 1>(0x000008407,  // vlse8.v v8, (x1), v0.t
                             {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143}});
  TestVlsegXeXX<UInt8, 1, 2>(
      0x000008407,  // vlse8.v v8, (x1), v0.t
      {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
       {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159}});
  TestVlsegXeXX<UInt8, 1, 4>(
      0x000008407,  // vlse8.v v8, (x1), v0.t
      {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
       {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
       {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 171, 44, 173, 46, 175},
       {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191}});
  TestVlsegXeXX<UInt8, 1, 8>(
      0x000008407,  // vlse8.v v8, (x1), v0.t
      {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
       {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
       {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 171, 44, 173, 46, 175},
       {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191},
       {64, 193, 66, 195, 68, 197, 70, 199, 72, 201, 74, 203, 76, 205, 78, 207},
       {80, 209, 82, 211, 84, 213, 86, 215, 88, 217, 90, 219, 92, 221, 94, 223},
       {96, 225, 98, 227, 100, 229, 102, 231, 104, 233, 106, 235, 108, 237, 110, 239},
       {112, 241, 114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255}});
  TestVlsegXeXX<UInt8, 2, 1>(
      0x20008407,  // vlseg2e8.v v8, (x1), v0.t
      {{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
       {129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159}});
  TestVlsegXeXX<UInt8, 2, 2>(
      0x20008407,  // vlseg2e8.v v8, (x1), v0.t
      {{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62},
       {129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159},
       {161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191}});
  TestVlsegXeXX<UInt8, 2, 4>(
      0x20008407,  // vlseg2e8.v v8, (x1), v0.t
      {{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62},
       {64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94},
       {96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126},
       {129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159},
       {161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191},
       {193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223},
       {225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255}});
  TestVlsegXeXX<UInt8, 3, 1>(
      0x40008407,  // vlseg3e8.v v8, (x1), v0.t
      {{0, 131, 6, 137, 12, 143, 18, 149, 24, 155, 30, 161, 36, 167, 42, 173},
       {129, 4, 135, 10, 141, 16, 147, 22, 153, 28, 159, 34, 165, 40, 171, 46},
       {2, 133, 8, 139, 14, 145, 20, 151, 26, 157, 32, 163, 38, 169, 44, 175}});
  TestVlsegXeXX<UInt8, 3, 2>(
      0x40008407,  // vlseg3e8.v v8, (x1), v0.t
      {{0, 131, 6, 137, 12, 143, 18, 149, 24, 155, 30, 161, 36, 167, 42, 173},
       {48, 179, 54, 185, 60, 191, 66, 197, 72, 203, 78, 209, 84, 215, 90, 221},
       {129, 4, 135, 10, 141, 16, 147, 22, 153, 28, 159, 34, 165, 40, 171, 46},
       {177, 52, 183, 58, 189, 64, 195, 70, 201, 76, 207, 82, 213, 88, 219, 94},
       {2, 133, 8, 139, 14, 145, 20, 151, 26, 157, 32, 163, 38, 169, 44, 175},
       {50, 181, 56, 187, 62, 193, 68, 199, 74, 205, 80, 211, 86, 217, 92, 223}});
  TestVlsegXeXX<UInt8, 4, 1>(
      0x60008407,  // vlseg4e8.v v8, (x1), v0.t
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
       {131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191}});
  TestVlsegXeXX<UInt8, 4, 2>(
      0x60008407,  // vlseg4e8.v v8, (x1), v0.t
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
       {66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126},
       {131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191},
       {195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255}});
  TestVlsegXeXX<UInt8, 5, 1>(
      0x80008407,  // vlseg5e8.v v8, (x1), v0.t
      {{0, 133, 10, 143, 20, 153, 30, 163, 40, 173, 50, 183, 60, 193, 70, 203},
       {129, 6, 139, 16, 149, 26, 159, 36, 169, 46, 179, 56, 189, 66, 199, 76},
       {2, 135, 12, 145, 22, 155, 32, 165, 42, 175, 52, 185, 62, 195, 72, 205},
       {131, 8, 141, 18, 151, 28, 161, 38, 171, 48, 181, 58, 191, 68, 201, 78},
       {4, 137, 14, 147, 24, 157, 34, 167, 44, 177, 54, 187, 64, 197, 74, 207}});
  TestVlsegXeXX<UInt8, 6, 1>(
      0xa0008407,  // vlseg6e8.v v8, (x1), v0.t
      {{0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90},
       {129, 135, 141, 147, 153, 159, 165, 171, 177, 183, 189, 195, 201, 207, 213, 219},
       {2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92},
       {131, 137, 143, 149, 155, 161, 167, 173, 179, 185, 191, 197, 203, 209, 215, 221},
       {4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94},
       {133, 139, 145, 151, 157, 163, 169, 175, 181, 187, 193, 199, 205, 211, 217, 223}});
  TestVlsegXeXX<UInt8, 7, 1>(
      0xc0008407,  // vlseg7e8.v v8, (x1), v0.t
      {{0, 135, 14, 149, 28, 163, 42, 177, 56, 191, 70, 205, 84, 219, 98, 233},
       {129, 8, 143, 22, 157, 36, 171, 50, 185, 64, 199, 78, 213, 92, 227, 106},
       {2, 137, 16, 151, 30, 165, 44, 179, 58, 193, 72, 207, 86, 221, 100, 235},
       {131, 10, 145, 24, 159, 38, 173, 52, 187, 66, 201, 80, 215, 94, 229, 108},
       {4, 139, 18, 153, 32, 167, 46, 181, 60, 195, 74, 209, 88, 223, 102, 237},
       {133, 12, 147, 26, 161, 40, 175, 54, 189, 68, 203, 82, 217, 96, 231, 110},
       {6, 141, 20, 155, 34, 169, 48, 183, 62, 197, 76, 211, 90, 225, 104, 239}});
  TestVlsegXeXX<UInt8, 8, 1>(
      0xe0008407,  // vlseg8e8.v v8, (x1), v0.t
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249},
       {2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122},
       {131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251},
       {4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124},
       {133, 141, 149, 157, 165, 173, 181, 189, 197, 205, 213, 221, 229, 237, 245, 253},
       {6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126},
       {135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239, 247, 255}});
  TestVlsegXeXX<UInt16, 1, 1>(0x000d407,  // vle16.v v8, (x1), v0.t
                              {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e}});
  TestVlsegXeXX<UInt16, 1, 2>(0x000d407,  // vle16.v v8, (x1), v0.t
                              {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
                               {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e}});
  TestVlsegXeXX<UInt16, 1, 4>(0x000d407,  // vle16.v v8, (x1), v0.t
                              {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
                               {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
                               {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
                               {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e}});
  TestVlsegXeXX<UInt16, 1, 8>(0x000d407,  // vle16.v v8, (x1), v0.t
                              {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
                               {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
                               {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
                               {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
                               {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
                               {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
                               {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
                               {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}});
  TestVlsegXeXX<UInt16, 2, 1>(0x2000d407,  // vlseg2e16.v v8, (x1), v0.t
                              {{0x8100, 0x8504, 0x8908, 0x8d0c, 0x9110, 0x9514, 0x9918, 0x9d1c},
                               {0x8302, 0x8706, 0x8b0a, 0x8f0e, 0x9312, 0x9716, 0x9b1a, 0x9f1e}});
  TestVlsegXeXX<UInt16, 2, 2>(0x2000d407,  // vlseg2e16.v v8, (x1), v0.t
                              {{0x8100, 0x8504, 0x8908, 0x8d0c, 0x9110, 0x9514, 0x9918, 0x9d1c},
                               {0xa120, 0xa524, 0xa928, 0xad2c, 0xb130, 0xb534, 0xb938, 0xbd3c},
                               {0x8302, 0x8706, 0x8b0a, 0x8f0e, 0x9312, 0x9716, 0x9b1a, 0x9f1e},
                               {0xa322, 0xa726, 0xab2a, 0xaf2e, 0xb332, 0xb736, 0xbb3a, 0xbf3e}});
  TestVlsegXeXX<UInt16, 2, 4>(0x2000d407,  // vlseg2e16.v v8, (x1), v0.t
                              {{0x8100, 0x8504, 0x8908, 0x8d0c, 0x9110, 0x9514, 0x9918, 0x9d1c},
                               {0xa120, 0xa524, 0xa928, 0xad2c, 0xb130, 0xb534, 0xb938, 0xbd3c},
                               {0xc140, 0xc544, 0xc948, 0xcd4c, 0xd150, 0xd554, 0xd958, 0xdd5c},
                               {0xe160, 0xe564, 0xe968, 0xed6c, 0xf170, 0xf574, 0xf978, 0xfd7c},
                               {0x8302, 0x8706, 0x8b0a, 0x8f0e, 0x9312, 0x9716, 0x9b1a, 0x9f1e},
                               {0xa322, 0xa726, 0xab2a, 0xaf2e, 0xb332, 0xb736, 0xbb3a, 0xbf3e},
                               {0xc342, 0xc746, 0xcb4a, 0xcf4e, 0xd352, 0xd756, 0xdb5a, 0xdf5e},
                               {0xe362, 0xe766, 0xeb6a, 0xef6e, 0xf372, 0xf776, 0xfb7a, 0xff7e}});
  TestVlsegXeXX<UInt16, 3, 1>(0x4000d407,  // vlseg3e16.v v8, (x1), v0.t
                              {{0x8100, 0x8706, 0x8d0c, 0x9312, 0x9918, 0x9f1e, 0xa524, 0xab2a},
                               {0x8302, 0x8908, 0x8f0e, 0x9514, 0x9b1a, 0xa120, 0xa726, 0xad2c},
                               {0x8504, 0x8b0a, 0x9110, 0x9716, 0x9d1c, 0xa322, 0xa928, 0xaf2e}});
  TestVlsegXeXX<UInt16, 3, 2>(0x4000d407,  // vlseg3e16.v v8, (x1), v0.t
                              {{0x8100, 0x8706, 0x8d0c, 0x9312, 0x9918, 0x9f1e, 0xa524, 0xab2a},
                               {0xb130, 0xb736, 0xbd3c, 0xc342, 0xc948, 0xcf4e, 0xd554, 0xdb5a},
                               {0x8302, 0x8908, 0x8f0e, 0x9514, 0x9b1a, 0xa120, 0xa726, 0xad2c},
                               {0xb332, 0xb938, 0xbf3e, 0xc544, 0xcb4a, 0xd150, 0xd756, 0xdd5c},
                               {0x8504, 0x8b0a, 0x9110, 0x9716, 0x9d1c, 0xa322, 0xa928, 0xaf2e},
                               {0xb534, 0xbb3a, 0xc140, 0xc746, 0xcd4c, 0xd352, 0xd958, 0xdf5e}});
  TestVlsegXeXX<UInt16, 4, 1>(0x6000d407,  // vlseg4e16.v v8, (x1), v0.t
                              {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                               {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                               {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c},
                               {0x8706, 0x8f0e, 0x9716, 0x9f1e, 0xa726, 0xaf2e, 0xb736, 0xbf3e}});
  TestVlsegXeXX<UInt16, 4, 2>(0x6000d407,  // vlseg4e16.v v8, (x1), v0.t
                              {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                               {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                               {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                               {0xc342, 0xcb4a, 0xd352, 0xdb5a, 0xe362, 0xeb6a, 0xf372, 0xfb7a},
                               {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c},
                               {0xc544, 0xcd4c, 0xd554, 0xdd5c, 0xe564, 0xed6c, 0xf574, 0xfd7c},
                               {0x8706, 0x8f0e, 0x9716, 0x9f1e, 0xa726, 0xaf2e, 0xb736, 0xbf3e},
                               {0xc746, 0xcf4e, 0xd756, 0xdf5e, 0xe766, 0xef6e, 0xf776, 0xff7e}});
  TestVlsegXeXX<UInt16, 5, 1>(0x8000d407,  // vlseg5e16.v v8, (x1), v0.t
                              {{0x8100, 0x8b0a, 0x9514, 0x9f1e, 0xa928, 0xb332, 0xbd3c, 0xc746},
                               {0x8302, 0x8d0c, 0x9716, 0xa120, 0xab2a, 0xb534, 0xbf3e, 0xc948},
                               {0x8504, 0x8f0e, 0x9918, 0xa322, 0xad2c, 0xb736, 0xc140, 0xcb4a},
                               {0x8706, 0x9110, 0x9b1a, 0xa524, 0xaf2e, 0xb938, 0xc342, 0xcd4c},
                               {0x8908, 0x9312, 0x9d1c, 0xa726, 0xb130, 0xbb3a, 0xc544, 0xcf4e}});
  TestVlsegXeXX<UInt16, 6, 1>(0xa000d407,  // vlseg6e16.v v8, (x1), v0.t
                              {{0x8100, 0x8d0c, 0x9918, 0xa524, 0xb130, 0xbd3c, 0xc948, 0xd554},
                               {0x8302, 0x8f0e, 0x9b1a, 0xa726, 0xb332, 0xbf3e, 0xcb4a, 0xd756},
                               {0x8504, 0x9110, 0x9d1c, 0xa928, 0xb534, 0xc140, 0xcd4c, 0xd958},
                               {0x8706, 0x9312, 0x9f1e, 0xab2a, 0xb736, 0xc342, 0xcf4e, 0xdb5a},
                               {0x8908, 0x9514, 0xa120, 0xad2c, 0xb938, 0xc544, 0xd150, 0xdd5c},
                               {0x8b0a, 0x9716, 0xa322, 0xaf2e, 0xbb3a, 0xc746, 0xd352, 0xdf5e}});
  TestVlsegXeXX<UInt16, 7, 1>(0xc000d407,  // vlseg7e16.v v8, (x1), v0.t
                              {{0x8100, 0x8f0e, 0x9d1c, 0xab2a, 0xb938, 0xc746, 0xd554, 0xe362},
                               {0x8302, 0x9110, 0x9f1e, 0xad2c, 0xbb3a, 0xc948, 0xd756, 0xe564},
                               {0x8504, 0x9312, 0xa120, 0xaf2e, 0xbd3c, 0xcb4a, 0xd958, 0xe766},
                               {0x8706, 0x9514, 0xa322, 0xb130, 0xbf3e, 0xcd4c, 0xdb5a, 0xe968},
                               {0x8908, 0x9716, 0xa524, 0xb332, 0xc140, 0xcf4e, 0xdd5c, 0xeb6a},
                               {0x8b0a, 0x9918, 0xa726, 0xb534, 0xc342, 0xd150, 0xdf5e, 0xed6c},
                               {0x8d0c, 0x9b1a, 0xa928, 0xb736, 0xc544, 0xd352, 0xe160, 0xef6e}});
  TestVlsegXeXX<UInt16, 8, 1>(0xe000d407,  // vlseg8e16.v v8, (x1), v0.t
                              {{0x8100, 0x9110, 0xa120, 0xb130, 0xc140, 0xd150, 0xe160, 0xf170},
                               {0x8302, 0x9312, 0xa322, 0xb332, 0xc342, 0xd352, 0xe362, 0xf372},
                               {0x8504, 0x9514, 0xa524, 0xb534, 0xc544, 0xd554, 0xe564, 0xf574},
                               {0x8706, 0x9716, 0xa726, 0xb736, 0xc746, 0xd756, 0xe766, 0xf776},
                               {0x8908, 0x9918, 0xa928, 0xb938, 0xc948, 0xd958, 0xe968, 0xf978},
                               {0x8b0a, 0x9b1a, 0xab2a, 0xbb3a, 0xcb4a, 0xdb5a, 0xeb6a, 0xfb7a},
                               {0x8d0c, 0x9d1c, 0xad2c, 0xbd3c, 0xcd4c, 0xdd5c, 0xed6c, 0xfd7c},
                               {0x8f0e, 0x9f1e, 0xaf2e, 0xbf3e, 0xcf4e, 0xdf5e, 0xef6e, 0xff7e}});
  TestVlsegXeXX<UInt32, 1, 1>(0x000e407,  // vle32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c}});
  TestVlsegXeXX<UInt32, 1, 2>(0x000e407,  // vle32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
                               {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c}});
  TestVlsegXeXX<UInt32, 1, 4>(0x000e407,  // vle32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
                               {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
                               {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
                               {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c}});
  TestVlsegXeXX<UInt32, 1, 8>(0x000e407,  // vle32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
                               {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
                               {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
                               {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
                               {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                               {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                               {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                               {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}});
  TestVlsegXeXX<UInt32, 2, 1>(0x2000e407,  // vlseg2e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8b0a'8908, 0x9312'9110, 0x9b1a'9918},
                               {0x8706'8504, 0x8f0e'8d0c, 0x9716'9514, 0x9f1e'9d1c}});
  TestVlsegXeXX<UInt32, 2, 2>(0x2000e407,  // vlseg2e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8b0a'8908, 0x9312'9110, 0x9b1a'9918},
                               {0xa322'a120, 0xab2a'a928, 0xb332'b130, 0xbb3a'b938},
                               {0x8706'8504, 0x8f0e'8d0c, 0x9716'9514, 0x9f1e'9d1c},
                               {0xa726'a524, 0xaf2e'ad2c, 0xb736'b534, 0xbf3e'bd3c}});
  TestVlsegXeXX<UInt32, 2, 4>(0x2000e407,  // vlseg2e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8b0a'8908, 0x9312'9110, 0x9b1a'9918},
                               {0xa322'a120, 0xab2a'a928, 0xb332'b130, 0xbb3a'b938},
                               {0xc342'c140, 0xcb4a'c948, 0xd352'd150, 0xdb5a'd958},
                               {0xe362'e160, 0xeb6a'e968, 0xf372'f170, 0xfb7a'f978},
                               {0x8706'8504, 0x8f0e'8d0c, 0x9716'9514, 0x9f1e'9d1c},
                               {0xa726'a524, 0xaf2e'ad2c, 0xb736'b534, 0xbf3e'bd3c},
                               {0xc746'c544, 0xcf4e'cd4c, 0xd756'd554, 0xdf5e'dd5c},
                               {0xe766'e564, 0xef6e'ed6c, 0xf776'f574, 0xff7e'fd7c}});
  TestVlsegXeXX<UInt32, 3, 1>(0x4000e407,  // vlseg3e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8f0e'8d0c, 0x9b1a'9918, 0xa726'a524},
                               {0x8706'8504, 0x9312'9110, 0x9f1e'9d1c, 0xab2a'a928},
                               {0x8b0a'8908, 0x9716'9514, 0xa322'a120, 0xaf2e'ad2c}});
  TestVlsegXeXX<UInt32, 3, 2>(0x4000e407,  // vlseg3e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x8f0e'8d0c, 0x9b1a'9918, 0xa726'a524},
                               {0xb332'b130, 0xbf3e'bd3c, 0xcb4a'c948, 0xd756'd554},
                               {0x8706'8504, 0x9312'9110, 0x9f1e'9d1c, 0xab2a'a928},
                               {0xb736'b534, 0xc342'c140, 0xcf4e'cd4c, 0xdb5a'd958},
                               {0x8b0a'8908, 0x9716'9514, 0xa322'a120, 0xaf2e'ad2c},
                               {0xbb3a'b938, 0xc746'c544, 0xd352'd150, 0xdf5e'dd5c}});
  TestVlsegXeXX<UInt32, 4, 1>(0x6000e407,  // vlseg4e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                               {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                               {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938},
                               {0x8f0e'8d0c, 0x9f1e'9d1c, 0xaf2e'ad2c, 0xbf3e'bd3c}});
  TestVlsegXeXX<UInt32, 4, 2>(0x6000e407,  // vlseg4e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                               {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                               {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                               {0xc746'c544, 0xd756'd554, 0xe766'e564, 0xf776'f574},
                               {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938},
                               {0xcb4a'c948, 0xdb5a'd958, 0xeb6a'e968, 0xfb7a'f978},
                               {0x8f0e'8d0c, 0x9f1e'9d1c, 0xaf2e'ad2c, 0xbf3e'bd3c},
                               {0xcf4e'cd4c, 0xdf5e'dd5c, 0xef6e'ed6c, 0xff7e'fd7c}});
  TestVlsegXeXX<UInt32, 5, 1>(0x8000e407,  // vlseg5e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x9716'9514, 0xab2a'a928, 0xbf3e'bd3c},
                               {0x8706'8504, 0x9b1a'9918, 0xaf2e'ad2c, 0xc342'c140},
                               {0x8b0a'8908, 0x9f1e'9d1c, 0xb332'b130, 0xc746'c544},
                               {0x8f0e'8d0c, 0xa322'a120, 0xb736'b534, 0xcb4a'c948},
                               {0x9312'9110, 0xa726'a524, 0xbb3a'b938, 0xcf4e'cd4c}});
  TestVlsegXeXX<UInt32, 6, 1>(0xa000e407,  // vlseg6e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x9b1a'9918, 0xb332'b130, 0xcb4a'c948},
                               {0x8706'8504, 0x9f1e'9d1c, 0xb736'b534, 0xcf4e'cd4c},
                               {0x8b0a'8908, 0xa322'a120, 0xbb3a'b938, 0xd352'd150},
                               {0x8f0e'8d0c, 0xa726'a524, 0xbf3e'bd3c, 0xd756'd554},
                               {0x9312'9110, 0xab2a'a928, 0xc342'c140, 0xdb5a'd958},
                               {0x9716'9514, 0xaf2e'ad2c, 0xc746'c544, 0xdf5e'dd5c}});
  TestVlsegXeXX<UInt32, 7, 1>(0xc000e407,  // vlseg7e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0x9f1e'9d1c, 0xbb3a'b938, 0xd756'd554},
                               {0x8706'8504, 0xa322'a120, 0xbf3e'bd3c, 0xdb5a'd958},
                               {0x8b0a'8908, 0xa726'a524, 0xc342'c140, 0xdf5e'dd5c},
                               {0x8f0e'8d0c, 0xab2a'a928, 0xc746'c544, 0xe362'e160},
                               {0x9312'9110, 0xaf2e'ad2c, 0xcb4a'c948, 0xe766'e564},
                               {0x9716'9514, 0xb332'b130, 0xcf4e'cd4c, 0xeb6a'e968},
                               {0x9b1a'9918, 0xb736'b534, 0xd352'd150, 0xef6e'ed6c}});
  TestVlsegXeXX<UInt32, 8, 1>(0xe000e407,  // vlseg8e32.v v8, (x1), v0.t
                              {{0x8302'8100, 0xa322'a120, 0xc342'c140, 0xe362'e160},
                               {0x8706'8504, 0xa726'a524, 0xc746'c544, 0xe766'e564},
                               {0x8b0a'8908, 0xab2a'a928, 0xcb4a'c948, 0xeb6a'e968},
                               {0x8f0e'8d0c, 0xaf2e'ad2c, 0xcf4e'cd4c, 0xef6e'ed6c},
                               {0x9312'9110, 0xb332'b130, 0xd352'd150, 0xf372'f170},
                               {0x9716'9514, 0xb736'b534, 0xd756'd554, 0xf776'f574},
                               {0x9b1a'9918, 0xbb3a'b938, 0xdb5a'd958, 0xfb7a'f978},
                               {0x9f1e'9d1c, 0xbf3e'bd3c, 0xdf5e'dd5c, 0xff7e'fd7c}});
  TestVlsegXeXX<UInt64, 1, 1>(0x000f407,  // vle64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908}});
  TestVlsegXeXX<UInt64, 1, 2>(0x000f407,  // vle64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
                               {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918}});
  TestVlsegXeXX<UInt64, 1, 4>(0x000f407,  // vle64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
                               {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
                               {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
                               {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938}});
  TestVlsegXeXX<UInt64, 1, 8>(0x000f407,  // vle64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
                               {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
                               {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
                               {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
                               {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                               {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                               {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                               {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}});
  TestVlsegXeXX<UInt64, 2, 1>(0x2000f407,  // vlseg2e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x9716'9514'9312'9110},
                               {0x8f0e'8d0c'8b0a'8908, 0x9f1e'9d1c'9b1a'9918}});
  TestVlsegXeXX<UInt64, 2, 2>(0x2000f407,  // vlseg2e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x9716'9514'9312'9110},
                               {0xa726'a524'a322'a120, 0xb736'b534'b332'b130},
                               {0x8f0e'8d0c'8b0a'8908, 0x9f1e'9d1c'9b1a'9918},
                               {0xaf2e'ad2c'ab2a'a928, 0xbf3e'bd3c'bb3a'b938}});
  TestVlsegXeXX<UInt64, 2, 4>(0x2000f407,  // vlseg2e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x9716'9514'9312'9110},
                               {0xa726'a524'a322'a120, 0xb736'b534'b332'b130},
                               {0xc746'c544'c342'c140, 0xd756'd554'd352'd150},
                               {0xe766'e564'e362'e160, 0xf776'f574'f372'f170},
                               {0x8f0e'8d0c'8b0a'8908, 0x9f1e'9d1c'9b1a'9918},
                               {0xaf2e'ad2c'ab2a'a928, 0xbf3e'bd3c'bb3a'b938},
                               {0xcf4e'cd4c'cb4a'c948, 0xdf5e'dd5c'db5a'd958},
                               {0xef6e'ed6c'eb6a'e968, 0xff7e'fd7c'fb7a'f978}});
  TestVlsegXeXX<UInt64, 3, 1>(0x4000f407,  // vlseg3e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x9f1e'9d1c'9b1a'9918},
                               {0x8f0e'8d0c'8b0a'8908, 0xa726'a524'a322'a120},
                               {0x9716'9514'9312'9110, 0xaf2e'ad2c'ab2a'a928}});
  TestVlsegXeXX<UInt64, 3, 2>(0x4000f407,  // vlseg3e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0x9f1e'9d1c'9b1a'9918},
                               {0xb736'b534'b332'b130, 0xcf4e'cd4c'cb4a'c948},
                               {0x8f0e'8d0c'8b0a'8908, 0xa726'a524'a322'a120},
                               {0xbf3e'bd3c'bb3a'b938, 0xd756'd554'd352'd150},
                               {0x9716'9514'9312'9110, 0xaf2e'ad2c'ab2a'a928},
                               {0xc746'c544'c342'c140, 0xdf5e'dd5c'db5a'd958}});
  TestVlsegXeXX<UInt64, 4, 1>(0x6000f407,  // vlseg4e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                               {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                               {0x9716'9514'9312'9110, 0xb736'b534'b332'b130},
                               {0x9f1e'9d1c'9b1a'9918, 0xbf3e'bd3c'bb3a'b938}});
  TestVlsegXeXX<UInt64, 4, 2>(0x6000f407,  // vlseg4e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                               {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                               {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                               {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968},
                               {0x9716'9514'9312'9110, 0xb736'b534'b332'b130},
                               {0xd756'd554'd352'd150, 0xf776'f574'f372'f170},
                               {0x9f1e'9d1c'9b1a'9918, 0xbf3e'bd3c'bb3a'b938},
                               {0xdf5e'dd5c'db5a'd958, 0xff7e'fd7c'fb7a'f978}});
  TestVlsegXeXX<UInt64, 5, 1>(0x8000f407,  // vlseg5e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xaf2e'ad2c'ab2a'a928},
                               {0x8f0e'8d0c'8b0a'8908, 0xb736'b534'b332'b130},
                               {0x9716'9514'9312'9110, 0xbf3e'bd3c'bb3a'b938},
                               {0x9f1e'9d1c'9b1a'9918, 0xc746'c544'c342'c140},
                               {0xa726'a524'a322'a120, 0xcf4e'cd4c'cb4a'c948}});
  TestVlsegXeXX<UInt64, 6, 1>(0xa000f407,  // vlseg6e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xb736'b534'b332'b130},
                               {0x8f0e'8d0c'8b0a'8908, 0xbf3e'bd3c'bb3a'b938},
                               {0x9716'9514'9312'9110, 0xc746'c544'c342'c140},
                               {0x9f1e'9d1c'9b1a'9918, 0xcf4e'cd4c'cb4a'c948},
                               {0xa726'a524'a322'a120, 0xd756'd554'd352'd150},
                               {0xaf2e'ad2c'ab2a'a928, 0xdf5e'dd5c'db5a'd958}});
  TestVlsegXeXX<UInt64, 7, 1>(0xc000f407,  // vlseg7e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xbf3e'bd3c'bb3a'b938},
                               {0x8f0e'8d0c'8b0a'8908, 0xc746'c544'c342'c140},
                               {0x9716'9514'9312'9110, 0xcf4e'cd4c'cb4a'c948},
                               {0x9f1e'9d1c'9b1a'9918, 0xd756'd554'd352'd150},
                               {0xa726'a524'a322'a120, 0xdf5e'dd5c'db5a'd958},
                               {0xaf2e'ad2c'ab2a'a928, 0xe766'e564'e362'e160},
                               {0xb736'b534'b332'b130, 0xef6e'ed6c'eb6a'e968}});
  TestVlsegXeXX<UInt64, 8, 1>(0xe000f407,  // vlseg8e64.v v8, (x1), v0.t
                              {{0x8706'8504'8302'8100, 0xc746'c544'c342'c140},
                               {0x8f0e'8d0c'8b0a'8908, 0xcf4e'cd4c'cb4a'c948},
                               {0x9716'9514'9312'9110, 0xd756'd554'd352'd150},
                               {0x9f1e'9d1c'9b1a'9918, 0xdf5e'dd5c'db5a'd958},
                               {0xa726'a524'a322'a120, 0xe766'e564'e362'e160},
                               {0xaf2e'ad2c'ab2a'a928, 0xef6e'ed6c'eb6a'e968},
                               {0xb736'b534'b332'b130, 0xf776'f574'f372'f170},
                               {0xbf3e'bd3c'bb3a'b938, 0xff7e'fd7c'fb7a'f978}});
}

TEST_F(Riscv64InterpreterTest, TestVlssegXeXX) {
  TestVlssegXeXX<UInt8, 1, 1>(0x08208407,  // vlse8.v v8, (x1), x2, v0.t
                              4,
                              {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60}});
  TestVlssegXeXX<UInt8, 1, 2>(
      0x08208407,  // vlse8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124}});
  TestVlssegXeXX<UInt8, 1, 4>(
      0x08208407,  // vlse8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {0, 9, 17, 24, 32, 41, 49, 56, 64, 73, 81, 88, 96, 105, 113, 120},
       {128, 137, 145, 152, 160, 169, 177, 184, 192, 201, 209, 216, 224, 233, 241, 248}});
  TestVlssegXeXX<UInt8, 1, 8>(
      0x08208407,  // vlse8.v v8, (x1), x2, v0.t
      2,
      {{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
       {32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62},
       {64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94},
       {96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126},
       {0, 4, 9, 12, 17, 20, 24, 28, 32, 36, 41, 44, 49, 52, 56, 60},
       {64, 68, 73, 76, 81, 84, 88, 92, 96, 100, 105, 108, 113, 116, 120, 124},
       {128, 132, 137, 140, 145, 148, 152, 156, 160, 164, 169, 172, 177, 180, 184, 188},
       {192, 196, 201, 204, 209, 212, 216, 220, 224, 228, 233, 236, 241, 244, 248, 252}});
  TestVlssegXeXX<UInt8, 2, 1>(
      0x28208407,  // vlsseg2e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189}});
  TestVlssegXeXX<UInt8, 2, 2>(
      0x28208407,  // vlsseg2e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253}});
  TestVlssegXeXX<UInt8, 2, 4>(
      0x28208407,  // vlsseg2e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {0, 9, 17, 24, 32, 41, 49, 56, 64, 73, 81, 88, 96, 105, 113, 120},
       {128, 137, 145, 152, 160, 169, 177, 184, 192, 201, 209, 216, 224, 233, 241, 248},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253},
       {146, 154, 130, 138, 178, 186, 162, 170, 210, 218, 194, 202, 242, 250, 226, 234},
       {18, 26, 2, 10, 50, 58, 34, 42, 82, 90, 66, 74, 114, 122, 98, 106}});
  TestVlssegXeXX<UInt8, 3, 1>(
      0x48208407,  // vlsseg3e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62}});
  TestVlssegXeXX<UInt8, 3, 2>(
      0x48208407,  // vlsseg3e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
       {66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126}});
  TestVlssegXeXX<UInt8, 4, 1>(
      0x68208407,  // vlsseg4e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
       {131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191}});
  TestVlssegXeXX<UInt8, 4, 2>(
      0x68208407,  // vlsseg4e8.v v8, (x1), x2, v0.t
      4,
      {{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
       {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124},
       {129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189},
       {193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253},
       {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
       {66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126},
       {131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191},
       {195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255}});
  TestVlssegXeXX<UInt8, 5, 1>(
      0x88208407,  // vlsseg5e8.v v8, (x1), x2, v0.t
      8,
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249},
       {2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122},
       {131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251},
       {4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124}});
  TestVlssegXeXX<UInt8, 6, 1>(
      0xa8208407,  // vlsseg6e8.v v8, (x1), x2, v0.t
      8,
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249},
       {2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122},
       {131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251},
       {4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124},
       {133, 141, 149, 157, 165, 173, 181, 189, 197, 205, 213, 221, 229, 237, 245, 253}});
  TestVlssegXeXX<UInt8, 7, 1>(
      0xc8208407,  // vlsseg7e8.v v8, (x1), x2, v0.t
      8,
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249},
       {2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122},
       {131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251},
       {4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124},
       {133, 141, 149, 157, 165, 173, 181, 189, 197, 205, 213, 221, 229, 237, 245, 253},
       {6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126}});
  TestVlssegXeXX<UInt8, 8, 1>(
      0xe8208407,  // vlsseg8e8.v v8, (x1), x2, v0.t
      8,
      {{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120},
       {129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249},
       {2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122},
       {131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251},
       {4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124},
       {133, 141, 149, 157, 165, 173, 181, 189, 197, 205, 213, 221, 229, 237, 245, 253},
       {6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126},
       {135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239, 247, 255}});
  TestVlssegXeXX<UInt16, 1, 1>(0x820d407,  // vlse16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938}});
  TestVlssegXeXX<UInt16, 1, 2>(0x820d407,  // vlse16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978}});
  TestVlssegXeXX<UInt16, 1, 4>(0x820d407,  // vlse16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                                {0x9200, 0x8211, 0xb220, 0xa231, 0xd240, 0xc251, 0xf260, 0xe271},
                                {0x1280, 0x0291, 0x32a0, 0x22b1, 0x52c0, 0x42d1, 0x72e0, 0x62f1}});
  TestVlssegXeXX<UInt16, 1, 8>(0x820d407,  // vlse16.v v8, (x1), x2, v0.t
                               4,
                               {{0x8100, 0x8504, 0x8908, 0x8d0c, 0x9110, 0x9514, 0x9918, 0x9d1c},
                                {0xa120, 0xa524, 0xa928, 0xad2c, 0xb130, 0xb534, 0xb938, 0xbd3c},
                                {0xc140, 0xc544, 0xc948, 0xcd4c, 0xd150, 0xd554, 0xd958, 0xdd5c},
                                {0xe160, 0xe564, 0xe968, 0xed6c, 0xf170, 0xf574, 0xf978, 0xfd7c},
                                {0x9200, 0x9a09, 0x8211, 0x8a18, 0xb220, 0xba29, 0xa231, 0xaa38},
                                {0xd240, 0xda49, 0xc251, 0xca58, 0xf260, 0xfa69, 0xe271, 0xea78},
                                {0x1280, 0x1a89, 0x0291, 0x0a98, 0x32a0, 0x3aa9, 0x22b1, 0x2ab8},
                                {0x52c0, 0x5ac9, 0x42d1, 0x4ad8, 0x72e0, 0x7ae9, 0x62f1, 0x6af8}});
  TestVlssegXeXX<UInt16, 2, 1>(0x2820d407,  // vlsseg2e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a}});
  TestVlssegXeXX<UInt16, 2, 2>(0x2820d407,  // vlsseg2e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0xc342, 0xcb4a, 0xd352, 0xdb5a, 0xe362, 0xeb6a, 0xf372, 0xfb7a}});
  TestVlssegXeXX<UInt16, 2, 4>(0x2820d407,  // vlsseg2e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                                {0x9200, 0x8211, 0xb220, 0xa231, 0xd240, 0xc251, 0xf260, 0xe271},
                                {0x1280, 0x0291, 0x32a0, 0x22b1, 0x52c0, 0x42d1, 0x72e0, 0x62f1},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0xc342, 0xcb4a, 0xd352, 0xdb5a, 0xe362, 0xeb6a, 0xf372, 0xfb7a},
                                {0x9604, 0x8614, 0xb624, 0xa634, 0xd644, 0xc654, 0xf664, 0xe674},
                                {0x1684, 0x0694, 0x36a4, 0x26b4, 0x56c4, 0x46d4, 0x76e4, 0x66f4}});
  TestVlssegXeXX<UInt16, 3, 1>(0x4820d407,  // vlsseg3e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c}});
  TestVlssegXeXX<UInt16, 3, 2>(0x4820d407,  // vlsseg3e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0xc342, 0xcb4a, 0xd352, 0xdb5a, 0xe362, 0xeb6a, 0xf372, 0xfb7a},
                                {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c},
                                {0xc544, 0xcd4c, 0xd554, 0xdd5c, 0xe564, 0xed6c, 0xf574, 0xfd7c}});
  TestVlssegXeXX<UInt16, 4, 1>(0x6820d407,  // vlsseg4e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c},
                                {0x8706, 0x8f0e, 0x9716, 0x9f1e, 0xa726, 0xaf2e, 0xb736, 0xbf3e}});
  TestVlssegXeXX<UInt16, 4, 2>(0x6820d407,  // vlsseg4e16.v v8, (x1), x2, v0.t
                               8,
                               {{0x8100, 0x8908, 0x9110, 0x9918, 0xa120, 0xa928, 0xb130, 0xb938},
                                {0xc140, 0xc948, 0xd150, 0xd958, 0xe160, 0xe968, 0xf170, 0xf978},
                                {0x8302, 0x8b0a, 0x9312, 0x9b1a, 0xa322, 0xab2a, 0xb332, 0xbb3a},
                                {0xc342, 0xcb4a, 0xd352, 0xdb5a, 0xe362, 0xeb6a, 0xf372, 0xfb7a},
                                {0x8504, 0x8d0c, 0x9514, 0x9d1c, 0xa524, 0xad2c, 0xb534, 0xbd3c},
                                {0xc544, 0xcd4c, 0xd554, 0xdd5c, 0xe564, 0xed6c, 0xf574, 0xfd7c},
                                {0x8706, 0x8f0e, 0x9716, 0x9f1e, 0xa726, 0xaf2e, 0xb736, 0xbf3e},
                                {0xc746, 0xcf4e, 0xd756, 0xdf5e, 0xe766, 0xef6e, 0xf776, 0xff7e}});
  TestVlssegXeXX<UInt16, 5, 1>(0x8820d407,  // vlsseg5e16.v v8, (x1), x2, v0.t
                               16,
                               {{0x8100, 0x9110, 0xa120, 0xb130, 0xc140, 0xd150, 0xe160, 0xf170},
                                {0x8302, 0x9312, 0xa322, 0xb332, 0xc342, 0xd352, 0xe362, 0xf372},
                                {0x8504, 0x9514, 0xa524, 0xb534, 0xc544, 0xd554, 0xe564, 0xf574},
                                {0x8706, 0x9716, 0xa726, 0xb736, 0xc746, 0xd756, 0xe766, 0xf776},
                                {0x8908, 0x9918, 0xa928, 0xb938, 0xc948, 0xd958, 0xe968, 0xf978}});
  TestVlssegXeXX<UInt16, 6, 1>(0xa820d407,  // vlsseg6e16.v v8, (x1), x2, v0.t
                               16,
                               {{0x8100, 0x9110, 0xa120, 0xb130, 0xc140, 0xd150, 0xe160, 0xf170},
                                {0x8302, 0x9312, 0xa322, 0xb332, 0xc342, 0xd352, 0xe362, 0xf372},
                                {0x8504, 0x9514, 0xa524, 0xb534, 0xc544, 0xd554, 0xe564, 0xf574},
                                {0x8706, 0x9716, 0xa726, 0xb736, 0xc746, 0xd756, 0xe766, 0xf776},
                                {0x8908, 0x9918, 0xa928, 0xb938, 0xc948, 0xd958, 0xe968, 0xf978},
                                {0x8b0a, 0x9b1a, 0xab2a, 0xbb3a, 0xcb4a, 0xdb5a, 0xeb6a, 0xfb7a}});
  TestVlssegXeXX<UInt16, 7, 1>(0xc820d407,  // vlsseg7e16.v v8, (x1), x2, v0.t
                               16,
                               {{0x8100, 0x9110, 0xa120, 0xb130, 0xc140, 0xd150, 0xe160, 0xf170},
                                {0x8302, 0x9312, 0xa322, 0xb332, 0xc342, 0xd352, 0xe362, 0xf372},
                                {0x8504, 0x9514, 0xa524, 0xb534, 0xc544, 0xd554, 0xe564, 0xf574},
                                {0x8706, 0x9716, 0xa726, 0xb736, 0xc746, 0xd756, 0xe766, 0xf776},
                                {0x8908, 0x9918, 0xa928, 0xb938, 0xc948, 0xd958, 0xe968, 0xf978},
                                {0x8b0a, 0x9b1a, 0xab2a, 0xbb3a, 0xcb4a, 0xdb5a, 0xeb6a, 0xfb7a},
                                {0x8d0c, 0x9d1c, 0xad2c, 0xbd3c, 0xcd4c, 0xdd5c, 0xed6c, 0xfd7c}});
  TestVlssegXeXX<UInt16, 8, 1>(0xe820d407,  // vlsseg8e16.v v8, (x1), x2, v0.t
                               16,
                               {{0x8100, 0x9110, 0xa120, 0xb130, 0xc140, 0xd150, 0xe160, 0xf170},
                                {0x8302, 0x9312, 0xa322, 0xb332, 0xc342, 0xd352, 0xe362, 0xf372},
                                {0x8504, 0x9514, 0xa524, 0xb534, 0xc544, 0xd554, 0xe564, 0xf574},
                                {0x8706, 0x9716, 0xa726, 0xb736, 0xc746, 0xd756, 0xe766, 0xf776},
                                {0x8908, 0x9918, 0xa928, 0xb938, 0xc948, 0xd958, 0xe968, 0xf978},
                                {0x8b0a, 0x9b1a, 0xab2a, 0xbb3a, 0xcb4a, 0xdb5a, 0xeb6a, 0xfb7a},
                                {0x8d0c, 0x9d1c, 0xad2c, 0xbd3c, 0xcd4c, 0xdd5c, 0xed6c, 0xfd7c},
                                {0x8f0e, 0x9f1e, 0xaf2e, 0xbf3e, 0xcf4e, 0xdf5e, 0xef6e, 0xff7e}});
  TestVlssegXeXX<UInt32, 1, 1>(0x820e407,  // vlse32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130}});
  TestVlssegXeXX<UInt32, 1, 2>(0x820e407,  // vlse32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170}});
  TestVlssegXeXX<UInt32, 1, 4>(0x820e407,  // vlse32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                                {0x9604'9200, 0xb624'b220, 0xd644'd240, 0xf664'f260},
                                {0x1684'1280, 0x36a4'32a0, 0x56c4'52c0, 0x76e4'72e0}});
  TestVlssegXeXX<UInt32, 1, 8>(0x820e407,  // vlse32.v v8, (x1), x2, v0.t
                               8,
                               {{0x8302'8100, 0x8b0a'8908, 0x9312'9110, 0x9b1a'9918},
                                {0xa322'a120, 0xab2a'a928, 0xb332'b130, 0xbb3a'b938},
                                {0xc342'c140, 0xcb4a'c948, 0xd352'd150, 0xdb5a'd958},
                                {0xe362'e160, 0xeb6a'e968, 0xf372'f170, 0xfb7a'f978},
                                {0x9604'9200, 0x8614'8211, 0xb624'b220, 0xa634'a231},
                                {0xd644'd240, 0xc654'c251, 0xf664'f260, 0xe674'e271},
                                {0x1684'1280, 0x0694'0291, 0x36a4'32a0, 0x26b4'22b1},
                                {0x56c4'52c0, 0x46d4'42d1, 0x76e4'72e0, 0x66f4'62f1}});
  TestVlssegXeXX<UInt32, 2, 1>(0x2820e407,  // vlsseg2e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534}});
  TestVlssegXeXX<UInt32, 2, 2>(0x2820e407,  // vlsseg2e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0xc746'c544, 0xd756'd554, 0xe766'e564, 0xf776'f574}});
  TestVlssegXeXX<UInt32, 2, 4>(0x2820e407,  // vlsseg2e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                                {0x9604'9200, 0xb624'b220, 0xd644'd240, 0xf664'f260},
                                {0x1684'1280, 0x36a4'32a0, 0x56c4'52c0, 0x76e4'72e0},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0xc746'c544, 0xd756'd554, 0xe766'e564, 0xf776'f574},
                                {0x9e0c'9a09, 0xbe2c'ba29, 0xde4c'da49, 0xfe6c'fa69},
                                {0x1e8c'1a89, 0x3eac'3aa9, 0x5ecc'5ac9, 0x7eec'7ae9}});
  TestVlssegXeXX<UInt32, 3, 1>(0x4820e407,  // vlsseg3e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938}});
  TestVlssegXeXX<UInt32, 3, 2>(0x4820e407,  // vlsseg3e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0xc746'c544, 0xd756'd554, 0xe766'e564, 0xf776'f574},
                                {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938},
                                {0xcb4a'c948, 0xdb5a'd958, 0xeb6a'e968, 0xfb7a'f978}});
  TestVlssegXeXX<UInt32, 4, 1>(0x6820e407,  // vlsseg4e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938},
                                {0x8f0e'8d0c, 0x9f1e'9d1c, 0xaf2e'ad2c, 0xbf3e'bd3c}});
  TestVlssegXeXX<UInt32, 4, 2>(0x6820e407,  // vlsseg4e32.v v8, (x1), x2, v0.t
                               16,
                               {{0x8302'8100, 0x9312'9110, 0xa322'a120, 0xb332'b130},
                                {0xc342'c140, 0xd352'd150, 0xe362'e160, 0xf372'f170},
                                {0x8706'8504, 0x9716'9514, 0xa726'a524, 0xb736'b534},
                                {0xc746'c544, 0xd756'd554, 0xe766'e564, 0xf776'f574},
                                {0x8b0a'8908, 0x9b1a'9918, 0xab2a'a928, 0xbb3a'b938},
                                {0xcb4a'c948, 0xdb5a'd958, 0xeb6a'e968, 0xfb7a'f978},
                                {0x8f0e'8d0c, 0x9f1e'9d1c, 0xaf2e'ad2c, 0xbf3e'bd3c},
                                {0xcf4e'cd4c, 0xdf5e'dd5c, 0xef6e'ed6c, 0xff7e'fd7c}});
  TestVlssegXeXX<UInt32, 5, 1>(0x8820e407,  // vlsseg5e32.v v8, (x1), x2, v0.t
                               32,
                               {{0x8302'8100, 0xa322'a120, 0xc342'c140, 0xe362'e160},
                                {0x8706'8504, 0xa726'a524, 0xc746'c544, 0xe766'e564},
                                {0x8b0a'8908, 0xab2a'a928, 0xcb4a'c948, 0xeb6a'e968},
                                {0x8f0e'8d0c, 0xaf2e'ad2c, 0xcf4e'cd4c, 0xef6e'ed6c},
                                {0x9312'9110, 0xb332'b130, 0xd352'd150, 0xf372'f170}});
  TestVlssegXeXX<UInt32, 6, 1>(0xa820e407,  // vlsseg6e32.v v8, (x1), x2, v0.t
                               32,
                               {{0x8302'8100, 0xa322'a120, 0xc342'c140, 0xe362'e160},
                                {0x8706'8504, 0xa726'a524, 0xc746'c544, 0xe766'e564},
                                {0x8b0a'8908, 0xab2a'a928, 0xcb4a'c948, 0xeb6a'e968},
                                {0x8f0e'8d0c, 0xaf2e'ad2c, 0xcf4e'cd4c, 0xef6e'ed6c},
                                {0x9312'9110, 0xb332'b130, 0xd352'd150, 0xf372'f170},
                                {0x9716'9514, 0xb736'b534, 0xd756'd554, 0xf776'f574}});
  TestVlssegXeXX<UInt32, 7, 1>(0xc820e407,  // vlsseg7e32.v v8, (x1), x2, v0.t
                               32,
                               {{0x8302'8100, 0xa322'a120, 0xc342'c140, 0xe362'e160},
                                {0x8706'8504, 0xa726'a524, 0xc746'c544, 0xe766'e564},
                                {0x8b0a'8908, 0xab2a'a928, 0xcb4a'c948, 0xeb6a'e968},
                                {0x8f0e'8d0c, 0xaf2e'ad2c, 0xcf4e'cd4c, 0xef6e'ed6c},
                                {0x9312'9110, 0xb332'b130, 0xd352'd150, 0xf372'f170},
                                {0x9716'9514, 0xb736'b534, 0xd756'd554, 0xf776'f574},
                                {0x9b1a'9918, 0xbb3a'b938, 0xdb5a'd958, 0xfb7a'f978}});
  TestVlssegXeXX<UInt32, 8, 1>(0xe820e407,  // vlsseg8e32.v v8, (x1), x2, v0.t
                               32,
                               {{0x8302'8100, 0xa322'a120, 0xc342'c140, 0xe362'e160},
                                {0x8706'8504, 0xa726'a524, 0xc746'c544, 0xe766'e564},
                                {0x8b0a'8908, 0xab2a'a928, 0xcb4a'c948, 0xeb6a'e968},
                                {0x8f0e'8d0c, 0xaf2e'ad2c, 0xcf4e'cd4c, 0xef6e'ed6c},
                                {0x9312'9110, 0xb332'b130, 0xd352'd150, 0xf372'f170},
                                {0x9716'9514, 0xb736'b534, 0xd756'd554, 0xf776'f574},
                                {0x9b1a'9918, 0xbb3a'b938, 0xdb5a'd958, 0xfb7a'f978},
                                {0x9f1e'9d1c, 0xbf3e'bd3c, 0xdf5e'dd5c, 0xff7e'fd7c}});
  TestVlssegXeXX<UInt64, 1, 1>(0x820f407,  // vlse64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120}});
  TestVlssegXeXX<UInt64, 1, 2>(0x820f407,  // vlse64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160}});
  TestVlssegXeXX<UInt64, 1, 4>(0x820f407,  // vlse64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                                {0x9e0c'9a09'9604'9200, 0xde4c'da49'd644'd240},
                                {0x1e8c'1a89'1684'1280, 0x5ecc'5ac9'56c4'52c0}});
  TestVlssegXeXX<UInt64, 1, 8>(0x820f407,  // vlse64.v v8, (x1), x2, v0.t
                               16,
                               {{0x8706'8504'8302'8100, 0x9716'9514'9312'9110},
                                {0xa726'a524'a322'a120, 0xb736'b534'b332'b130},
                                {0xc746'c544'c342'c140, 0xd756'd554'd352'd150},
                                {0xe766'e564'e362'e160, 0xf776'f574'f372'f170},
                                {0x9e0c'9a09'9604'9200, 0xbe2c'ba29'b624'b220},
                                {0xde4c'da49'd644'd240, 0xfe6c'fa69'f664'f260},
                                {0x1e8c'1a89'1684'1280, 0x3eac'3aa9'36a4'32a0},
                                {0x5ecc'5ac9'56c4'52c0, 0x7eec'7ae9'76e4'72e0}});
  TestVlssegXeXX<UInt64, 2, 1>(0x2820f407,  // vlsseg2e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928}});
  TestVlssegXeXX<UInt64, 2, 2>(0x2820f407,  // vlsseg2e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968}});
  TestVlssegXeXX<UInt64, 2, 4>(0x2820f407,  // vlsseg2e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                                {0x9e0c'9a09'9604'9200, 0xde4c'da49'd644'd240},
                                {0x1e8c'1a89'1684'1280, 0x5ecc'5ac9'56c4'52c0},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968},
                                {0x8e1c'8a18'8614'8211, 0xce5c'ca58'c654'c251},
                                {0x0e9c'0a98'0694'0291, 0x4edc'4ad8'46d4'42d1}});
  TestVlssegXeXX<UInt64, 3, 1>(0x4820f407,  // vlsseg3e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0x9716'9514'9312'9110, 0xb736'b534'b332'b130}});
  TestVlssegXeXX<UInt64, 3, 2>(0x4820f407,  // vlsseg3e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968},
                                {0x9716'9514'9312'9110, 0xb736'b534'b332'b130},
                                {0xd756'd554'd352'd150, 0xf776'f574'f372'f170}});
  TestVlssegXeXX<UInt64, 4, 1>(0x6820f407,  // vlsseg4e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0x9716'9514'9312'9110, 0xb736'b534'b332'b130},
                                {0x9f1e'9d1c'9b1a'9918, 0xbf3e'bd3c'bb3a'b938}});
  TestVlssegXeXX<UInt64, 4, 2>(0x6820f407,  // vlsseg4e64.v v8, (x1), x2, v0.t
                               32,
                               {{0x8706'8504'8302'8100, 0xa726'a524'a322'a120},
                                {0xc746'c544'c342'c140, 0xe766'e564'e362'e160},
                                {0x8f0e'8d0c'8b0a'8908, 0xaf2e'ad2c'ab2a'a928},
                                {0xcf4e'cd4c'cb4a'c948, 0xef6e'ed6c'eb6a'e968},
                                {0x9716'9514'9312'9110, 0xb736'b534'b332'b130},
                                {0xd756'd554'd352'd150, 0xf776'f574'f372'f170},
                                {0x9f1e'9d1c'9b1a'9918, 0xbf3e'bd3c'bb3a'b938},
                                {0xdf5e'dd5c'db5a'd958, 0xff7e'fd7c'fb7a'f978}});
  TestVlssegXeXX<UInt64, 5, 1>(0x8820f407,  // vlsseg5e64.v v8, (x1), x2, v0.t
                               64,
                               {{0x8706'8504'8302'8100, 0xc746'c544'c342'c140},
                                {0x8f0e'8d0c'8b0a'8908, 0xcf4e'cd4c'cb4a'c948},
                                {0x9716'9514'9312'9110, 0xd756'd554'd352'd150},
                                {0x9f1e'9d1c'9b1a'9918, 0xdf5e'dd5c'db5a'd958},
                                {0xa726'a524'a322'a120, 0xe766'e564'e362'e160}});
  TestVlssegXeXX<UInt64, 6, 1>(0xa820f407,  // vlsseg6e64.v v8, (x1), x2, v0.t
                               64,
                               {{0x8706'8504'8302'8100, 0xc746'c544'c342'c140},
                                {0x8f0e'8d0c'8b0a'8908, 0xcf4e'cd4c'cb4a'c948},
                                {0x9716'9514'9312'9110, 0xd756'd554'd352'd150},
                                {0x9f1e'9d1c'9b1a'9918, 0xdf5e'dd5c'db5a'd958},
                                {0xa726'a524'a322'a120, 0xe766'e564'e362'e160},
                                {0xaf2e'ad2c'ab2a'a928, 0xef6e'ed6c'eb6a'e968}});
  TestVlssegXeXX<UInt64, 7, 1>(0xc820f407,  // vlsseg7e64.v v8, (x1), x2, v0.t
                               64,
                               {{0x8706'8504'8302'8100, 0xc746'c544'c342'c140},
                                {0x8f0e'8d0c'8b0a'8908, 0xcf4e'cd4c'cb4a'c948},
                                {0x9716'9514'9312'9110, 0xd756'd554'd352'd150},
                                {0x9f1e'9d1c'9b1a'9918, 0xdf5e'dd5c'db5a'd958},
                                {0xa726'a524'a322'a120, 0xe766'e564'e362'e160},
                                {0xaf2e'ad2c'ab2a'a928, 0xef6e'ed6c'eb6a'e968},
                                {0xb736'b534'b332'b130, 0xf776'f574'f372'f170}});
  TestVlssegXeXX<UInt64, 8, 1>(0xe820f407,  // vlsseg8e64.v v8, (x1), x2, v0.t
                               64,
                               {{0x8706'8504'8302'8100, 0xc746'c544'c342'c140},
                                {0x8f0e'8d0c'8b0a'8908, 0xcf4e'cd4c'cb4a'c948},
                                {0x9716'9514'9312'9110, 0xd756'd554'd352'd150},
                                {0x9f1e'9d1c'9b1a'9918, 0xdf5e'dd5c'db5a'd958},
                                {0xa726'a524'a322'a120, 0xe766'e564'e362'e160},
                                {0xaf2e'ad2c'ab2a'a928, 0xef6e'ed6c'eb6a'e968},
                                {0xb736'b534'b332'b130, 0xf776'f574'f372'f170},
                                {0xbf3e'bd3c'bb3a'b938, 0xff7e'fd7c'fb7a'f978}});
}

TEST_F(Riscv64InterpreterTest, TestVlm) {
  TestVlm(0x2b08407,  // vlm.v v8, (x1)
          {0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143});
}

TEST_F(Riscv64InterpreterTest, VsxsegXeiXX) {
  VsxsegXeiXX<UInt8, 1, 1>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                           {0x0487'8506'0283'0081, 0x0a89'0c8d'8b8f'080e});
  VsxsegXeiXX<UInt8, 1, 2>(
      0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
      {0x0487'8506'0283'0081, 0x0a89'0c8d'8b8f'080e, 0x9f93'9b1e'9714'121a, 0x9110'1899'1c95'169d});
  VsxsegXeiXX<UInt8, 1, 4>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                           {0x0487'8506'0283'0081,
                            0x0a89'0c8d'8b8f'080e,
                            0x9f93'9b1e'9714'121a,
                            0x9110'1899'1c95'169d,
                            0x2ea5'bd2c'30a3'38b9,
                            0xafad'3e20'a728'b1ab,
                            0x3626'b722'b5a1'bbbf,
                            0xa932'2434'b33a'2a3c});
  VsxsegXeiXX<UInt8, 1, 8>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                           {0x0487'8506'0283'0081,
                            0x0a89'0c8d'8b8f'080e,
                            0x9f93'9b1e'9714'121a,
                            0x9110'1899'1c95'169d,
                            0x2ea5'bd2c'30a3'38b9,
                            0xafad'3e20'a728'b1ab,
                            0x3626'b722'b5a1'bbbf,
                            0xa932'2434'b33a'2a3c,
                            0x6aed'ddd1'e35c'66c7,
                            0xd542'72e1'4674'4ed3,
                            0x78ef'd9e7'7a7e'eb76,
                            0xfbf9'f5c1'6e5a'c5ff,
                            0x52cd'c362'6c48'dfd7,
                            0x4a54'50fd'f3f7'cf68,
                            0x56cb'e57c'7044'60c9,
                            0xf1db'6440'58e9'4c5e});
  VsxsegXeiXX<UInt8, 2, 1>(
      0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
      {0x1202'9383'1000'9181, 0x1404'9787'9585'1606, 0x9b8b'9f8f'1808'1e0e, 0x1a0a'9989'1c0c'9d8d});
  VsxsegXeiXX<UInt8, 2, 2>(0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                           {0x2202'a383'2000'a181,
                            0x2404'a787'a585'2606,
                            0xab8b'af8f'2808'2e0e,
                            0x2a0a'a989'2c0c'ad8d,
                            0xb797'3414'3212'3a1a,
                            0xbf9f'b393'bb9b'3e1e,
                            0x3c1c'b595'3616'bd9d,
                            0xb191'3010'3818'b999});
  VsxsegXeiXX<UInt8, 2, 4>(0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                           {0x4202'c383'4000'c181,
                            0x4404'c787'c585'4606,
                            0xcb8b'cf8f'4808'4e0e,
                            0x4a0a'c989'4c0c'cd8d,
                            0xd797'5414'5212'5a1a,
                            0xdf9f'd393'db9b'5e1e,
                            0x5c1c'd595'5616'dd9d,
                            0xd191'5010'5818'd999,
                            0x7030'e3a3'7838'f9b9,
                            0x6e2e'e5a5'fdbd'6c2c,
                            0xe7a7'6828'f1b1'ebab,
                            0xefaf'edad'7e3e'6020,
                            0xf5b5'e1a1'fbbb'ffbf,
                            0x7636'6626'f7b7'6222,
                            0xf3b3'7a3a'6a2a'7c3c,
                            0xe9a9'7232'6424'7434});
  VsxsegXeiXX<UInt8, 3, 1>(0x45008427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                           {0x9383'2010'00a1'9181,
                            0x8526'1606'2212'02a3,
                            0x2414'04a7'9787'a595,
                            0x9f8f'2818'082e'1e0e,
                            0x0cad'9d8d'ab9b'8baf,
                            0x2a1a'0aa9'9989'2c1c});
  VsxsegXeiXX<UInt8, 3, 2>(0x45008427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                           {0xa383'4020'00c1'a181,
                            0x8546'2606'4222'02c3,
                            0x4424'04c7'a787'c5a5,
                            0xaf8f'4828'084e'2e0e,
                            0x0ccd'ad8d'cbab'8bcf,
                            0x4a2a'0ac9'a989'4c2c,
                            0x3414'5232'125a'3a1a,
                            0x9b5e'3e1e'd7b7'9754,
                            0xdfbf'9fd3'b393'dbbb,
                            0xb595'5636'16dd'bd9d,
                            0x18d9'b999'5c3c'1cd5,
                            0xd1b1'9150'3010'5838});
  VsxsegXeiXX<UInt8, 4, 1>(0x65008427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                           {0x3020'1000'b1a1'9181,
                            0x3222'1202'b3a3'9383,
                            0xb5a5'9585'3626'1606,
                            0x3424'1404'b7a7'9787,
                            0x3828'1808'3e2e'1e0e,
                            0xbbab'9b8b'bfaf'9f8f,
                            0x3c2c'1c0c'bdad'9d8d,
                            0x3a2a'1a0a'b9a9'9989});
  VsxsegXeiXX<UInt8, 4, 2>(0x65008427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                           {0x6040'2000'e1c1'a181,
                            0x6242'2202'e3c3'a383,
                            0xe5c5'a585'6646'2606,
                            0x6444'2404'e7c7'a787,
                            0x6848'2808'6e4e'2e0e,
                            0xebcb'ab8b'efcf'af8f,
                            0x6c4c'2c0c'edcd'ad8d,
                            0x6a4a'2a0a'e9c9'a989,
                            0x7252'3212'7a5a'3a1a,
                            0xf7d7'b797'7454'3414,
                            0xfbdb'bb9b'7e5e'3e1e,
                            0xffdf'bf9f'f3d3'b393,
                            0x7656'3616'fddd'bd9d,
                            0x7c5c'3c1c'f5d5'b595,
                            0x7858'3818'f9d9'b999,
                            0xf1d1'b191'7050'3010});
  VsxsegXeiXX<UInt8, 5, 1>(0x85008427,  // Vsuxseg5ei8.v v8, (x1), v16, v0.t
                           {0x2010'00c1'b1a1'9181,
                            0x02c3'b3a3'9383'4030,
                            0x3626'1606'4232'2212,
                            0x9787'c5b5'a595'8546,
                            0x4434'2414'04c7'b7a7,
                            0x2818'084e'3e2e'1e0e,
                            0x8bcf'bfaf'9f8f'4838,
                            0xbdad'9d8d'cbbb'ab9b,
                            0x9989'4c3c'2c1c'0ccd,
                            0x4a3a'2a1a'0ac9'b9a9});
  VsxsegXeiXX<UInt8, 6, 1>(0xa5008427,  // Vsuxseg6ei8.v v8, (x1), v16, v0.t
                           {0x1000'd1c1'b1a1'9181,
                            0xb3a3'9383'5040'3020,
                            0x5242'3222'1202'd3c3,
                            0x9585'5646'3626'1606,
                            0xb7a7'9787'd5c5'b5a5,
                            0x5444'3424'1404'd7c7,
                            0x1808'5e4e'3e2e'1e0e,
                            0xbfaf'9f8f'5848'3828,
                            0xdbcb'bbab'9b8b'dfcf,
                            0x1c0c'ddcd'bdad'9d8d,
                            0xb9a9'9989'5c4c'3c2c,
                            0x5a4a'3a2a'1a0a'd9c9});
  VsxsegXeiXX<UInt8, 7, 1>(0xc5008427,  // Vsuxseg7ei8.v v8, (x1), v16, v0.t
                           {0x00e1'd1c1'b1a1'9181,
                            0x9383'6050'4030'2010,
                            0x2212'02e3'd3c3'b3a3,
                            0x3626'1606'6252'4232,
                            0xc5b5'a595'8566'5646,
                            0xd7c7'b7a7'9787'e5d5,
                            0x6454'4434'2414'04e7,
                            0x086e'5e4e'3e2e'1e0e,
                            0x9f8f'6858'4838'2818,
                            0xab9b'8bef'dfcf'bfaf,
                            0xbdad'9d8d'ebdb'cbbb,
                            0x4c3c'2c1c'0ced'ddcd,
                            0xd9c9'b9a9'9989'6c5c,
                            0x6a5a'4a3a'2a1a'0ae9});
  VsxsegXeiXX<UInt8, 8, 1>(0xe5008427,  // Vsuxseg8ei8.v v8, (x1), v16, v0.t
                           {0xf1e1'd1c1'b1a1'9181,
                            0x7060'5040'3020'1000,
                            0xf3e3'd3c3'b3a3'9383,
                            0x7262'5242'3222'1202,
                            0x7666'5646'3626'1606,
                            0xf5e5'd5c5'b5a5'9585,
                            0xf7e7'd7c7'b7a7'9787,
                            0x7464'5444'3424'1404,
                            0x7e6e'5e4e'3e2e'1e0e,
                            0x7868'5848'3828'1808,
                            0xffef'dfcf'bfaf'9f8f,
                            0xfbeb'dbcb'bbab'9b8b,
                            0xfded'ddcd'bdad'9d8d,
                            0x7c6c'5c4c'3c2c'1c0c,
                            0xf9e9'd9c9'b9a9'9989,
                            0x7a6a'5a4a'3a2a'1a0a});
  VsxsegXeiXX<UInt16, 1, 1>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8504'8706'8100'8302, 0x8908'8f0e'8b0a'8d0c});
  VsxsegXeiXX<UInt16, 1, 2>(
      0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
      {0x8504'8706'8100'8302, 0x8908'8f0e'8b0a'8d0c, 0x9716'9f1e'9110'9d1c, 0x9514'9312'9918'9b1a});
  VsxsegXeiXX<UInt16, 1, 4>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8504'8706'8100'8302,
                             0x8908'8f0e'8b0a'8d0c,
                             0x9716'9f1e'9110'9d1c,
                             0x9514'9312'9918'9b1a,
                             0xaf2e'a928'a524'b534,
                             0xbf3e'a726'b736'bd3c,
                             0xb938'ab2a'ad2c'bb3a,
                             0xa322'a120'b130'b332});
  VsxsegXeiXX<UInt16, 1, 8>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8504'8706'8100'8302,
                             0x8908'8f0e'8b0a'8d0c,
                             0x9716'9f1e'9110'9d1c,
                             0x9514'9312'9918'9b1a,
                             0xaf2e'a928'a524'b534,
                             0xbf3e'a726'b736'bd3c,
                             0xb938'ab2a'ad2c'bb3a,
                             0xa322'a120'b130'b332,
                             0xe160'c746'f170'f372,
                             0xdd5c'cb4a'fb7a'd958,
                             0xcf4e'd150'e362'd756,
                             0xdf5e'db5a'fd7c'c140,
                             0xeb6a'c342'f776'ff7e,
                             0xed6c'cd4c'ef6e'c544,
                             0xe766'f574'd554'f978,
                             0xd352'e564'c948'e968});
  VsxsegXeiXX<UInt16, 2, 1>(
      0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
      {0x9110'8100'9312'8302, 0x9514'8504'9716'8706, 0x9b1a'8b0a'9d1c'8d0c, 0x9918'8908'9f1e'8f0e});
  VsxsegXeiXX<UInt16, 2, 2>(0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0xa120'8100'a322'8302,
                             0xa524'8504'a726'8706,
                             0xab2a'8b0a'ad2c'8d0c,
                             0xa928'8908'af2e'8f0e,
                             0xb130'9110'bd3c'9d1c,
                             0xb736'9716'bf3e'9f1e,
                             0xb938'9918'bb3a'9b1a,
                             0xb534'9514'b332'9312});
  VsxsegXeiXX<UInt16, 2, 4>(0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0xc140'8100'c342'8302,
                             0xc544'8504'c746'8706,
                             0xcb4a'8b0a'cd4c'8d0c,
                             0xc948'8908'cf4e'8f0e,
                             0xd150'9110'dd5c'9d1c,
                             0xd756'9716'df5e'9f1e,
                             0xd958'9918'db5a'9b1a,
                             0xd554'9514'd352'9312,
                             0xe564'a524'f574'b534,
                             0xef6e'af2e'e968'a928,
                             0xf776'b736'fd7c'bd3c,
                             0xff7e'bf3e'e766'a726,
                             0xed6c'ad2c'fb7a'bb3a,
                             0xf978'b938'eb6a'ab2a,
                             0xf170'b130'f372'b332,
                             0xe362'a322'e160'a120});
  VsxsegXeiXX<UInt16, 3, 1>(0x45008427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0x8100'a322'9312'8302,
                             0x9716'8706'a120'9110,
                             0xa524'9514'8504'a726,
                             0x8b0a'ad2c'9d1c'8d0c,
                             0x9f1e'8f0e'ab2a'9b1a,
                             0xa928'9918'8908'af2e});
  VsxsegXeiXX<UInt16, 3, 2>(0x45008427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0x8100'c342'a322'8302,
                             0xa726'8706'c140'a120,
                             0xc544'a524'8504'c746,
                             0x8b0a'cd4c'ad2c'8d0c,
                             0xaf2e'8f0e'cb4a'ab2a,
                             0xc948'a928'8908'cf4e,
                             0x9110'dd5c'bd3c'9d1c,
                             0xbf3e'9f1e'd150'b130,
                             0xd756'b736'9716'df5e,
                             0x9918'db5a'bb3a'9b1a,
                             0xb332'9312'd958'b938,
                             0xd554'b534'9514'd352});
  VsxsegXeiXX<UInt16, 4, 1>(0x65008427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0xb332'a322'9312'8302,
                             0xb130'a120'9110'8100,
                             0xb736'a726'9716'8706,
                             0xb534'a524'9514'8504,
                             0xbd3c'ad2c'9d1c'8d0c,
                             0xbb3a'ab2a'9b1a'8b0a,
                             0xbf3e'af2e'9f1e'8f0e,
                             0xb938'a928'9918'8908});
  VsxsegXeiXX<UInt16, 4, 2>(0x65008427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0xe362'c342'a322'8302,
                             0xe160'c140'a120'8100,
                             0xe766'c746'a726'8706,
                             0xe564'c544'a524'8504,
                             0xed6c'cd4c'ad2c'8d0c,
                             0xeb6a'cb4a'ab2a'8b0a,
                             0xef6e'cf4e'af2e'8f0e,
                             0xe968'c948'a928'8908,
                             0xfd7c'dd5c'bd3c'9d1c,
                             0xf170'd150'b130'9110,
                             0xff7e'df5e'bf3e'9f1e,
                             0xf776'd756'b736'9716,
                             0xfb7a'db5a'bb3a'9b1a,
                             0xf978'd958'b938'9918,
                             0xf372'd352'b332'9312,
                             0xf574'd554'b534'9514});
  VsxsegXeiXX<UInt16, 5, 1>(0x85008427,  // Vsuxseg5ei8.v v8, (x1), v16, v0.t
                            {0xb332'a322'9312'8302,
                             0xa120'9110'8100'c342,
                             0x9716'8706'c140'b130,
                             0x8504'c746'b736'a726,
                             0xc544'b534'a524'9514,
                             0xbd3c'ad2c'9d1c'8d0c,
                             0xab2a'9b1a'8b0a'cd4c,
                             0x9f1e'8f0e'cb4a'bb3a,
                             0x8908'cf4e'bf3e'af2e,
                             0xc948'b938'a928'9918});
  VsxsegXeiXX<UInt16, 6, 1>(0xa5008427,  // Vsuxseg6ei8.v v8, (x1), v16, v0.t
                            {0xb332'a322'9312'8302,
                             0x9110'8100'd352'c342,
                             0xd150'c140'b130'a120,
                             0xb736'a726'9716'8706,
                             0x9514'8504'd756'c746,
                             0xd554'c544'b534'a524,
                             0xbd3c'ad2c'9d1c'8d0c,
                             0x9b1a'8b0a'dd5c'cd4c,
                             0xdb5a'cb4a'bb3a'ab2a,
                             0xbf3e'af2e'9f1e'8f0e,
                             0x9918'8908'df5e'cf4e,
                             0xd958'c948'b938'a928});
  VsxsegXeiXX<UInt16, 7, 1>(0xc5008427,  // Vsuxseg7ei8.v v8, (x1), v16, v0.t
                            {0xb332'a322'9312'8302,
                             0x8100'e362'd352'c342,
                             0xc140'b130'a120'9110,
                             0x9716'8706'e160'd150,
                             0xd756'c746'b736'a726,
                             0xa524'9514'8504'e766,
                             0xe564'd554'c544'b534,
                             0xbd3c'ad2c'9d1c'8d0c,
                             0x8b0a'ed6c'dd5c'cd4c,
                             0xcb4a'bb3a'ab2a'9b1a,
                             0x9f1e'8f0e'eb6a'db5a,
                             0xdf5e'cf4e'bf3e'af2e,
                             0xa928'9918'8908'ef6e,
                             0xe968'd958'c948'b938});
  VsxsegXeiXX<UInt16, 8, 1>(0xe5008427,  // Vsuxseg8ei8.v v8, (x1), v16, v0.t
                            {0xb332'a322'9312'8302,
                             0xf372'e362'd352'c342,
                             0xb130'a120'9110'8100,
                             0xf170'e160'd150'c140,
                             0xb736'a726'9716'8706,
                             0xf776'e766'd756'c746,
                             0xb534'a524'9514'8504,
                             0xf574'e564'd554'c544,
                             0xbd3c'ad2c'9d1c'8d0c,
                             0xfd7c'ed6c'dd5c'cd4c,
                             0xbb3a'ab2a'9b1a'8b0a,
                             0xfb7a'eb6a'db5a'cb4a,
                             0xbf3e'af2e'9f1e'8f0e,
                             0xff7e'ef6e'df5e'cf4e,
                             0xb938'a928'9918'8908,
                             0xf978'e968'd958'c948});
  VsxsegXeiXX<UInt32, 1, 1>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8302'8100'8706'8504, 0x8b0a'8908'8f0e'8d0c});
  VsxsegXeiXX<UInt32, 1, 2>(
      0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
      {0x8302'8100'8706'8504, 0x8b0a'8908'8f0e'8d0c, 0x9716'9514'9b1a'9918, 0x9312'9110'9f1e'9d1c});
  VsxsegXeiXX<UInt32, 1, 4>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8302'8100'8706'8504,
                             0x8b0a'8908'8f0e'8d0c,
                             0x9716'9514'9b1a'9918,
                             0x9312'9110'9f1e'9d1c,
                             0xa322'a120'bb3a'b938,
                             0xaf2e'ad2c'bf3e'bd3c,
                             0xb332'b130'b736'b534,
                             0xab2a'a928'a726'a524});
  VsxsegXeiXX<UInt32, 1, 8>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8302'8100'8706'8504,
                             0x8b0a'8908'8f0e'8d0c,
                             0x9716'9514'9b1a'9918,
                             0x9312'9110'9f1e'9d1c,
                             0xa322'a120'bb3a'b938,
                             0xaf2e'ad2c'bf3e'bd3c,
                             0xb332'b130'b736'b534,
                             0xab2a'a928'a726'a524,
                             0xcb4a'c948'eb6a'e968,
                             0xdf5e'dd5c'd352'd150,
                             0xef6e'ed6c'fb7a'f978,
                             0xff7e'fd7c'cf4e'cd4c,
                             0xdb5a'd958'f776'f574,
                             0xf372'f170'd756'd554,
                             0xe362'e160'e766'e564,
                             0xc746'c544'c342'c140});
  VsxsegXeiXX<UInt32, 2, 1>(
      0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
      {0x9716'9514'8706'8504, 0x9312'9110'8302'8100, 0x9f1e'9d1c'8f0e'8d0c, 0x9b1a'9918'8b0a'8908});
  VsxsegXeiXX<UInt32, 2, 2>(0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0xa726'a524'8706'8504,
                             0xa322'a120'8302'8100,
                             0xaf2e'ad2c'8f0e'8d0c,
                             0xab2a'a928'8b0a'8908,
                             0xbb3a'b938'9b1a'9918,
                             0xb736'b534'9716'9514,
                             0xbf3e'bd3c'9f1e'9d1c,
                             0xb332'b130'9312'9110});
  VsxsegXeiXX<UInt32, 2, 4>(0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0xc746'c544'8706'8504,
                             0xc342'c140'8302'8100,
                             0xcf4e'cd4c'8f0e'8d0c,
                             0xcb4a'c948'8b0a'8908,
                             0xdb5a'd958'9b1a'9918,
                             0xd756'd554'9716'9514,
                             0xdf5e'dd5c'9f1e'9d1c,
                             0xd352'd150'9312'9110,
                             0xfb7a'f978'bb3a'b938,
                             0xe362'e160'a322'a120,
                             0xff7e'fd7c'bf3e'bd3c,
                             0xef6e'ed6c'af2e'ad2c,
                             0xf776'f574'b736'b534,
                             0xf372'f170'b332'b130,
                             0xe766'e564'a726'a524,
                             0xeb6a'e968'ab2a'a928});
  VsxsegXeiXX<UInt32, 3, 1>(0x45008427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0x8302'8100'a726'a524,
                             0xa322'a120'9312'9110,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0x8b0a'8908'af2e'ad2c,
                             0xab2a'a928'9b1a'9918});
  VsxsegXeiXX<UInt32, 3, 2>(0x45008427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0xa726'a524'8706'8504,
                             0x8302'8100'c746'c544,
                             0xc342'c140'a322'a120,
                             0xaf2e'ad2c'8f0e'8d0c,
                             0x8b0a'8908'cf4e'cd4c,
                             0xcb4a'c948'ab2a'a928,
                             0xbb3a'b938'9b1a'9918,
                             0x9716'9514'db5a'd958,
                             0xd756'd554'b736'b534,
                             0xbf3e'bd3c'9f1e'9d1c,
                             0x9312'9110'df5e'dd5c,
                             0xd352'd150'b332'b130});
  VsxsegXeiXX<UInt32, 4, 1>(0x65008427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0xb736'b534'a726'a524,
                             0x9312'9110'8302'8100,
                             0xb332'b130'a322'a120,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0xbf3e'bd3c'af2e'ad2c,
                             0x9b1a'9918'8b0a'8908,
                             0xbb3a'b938'ab2a'a928});
  VsxsegXeiXX<UInt32, 4, 2>(0x65008427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0xa726'a524'8706'8504,
                             0xe766'e564'c746'c544,
                             0xa322'a120'8302'8100,
                             0xe362'e160'c342'c140,
                             0xaf2e'ad2c'8f0e'8d0c,
                             0xef6e'ed6c'cf4e'cd4c,
                             0xab2a'a928'8b0a'8908,
                             0xeb6a'e968'cb4a'c948,
                             0xbb3a'b938'9b1a'9918,
                             0xfb7a'f978'db5a'd958,
                             0xb736'b534'9716'9514,
                             0xf776'f574'd756'd554,
                             0xbf3e'bd3c'9f1e'9d1c,
                             0xff7e'fd7c'df5e'dd5c,
                             0xb332'b130'9312'9110,
                             0xf372'f170'd352'd150});
  VsxsegXeiXX<UInt32, 5, 1>(0x85008427,  // Vsuxseg5ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0xb736'b534'a726'a524,
                             0x8302'8100'c746'c544,
                             0xa322'a120'9312'9110,
                             0xc342'c140'b332'b130,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0xbf3e'bd3c'af2e'ad2c,
                             0x8b0a'8908'cf4e'cd4c,
                             0xab2a'a928'9b1a'9918,
                             0xcb4a'c948'bb3a'b938});
  VsxsegXeiXX<UInt32, 6, 1>(0xa5008427,  // Vsuxseg6ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0xb736'b534'a726'a524,
                             0xd756'd554'c746'c544,
                             0x9312'9110'8302'8100,
                             0xb332'b130'a322'a120,
                             0xd352'd150'c342'c140,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0xbf3e'bd3c'af2e'ad2c,
                             0xdf5e'dd5c'cf4e'cd4c,
                             0x9b1a'9918'8b0a'8908,
                             0xbb3a'b938'ab2a'a928,
                             0xdb5a'd958'cb4a'c948});
  VsxsegXeiXX<UInt32, 7, 1>(0xc5008427,  // Vsuxseg7ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0xb736'b534'a726'a524,
                             0xd756'd554'c746'c544,
                             0x8302'8100'e766'e564,
                             0xa322'a120'9312'9110,
                             0xc342'c140'b332'b130,
                             0xe362'e160'd352'd150,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0xbf3e'bd3c'af2e'ad2c,
                             0xdf5e'dd5c'cf4e'cd4c,
                             0x8b0a'8908'ef6e'ed6c,
                             0xab2a'a928'9b1a'9918,
                             0xcb4a'c948'bb3a'b938,
                             0xeb6a'e968'db5a'd958});
  VsxsegXeiXX<UInt32, 8, 1>(0xe5008427,  // Vsuxseg8ei8.v v8, (x1), v16, v0.t
                            {0x9716'9514'8706'8504,
                             0xb736'b534'a726'a524,
                             0xd756'd554'c746'c544,
                             0xf776'f574'e766'e564,
                             0x9312'9110'8302'8100,
                             0xb332'b130'a322'a120,
                             0xd352'd150'c342'c140,
                             0xf372'f170'e362'e160,
                             0x9f1e'9d1c'8f0e'8d0c,
                             0xbf3e'bd3c'af2e'ad2c,
                             0xdf5e'dd5c'cf4e'cd4c,
                             0xff7e'fd7c'ef6e'ed6c,
                             0x9b1a'9918'8b0a'8908,
                             0xbb3a'b938'ab2a'a928,
                             0xdb5a'd958'cb4a'c948,
                             0xfb7a'f978'eb6a'e968});
  VsxsegXeiXX<UInt64, 1, 1>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100});
  VsxsegXeiXX<UInt64, 1, 2>(
      0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
      {0x8f0e'8d0c'8b0a'8908, 0x8706'8504'8302'8100, 0x9f1e'9d1c'9b1a'9918, 0x9716'9514'9312'9110});
  VsxsegXeiXX<UInt64, 1, 4>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x8706'8504'8302'8100,
                             0x9f1e'9d1c'9b1a'9918,
                             0x9716'9514'9312'9110,
                             0xb736'b534'b332'b130,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xa726'a524'a322'a120});
  VsxsegXeiXX<UInt64, 1, 8>(0x5008427,  // Vsuxei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x8706'8504'8302'8100,
                             0x9f1e'9d1c'9b1a'9918,
                             0x9716'9514'9312'9110,
                             0xb736'b534'b332'b130,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xa726'a524'a322'a120,
                             0xf776'f574'f372'f170,
                             0xc746'c544'c342'c140,
                             0xff7e'fd7c'fb7a'f978,
                             0xdf5e'dd5c'db5a'd958,
                             0xef6e'ed6c'eb6a'e968,
                             0xe766'e564'e362'e160,
                             0xcf4e'cd4c'cb4a'c948,
                             0xd756'd554'd352'd150});
  VsxsegXeiXX<UInt64, 2, 1>(
      0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
      {0x8f0e'8d0c'8b0a'8908, 0x9f1e'9d1c'9b1a'9918, 0x8706'8504'8302'8100, 0x9716'9514'9312'9110});
  VsxsegXeiXX<UInt64, 2, 2>(0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0xaf2e'ad2c'ab2a'a928,
                             0x8706'8504'8302'8100,
                             0xa726'a524'a322'a120,
                             0x9f1e'9d1c'9b1a'9918,
                             0xbf3e'bd3c'bb3a'b938,
                             0x9716'9514'9312'9110,
                             0xb736'b534'b332'b130});
  VsxsegXeiXX<UInt64, 2, 4>(0x25008427,  // Vsuxseg2ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0xcf4e'cd4c'cb4a'c948,
                             0x8706'8504'8302'8100,
                             0xc746'c544'c342'c140,
                             0x9f1e'9d1c'9b1a'9918,
                             0xdf5e'dd5c'db5a'd958,
                             0x9716'9514'9312'9110,
                             0xd756'd554'd352'd150,
                             0xb736'b534'b332'b130,
                             0xf776'f574'f372'f170,
                             0xaf2e'ad2c'ab2a'a928,
                             0xef6e'ed6c'eb6a'e968,
                             0xbf3e'bd3c'bb3a'b938,
                             0xff7e'fd7c'fb7a'f978,
                             0xa726'a524'a322'a120,
                             0xe766'e564'e362'e160});
  VsxsegXeiXX<UInt64, 3, 1>(0x45008427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120});
  VsxsegXeiXX<UInt64, 3, 2>(0x45008427,  // Vsuxseg3ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0xaf2e'ad2c'ab2a'a928,
                             0xcf4e'cd4c'cb4a'c948,
                             0x8706'8504'8302'8100,
                             0xa726'a524'a322'a120,
                             0xc746'c544'c342'c140,
                             0x9f1e'9d1c'9b1a'9918,
                             0xbf3e'bd3c'bb3a'b938,
                             0xdf5e'dd5c'db5a'd958,
                             0x9716'9514'9312'9110,
                             0xb736'b534'b332'b130,
                             0xd756'd554'd352'd150});
  VsxsegXeiXX<UInt64, 4, 1>(0x65008427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120,
                             0xb736'b534'b332'b130});
  VsxsegXeiXX<UInt64, 4, 2>(0x65008427,  // Vsuxseg4ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0xaf2e'ad2c'ab2a'a928,
                             0xcf4e'cd4c'cb4a'c948,
                             0xef6e'ed6c'eb6a'e968,
                             0x8706'8504'8302'8100,
                             0xa726'a524'a322'a120,
                             0xc746'c544'c342'c140,
                             0xe766'e564'e362'e160,
                             0x9f1e'9d1c'9b1a'9918,
                             0xbf3e'bd3c'bb3a'b938,
                             0xdf5e'dd5c'db5a'd958,
                             0xff7e'fd7c'fb7a'f978,
                             0x9716'9514'9312'9110,
                             0xb736'b534'b332'b130,
                             0xd756'd554'd352'd150,
                             0xf776'f574'f372'f170});
  VsxsegXeiXX<UInt64, 5, 1>(0x85008427,  // Vsuxseg5ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xcf4e'cd4c'cb4a'c948,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120,
                             0xb736'b534'b332'b130,
                             0xc746'c544'c342'c140});
  VsxsegXeiXX<UInt64, 6, 1>(0xa5008427,  // Vsuxseg6ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xcf4e'cd4c'cb4a'c948,
                             0xdf5e'dd5c'db5a'd958,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120,
                             0xb736'b534'b332'b130,
                             0xc746'c544'c342'c140,
                             0xd756'd554'd352'd150});
  VsxsegXeiXX<UInt64, 7, 1>(0xc5008427,  // Vsuxseg7ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xcf4e'cd4c'cb4a'c948,
                             0xdf5e'dd5c'db5a'd958,
                             0xef6e'ed6c'eb6a'e968,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120,
                             0xb736'b534'b332'b130,
                             0xc746'c544'c342'c140,
                             0xd756'd554'd352'd150,
                             0xe766'e564'e362'e160});
  VsxsegXeiXX<UInt64, 8, 1>(0xe5008427,  // Vsuxseg8ei8.v v8, (x1), v16, v0.t
                            {0x8f0e'8d0c'8b0a'8908,
                             0x9f1e'9d1c'9b1a'9918,
                             0xaf2e'ad2c'ab2a'a928,
                             0xbf3e'bd3c'bb3a'b938,
                             0xcf4e'cd4c'cb4a'c948,
                             0xdf5e'dd5c'db5a'd958,
                             0xef6e'ed6c'eb6a'e968,
                             0xff7e'fd7c'fb7a'f978,
                             0x8706'8504'8302'8100,
                             0x9716'9514'9312'9110,
                             0xa726'a524'a322'a120,
                             0xb736'b534'b332'b130,
                             0xc746'c544'c342'c140,
                             0xd756'd554'd352'd150,
                             0xe766'e564'e362'e160,
                             0xf776'f574'f372'f170});
}

TEST_F(Riscv64InterpreterTest, TestVssegXeXX) {
  TestVssegXeXX<UInt8, 1, 1>(0x000008427,  // vsse8.v v8, (x1), v0.t
                             {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908});
  TestVssegXeXX<UInt8, 1, 2>(
      0x000008427,  // vsse8.v v8, (x1), v0.t
      {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908, 0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918});
  TestVssegXeXX<UInt8, 1, 4>(0x000008427,  // vsse8.v v8, (x1), v0.t
                             {0x8706'8504'8302'8100,
                              0x8f0e'8d0c'8b0a'8908,
                              0x9716'9514'9312'9110,
                              0x9f1e'9d1c'9b1a'9918,
                              0xa726'a524'a322'a120,
                              0xaf2e'ad2c'ab2a'a928,
                              0xb736'b534'b332'b130,
                              0xbf3e'bd3c'bb3a'b938});
  TestVssegXeXX<UInt8, 1, 8>(0x000008427,  // vsse8.v v8, (x1), v0.t
                             {0x8706'8504'8302'8100,
                              0x8f0e'8d0c'8b0a'8908,
                              0x9716'9514'9312'9110,
                              0x9f1e'9d1c'9b1a'9918,
                              0xa726'a524'a322'a120,
                              0xaf2e'ad2c'ab2a'a928,
                              0xb736'b534'b332'b130,
                              0xbf3e'bd3c'bb3a'b938,
                              0xc746'c544'c342'c140,
                              0xcf4e'cd4c'cb4a'c948,
                              0xd756'd554'd352'd150,
                              0xdf5e'dd5c'db5a'd958,
                              0xe766'e564'e362'e160,
                              0xef6e'ed6c'eb6a'e968,
                              0xf776'f574'f372'f170,
                              0xff7e'fd7c'fb7a'f978});
  TestVssegXeXX<UInt8, 2, 1>(
      0x20008427,  // vsseg2e8.v v8, (x1), v0.t
      {0x9383'1202'9181'1000, 0x9787'1606'9585'1404, 0x9b8b'1a0a'9989'1808, 0x9f8f'1e0e'9d8d'1c0c});
  TestVssegXeXX<UInt8, 2, 2>(0x20008427,  // vsseg2e8.v v8, (x1), v0.t
                             {0xa383'2202'a181'2000,
                              0xa787'2606'a585'2404,
                              0xab8b'2a0a'a989'2808,
                              0xaf8f'2e0e'ad8d'2c0c,
                              0xb393'3212'b191'3010,
                              0xb797'3616'b595'3414,
                              0xbb9b'3a1a'b999'3818,
                              0xbf9f'3e1e'bd9d'3c1c});
  TestVssegXeXX<UInt8, 2, 4>(0x20008427,  // vsseg2e8.v v8, (x1), v0.t
                             {0xc383'4202'c181'4000,
                              0xc787'4606'c585'4404,
                              0xcb8b'4a0a'c989'4808,
                              0xcf8f'4e0e'cd8d'4c0c,
                              0xd393'5212'd191'5010,
                              0xd797'5616'd595'5414,
                              0xdb9b'5a1a'd999'5818,
                              0xdf9f'5e1e'dd9d'5c1c,
                              0xe3a3'6222'e1a1'6020,
                              0xe7a7'6626'e5a5'6424,
                              0xebab'6a2a'e9a9'6828,
                              0xefaf'6e2e'edad'6c2c,
                              0xf3b3'7232'f1b1'7030,
                              0xf7b7'7636'f5b5'7434,
                              0xfbbb'7a3a'f9b9'7838,
                              0xffbf'7e3e'fdbd'7c3c});
  TestVssegXeXX<UInt8, 3, 1>(0x40008427,  // vsseg3e8.v v8, (x1), v0.t
                             {0x1202'a191'8120'1000,
                              0x8524'1404'a393'8322,
                              0xa797'8726'1606'a595,
                              0x1a0a'a999'8928'1808,
                              0x8d2c'1c0c'ab9b'8b2a,
                              0xaf9f'8f2e'1e0e'ad9d});
  TestVssegXeXX<UInt8, 3, 2>(0x40008427,  // vsseg3e8.v v8, (x1), v0.t
                             {0x2202'c1a1'8140'2000,
                              0x8544'2404'c3a3'8342,
                              0xc7a7'8746'2606'c5a5,
                              0x2a0a'c9a9'8948'2808,
                              0x8d4c'2c0c'cbab'8b4a,
                              0xcfaf'8f4e'2e0e'cdad,
                              0x3212'd1b1'9150'3010,
                              0x9554'3414'd3b3'9352,
                              0xd7b7'9756'3616'd5b5,
                              0x3a1a'd9b9'9958'3818,
                              0x9d5c'3c1c'dbbb'9b5a,
                              0xdfbf'9f5e'3e1e'ddbd});
  TestVssegXeXX<UInt8, 4, 1>(0x60008427,  // vsseg4e8.v v8, (x1), v0.t
                             {0xb1a1'9181'3020'1000,
                              0xb3a3'9383'3222'1202,
                              0xb5a5'9585'3424'1404,
                              0xb7a7'9787'3626'1606,
                              0xb9a9'9989'3828'1808,
                              0xbbab'9b8b'3a2a'1a0a,
                              0xbdad'9d8d'3c2c'1c0c,
                              0xbfaf'9f8f'3e2e'1e0e});
  TestVssegXeXX<UInt8, 4, 2>(0x60008427,  // vsseg4e8.v v8, (x1), v0.t
                             {0xe1c1'a181'6040'2000,
                              0xe3c3'a383'6242'2202,
                              0xe5c5'a585'6444'2404,
                              0xe7c7'a787'6646'2606,
                              0xe9c9'a989'6848'2808,
                              0xebcb'ab8b'6a4a'2a0a,
                              0xedcd'ad8d'6c4c'2c0c,
                              0xefcf'af8f'6e4e'2e0e,
                              0xf1d1'b191'7050'3010,
                              0xf3d3'b393'7252'3212,
                              0xf5d5'b595'7454'3414,
                              0xf7d7'b797'7656'3616,
                              0xf9d9'b999'7858'3818,
                              0xfbdb'bb9b'7a5a'3a1a,
                              0xfddd'bd9d'7c5c'3c1c,
                              0xffdf'bf9f'7e5e'3e1e});
  TestVssegXeXX<UInt8, 5, 1>(0x80008427,  // vsseg5e8.v v8, (x1), v0.t
                             {0xa191'8140'3020'1000,
                              0x8342'3222'1202'c1b1,
                              0x3424'1404'c3b3'a393,
                              0x1606'c5b5'a595'8544,
                              0xc7b7'a797'8746'3626,
                              0xa999'8948'3828'1808,
                              0x8b4a'3a2a'1a0a'c9b9,
                              0x3c2c'1c0c'cbbb'ab9b,
                              0x1e0e'cdbd'ad9d'8d4c,
                              0xcfbf'af9f'8f4e'3e2e});
  TestVssegXeXX<UInt8, 6, 1>(0xa0008427,  // vsseg6e8.v v8, (x1), v0.t
                             {0x9181'5040'3020'1000,
                              0x3222'1202'd1c1'b1a1,
                              0xd3c3'b3a3'9383'5242,
                              0x9585'5444'3424'1404,
                              0x3626'1606'd5c5'b5a5,
                              0xd7c7'b7a7'9787'5646,
                              0x9989'5848'3828'1808,
                              0x3a2a'1a0a'd9c9'b9a9,
                              0xdbcb'bbab'9b8b'5a4a,
                              0x9d8d'5c4c'3c2c'1c0c,
                              0x3e2e'1e0e'ddcd'bdad,
                              0xdfcf'bfaf'9f8f'5e4e});
  TestVssegXeXX<UInt8, 7, 1>(0xc0008427,  // vsseg7e8.v v8, (x1), v0.t
                             {0x8160'5040'3020'1000,
                              0x1202'e1d1'c1b1'a191,
                              0xa393'8362'5242'3222,
                              0x3424'1404'e3d3'c3b3,
                              0xc5b5'a595'8564'5444,
                              0x5646'3626'1606'e5d5,
                              0xe7d7'c7b7'a797'8766,
                              0x8968'5848'3828'1808,
                              0x1a0a'e9d9'c9b9'a999,
                              0xab9b'8b6a'5a4a'3a2a,
                              0x3c2c'1c0c'ebdb'cbbb,
                              0xcdbd'ad9d'8d6c'5c4c,
                              0x5e4e'3e2e'1e0e'eddd,
                              0xefdf'cfbf'af9f'8f6e});
  TestVssegXeXX<UInt8, 8, 1>(0xe0008427,  // vsseg8e8.v v8, (x1), v0.t
                             {0x7060'5040'3020'1000,
                              0xf1e1'd1c1'b1a1'9181,
                              0x7262'5242'3222'1202,
                              0xf3e3'd3c3'b3a3'9383,
                              0x7464'5444'3424'1404,
                              0xf5e5'd5c5'b5a5'9585,
                              0x7666'5646'3626'1606,
                              0xf7e7'd7c7'b7a7'9787,
                              0x7868'5848'3828'1808,
                              0xf9e9'd9c9'b9a9'9989,
                              0x7a6a'5a4a'3a2a'1a0a,
                              0xfbeb'dbcb'bbab'9b8b,
                              0x7c6c'5c4c'3c2c'1c0c,
                              0xfded'ddcd'bdad'9d8d,
                              0x7e6e'5e4e'3e2e'1e0e,
                              0xffef'dfcf'bfaf'9f8f});
  TestVssegXeXX<UInt16, 1, 1>(0x000d427,  // vse16.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908});
  TestVssegXeXX<UInt16, 1, 2>(
      0x000d427,  // vse16.v v8, (x1), v0.t
      {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908, 0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918});
  TestVssegXeXX<UInt16, 1, 4>(0x000d427,  // vse16.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938});
  TestVssegXeXX<UInt16, 1, 8>(0x000d427,  // vse16.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938,
                               0xc746'c544'c342'c140,
                               0xcf4e'cd4c'cb4a'c948,
                               0xd756'd554'd352'd150,
                               0xdf5e'dd5c'db5a'd958,
                               0xe766'e564'e362'e160,
                               0xef6e'ed6c'eb6a'e968,
                               0xf776'f574'f372'f170,
                               0xff7e'fd7c'fb7a'f978});
  TestVssegXeXX<UInt16, 2, 1>(
      0x2000d427,  // vsseg2e16.v v8, (x1), v0.t
      {0x9312'8302'9110'8100, 0x9716'8706'9514'8504, 0x9b1a'8b0a'9918'8908, 0x9f1e'8f0e'9d1c'8d0c});
  TestVssegXeXX<UInt16, 2, 2>(0x2000d427,  // vsseg2e16.v v8, (x1), v0.t
                              {0xa322'8302'a120'8100,
                               0xa726'8706'a524'8504,
                               0xab2a'8b0a'a928'8908,
                               0xaf2e'8f0e'ad2c'8d0c,
                               0xb332'9312'b130'9110,
                               0xb736'9716'b534'9514,
                               0xbb3a'9b1a'b938'9918,
                               0xbf3e'9f1e'bd3c'9d1c});
  TestVssegXeXX<UInt16, 2, 4>(0x2000d427,  // vsseg2e16.v v8, (x1), v0.t
                              {0xc342'8302'c140'8100,
                               0xc746'8706'c544'8504,
                               0xcb4a'8b0a'c948'8908,
                               0xcf4e'8f0e'cd4c'8d0c,
                               0xd352'9312'd150'9110,
                               0xd756'9716'd554'9514,
                               0xdb5a'9b1a'd958'9918,
                               0xdf5e'9f1e'dd5c'9d1c,
                               0xe362'a322'e160'a120,
                               0xe766'a726'e564'a524,
                               0xeb6a'ab2a'e968'a928,
                               0xef6e'af2e'ed6c'ad2c,
                               0xf372'b332'f170'b130,
                               0xf776'b736'f574'b534,
                               0xfb7a'bb3a'f978'b938,
                               0xff7e'bf3e'fd7c'bd3c});
  TestVssegXeXX<UInt16, 3, 1>(0x4000d427,  // vsseg3e16.v v8, (x1), v0.t
                              {0x8302'a120'9110'8100,
                               0x9514'8504'a322'9312,
                               0xa726'9716'8706'a524,
                               0x8b0a'a928'9918'8908,
                               0x9d1c'8d0c'ab2a'9b1a,
                               0xaf2e'9f1e'8f0e'ad2c});
  TestVssegXeXX<UInt16, 3, 2>(0x4000d427,  // vsseg3e16.v v8, (x1), v0.t
                              {0x8302'c140'a120'8100,
                               0xa524'8504'c342'a322,
                               0xc746'a726'8706'c544,
                               0x8b0a'c948'a928'8908,
                               0xad2c'8d0c'cb4a'ab2a,
                               0xcf4e'af2e'8f0e'cd4c,
                               0x9312'd150'b130'9110,
                               0xb534'9514'd352'b332,
                               0xd756'b736'9716'd554,
                               0x9b1a'd958'b938'9918,
                               0xbd3c'9d1c'db5a'bb3a,
                               0xdf5e'bf3e'9f1e'dd5c});
  TestVssegXeXX<UInt16, 4, 1>(0x6000d427,  // vsseg4e16.v v8, (x1), v0.t
                              {0xb130'a120'9110'8100,
                               0xb332'a322'9312'8302,
                               0xb534'a524'9514'8504,
                               0xb736'a726'9716'8706,
                               0xb938'a928'9918'8908,
                               0xbb3a'ab2a'9b1a'8b0a,
                               0xbd3c'ad2c'9d1c'8d0c,
                               0xbf3e'af2e'9f1e'8f0e});
  TestVssegXeXX<UInt16, 4, 2>(0x6000d427,  // vsseg4e16.v v8, (x1), v0.t
                              {0xe160'c140'a120'8100,
                               0xe362'c342'a322'8302,
                               0xe564'c544'a524'8504,
                               0xe766'c746'a726'8706,
                               0xe968'c948'a928'8908,
                               0xeb6a'cb4a'ab2a'8b0a,
                               0xed6c'cd4c'ad2c'8d0c,
                               0xef6e'cf4e'af2e'8f0e,
                               0xf170'd150'b130'9110,
                               0xf372'd352'b332'9312,
                               0xf574'd554'b534'9514,
                               0xf776'd756'b736'9716,
                               0xf978'd958'b938'9918,
                               0xfb7a'db5a'bb3a'9b1a,
                               0xfd7c'dd5c'bd3c'9d1c,
                               0xff7e'df5e'bf3e'9f1e});
  TestVssegXeXX<UInt16, 5, 1>(0x8000d427,  // vsseg5e16.v v8, (x1), v0.t
                              {0xb130'a120'9110'8100,
                               0xa322'9312'8302'c140,
                               0x9514'8504'c342'b332,
                               0x8706'c544'b534'a524,
                               0xc746'b736'a726'9716,
                               0xb938'a928'9918'8908,
                               0xab2a'9b1a'8b0a'c948,
                               0x9d1c'8d0c'cb4a'bb3a,
                               0x8f0e'cd4c'bd3c'ad2c,
                               0xcf4e'bf3e'af2e'9f1e});
  TestVssegXeXX<UInt16, 6, 1>(0xa000d427,  // vsseg6e16.v v8, (x1), v0.t
                              {0xb130'a120'9110'8100,
                               0x9312'8302'd150'c140,
                               0xd352'c342'b332'a322,
                               0xb534'a524'9514'8504,
                               0x9716'8706'd554'c544,
                               0xd756'c746'b736'a726,
                               0xb938'a928'9918'8908,
                               0x9b1a'8b0a'd958'c948,
                               0xdb5a'cb4a'bb3a'ab2a,
                               0xbd3c'ad2c'9d1c'8d0c,
                               0x9f1e'8f0e'dd5c'cd4c,
                               0xdf5e'cf4e'bf3e'af2e});
  TestVssegXeXX<UInt16, 7, 1>(0xc000d427,  // vsseg7e16.v v8, (x1), v0.t
                              {0xb130'a120'9110'8100,
                               0x8302'e160'd150'c140,
                               0xc342'b332'a322'9312,
                               0x9514'8504'e362'd352,
                               0xd554'c544'b534'a524,
                               0xa726'9716'8706'e564,
                               0xe766'd756'c746'b736,
                               0xb938'a928'9918'8908,
                               0x8b0a'e968'd958'c948,
                               0xcb4a'bb3a'ab2a'9b1a,
                               0x9d1c'8d0c'eb6a'db5a,
                               0xdd5c'cd4c'bd3c'ad2c,
                               0xaf2e'9f1e'8f0e'ed6c,
                               0xef6e'df5e'cf4e'bf3e});
  TestVssegXeXX<UInt16, 8, 1>(0xe000d427,  // vsseg8e16.v v8, (x1), v0.t
                              {0xb130'a120'9110'8100,
                               0xf170'e160'd150'c140,
                               0xb332'a322'9312'8302,
                               0xf372'e362'd352'c342,
                               0xb534'a524'9514'8504,
                               0xf574'e564'd554'c544,
                               0xb736'a726'9716'8706,
                               0xf776'e766'd756'c746,
                               0xb938'a928'9918'8908,
                               0xf978'e968'd958'c948,
                               0xbb3a'ab2a'9b1a'8b0a,
                               0xfb7a'eb6a'db5a'cb4a,
                               0xbd3c'ad2c'9d1c'8d0c,
                               0xfd7c'ed6c'dd5c'cd4c,
                               0xbf3e'af2e'9f1e'8f0e,
                               0xff7e'ef6e'df5e'cf4e});
  TestVssegXeXX<UInt32, 1, 1>(0x000e427,  // vse32.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908});
  TestVssegXeXX<UInt32, 1, 2>(
      0x000e427,  // vse32.v v8, (x1), v0.t
      {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908, 0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918});
  TestVssegXeXX<UInt32, 1, 4>(0x000e427,  // vse32.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938});
  TestVssegXeXX<UInt32, 1, 8>(0x000e427,  // vse32.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938,
                               0xc746'c544'c342'c140,
                               0xcf4e'cd4c'cb4a'c948,
                               0xd756'd554'd352'd150,
                               0xdf5e'dd5c'db5a'd958,
                               0xe766'e564'e362'e160,
                               0xef6e'ed6c'eb6a'e968,
                               0xf776'f574'f372'f170,
                               0xff7e'fd7c'fb7a'f978});
  TestVssegXeXX<UInt32, 2, 1>(
      0x2000e427,  // vsseg2e32.v v8, (x1), v0.t
      {0x9312'9110'8302'8100, 0x9716'9514'8706'8504, 0x9b1a'9918'8b0a'8908, 0x9f1e'9d1c'8f0e'8d0c});
  TestVssegXeXX<UInt32, 2, 2>(0x2000e427,  // vsseg2e32.v v8, (x1), v0.t
                              {0xa322'a120'8302'8100,
                               0xa726'a524'8706'8504,
                               0xab2a'a928'8b0a'8908,
                               0xaf2e'ad2c'8f0e'8d0c,
                               0xb332'b130'9312'9110,
                               0xb736'b534'9716'9514,
                               0xbb3a'b938'9b1a'9918,
                               0xbf3e'bd3c'9f1e'9d1c});
  TestVssegXeXX<UInt32, 2, 4>(0x2000e427,  // vsseg2e32.v v8, (x1), v0.t
                              {0xc342'c140'8302'8100,
                               0xc746'c544'8706'8504,
                               0xcb4a'c948'8b0a'8908,
                               0xcf4e'cd4c'8f0e'8d0c,
                               0xd352'd150'9312'9110,
                               0xd756'd554'9716'9514,
                               0xdb5a'd958'9b1a'9918,
                               0xdf5e'dd5c'9f1e'9d1c,
                               0xe362'e160'a322'a120,
                               0xe766'e564'a726'a524,
                               0xeb6a'e968'ab2a'a928,
                               0xef6e'ed6c'af2e'ad2c,
                               0xf372'f170'b332'b130,
                               0xf776'f574'b736'b534,
                               0xfb7a'f978'bb3a'b938,
                               0xff7e'fd7c'bf3e'bd3c});
  TestVssegXeXX<UInt32, 3, 1>(0x4000e427,  // vsseg3e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0x8706'8504'a322'a120,
                               0xa726'a524'9716'9514,
                               0x9b1a'9918'8b0a'8908,
                               0x8f0e'8d0c'ab2a'a928,
                               0xaf2e'ad2c'9f1e'9d1c});
  TestVssegXeXX<UInt32, 3, 2>(0x4000e427,  // vsseg3e32.v v8, (x1), v0.t
                              {0xa322'a120'8302'8100,
                               0x8706'8504'c342'c140,
                               0xc746'c544'a726'a524,
                               0xab2a'a928'8b0a'8908,
                               0x8f0e'8d0c'cb4a'c948,
                               0xcf4e'cd4c'af2e'ad2c,
                               0xb332'b130'9312'9110,
                               0x9716'9514'd352'd150,
                               0xd756'd554'b736'b534,
                               0xbb3a'b938'9b1a'9918,
                               0x9f1e'9d1c'db5a'd958,
                               0xdf5e'dd5c'bf3e'bd3c});
  TestVssegXeXX<UInt32, 4, 1>(0x6000e427,  // vsseg4e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0xb332'b130'a322'a120,
                               0x9716'9514'8706'8504,
                               0xb736'b534'a726'a524,
                               0x9b1a'9918'8b0a'8908,
                               0xbb3a'b938'ab2a'a928,
                               0x9f1e'9d1c'8f0e'8d0c,
                               0xbf3e'bd3c'af2e'ad2c});
  TestVssegXeXX<UInt32, 4, 2>(0x6000e427,  // vsseg4e32.v v8, (x1), v0.t
                              {0xa322'a120'8302'8100,
                               0xe362'e160'c342'c140,
                               0xa726'a524'8706'8504,
                               0xe766'e564'c746'c544,
                               0xab2a'a928'8b0a'8908,
                               0xeb6a'e968'cb4a'c948,
                               0xaf2e'ad2c'8f0e'8d0c,
                               0xef6e'ed6c'cf4e'cd4c,
                               0xb332'b130'9312'9110,
                               0xf372'f170'd352'd150,
                               0xb736'b534'9716'9514,
                               0xf776'f574'd756'd554,
                               0xbb3a'b938'9b1a'9918,
                               0xfb7a'f978'db5a'd958,
                               0xbf3e'bd3c'9f1e'9d1c,
                               0xff7e'fd7c'df5e'dd5c});
  TestVssegXeXX<UInt32, 5, 1>(0x8000e427,  // vsseg5e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0xb332'b130'a322'a120,
                               0x8706'8504'c342'c140,
                               0xa726'a524'9716'9514,
                               0xc746'c544'b736'b534,
                               0x9b1a'9918'8b0a'8908,
                               0xbb3a'b938'ab2a'a928,
                               0x8f0e'8d0c'cb4a'c948,
                               0xaf2e'ad2c'9f1e'9d1c,
                               0xcf4e'cd4c'bf3e'bd3c});
  TestVssegXeXX<UInt32, 6, 1>(0xa000e427,  // vsseg6e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0xb332'b130'a322'a120,
                               0xd352'd150'c342'c140,
                               0x9716'9514'8706'8504,
                               0xb736'b534'a726'a524,
                               0xd756'd554'c746'c544,
                               0x9b1a'9918'8b0a'8908,
                               0xbb3a'b938'ab2a'a928,
                               0xdb5a'd958'cb4a'c948,
                               0x9f1e'9d1c'8f0e'8d0c,
                               0xbf3e'bd3c'af2e'ad2c,
                               0xdf5e'dd5c'cf4e'cd4c});
  TestVssegXeXX<UInt32, 7, 1>(0xc000e427,  // vsseg7e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0xb332'b130'a322'a120,
                               0xd352'd150'c342'c140,
                               0x8706'8504'e362'e160,
                               0xa726'a524'9716'9514,
                               0xc746'c544'b736'b534,
                               0xe766'e564'd756'd554,
                               0x9b1a'9918'8b0a'8908,
                               0xbb3a'b938'ab2a'a928,
                               0xdb5a'd958'cb4a'c948,
                               0x8f0e'8d0c'eb6a'e968,
                               0xaf2e'ad2c'9f1e'9d1c,
                               0xcf4e'cd4c'bf3e'bd3c,
                               0xef6e'ed6c'df5e'dd5c});
  TestVssegXeXX<UInt32, 8, 1>(0xe000e427,  // vsseg8e32.v v8, (x1), v0.t
                              {0x9312'9110'8302'8100,
                               0xb332'b130'a322'a120,
                               0xd352'd150'c342'c140,
                               0xf372'f170'e362'e160,
                               0x9716'9514'8706'8504,
                               0xb736'b534'a726'a524,
                               0xd756'd554'c746'c544,
                               0xf776'f574'e766'e564,
                               0x9b1a'9918'8b0a'8908,
                               0xbb3a'b938'ab2a'a928,
                               0xdb5a'd958'cb4a'c948,
                               0xfb7a'f978'eb6a'e968,
                               0x9f1e'9d1c'8f0e'8d0c,
                               0xbf3e'bd3c'af2e'ad2c,
                               0xdf5e'dd5c'cf4e'cd4c,
                               0xff7e'fd7c'ef6e'ed6c});
  TestVssegXeXX<UInt64, 1, 1>(0x000f427,  // vse64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908});
  TestVssegXeXX<UInt64, 1, 2>(
      0x000f427,  // vse64.v v8, (x1), v0.t
      {0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908, 0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918});
  TestVssegXeXX<UInt64, 1, 4>(0x000f427,  // vse64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938});
  TestVssegXeXX<UInt64, 1, 8>(0x000f427,  // vse64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9716'9514'9312'9110,
                               0x9f1e'9d1c'9b1a'9918,
                               0xa726'a524'a322'a120,
                               0xaf2e'ad2c'ab2a'a928,
                               0xb736'b534'b332'b130,
                               0xbf3e'bd3c'bb3a'b938,
                               0xc746'c544'c342'c140,
                               0xcf4e'cd4c'cb4a'c948,
                               0xd756'd554'd352'd150,
                               0xdf5e'dd5c'db5a'd958,
                               0xe766'e564'e362'e160,
                               0xef6e'ed6c'eb6a'e968,
                               0xf776'f574'f372'f170,
                               0xff7e'fd7c'fb7a'f978});
  TestVssegXeXX<UInt64, 2, 1>(
      0x2000f427,  // vsseg2e64.v v8, (x1), v0.t
      {0x8706'8504'8302'8100, 0x9716'9514'9312'9110, 0x8f0e'8d0c'8b0a'8908, 0x9f1e'9d1c'9b1a'9918});
  TestVssegXeXX<UInt64, 2, 2>(0x2000f427,  // vsseg2e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0xa726'a524'a322'a120,
                               0x8f0e'8d0c'8b0a'8908,
                               0xaf2e'ad2c'ab2a'a928,
                               0x9716'9514'9312'9110,
                               0xb736'b534'b332'b130,
                               0x9f1e'9d1c'9b1a'9918,
                               0xbf3e'bd3c'bb3a'b938});
  TestVssegXeXX<UInt64, 2, 4>(0x2000f427,  // vsseg2e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0xc746'c544'c342'c140,
                               0x8f0e'8d0c'8b0a'8908,
                               0xcf4e'cd4c'cb4a'c948,
                               0x9716'9514'9312'9110,
                               0xd756'd554'd352'd150,
                               0x9f1e'9d1c'9b1a'9918,
                               0xdf5e'dd5c'db5a'd958,
                               0xa726'a524'a322'a120,
                               0xe766'e564'e362'e160,
                               0xaf2e'ad2c'ab2a'a928,
                               0xef6e'ed6c'eb6a'e968,
                               0xb736'b534'b332'b130,
                               0xf776'f574'f372'f170,
                               0xbf3e'bd3c'bb3a'b938,
                               0xff7e'fd7c'fb7a'f978});
  TestVssegXeXX<UInt64, 3, 1>(0x4000f427,  // vsseg3e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928});
  TestVssegXeXX<UInt64, 3, 2>(0x4000f427,  // vsseg3e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0xa726'a524'a322'a120,
                               0xc746'c544'c342'c140,
                               0x8f0e'8d0c'8b0a'8908,
                               0xaf2e'ad2c'ab2a'a928,
                               0xcf4e'cd4c'cb4a'c948,
                               0x9716'9514'9312'9110,
                               0xb736'b534'b332'b130,
                               0xd756'd554'd352'd150,
                               0x9f1e'9d1c'9b1a'9918,
                               0xbf3e'bd3c'bb3a'b938,
                               0xdf5e'dd5c'db5a'd958});
  TestVssegXeXX<UInt64, 4, 1>(0x6000f427,  // vsseg4e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0xb736'b534'b332'b130,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928,
                               0xbf3e'bd3c'bb3a'b938});
  TestVssegXeXX<UInt64, 4, 2>(0x6000f427,  // vsseg4e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0xa726'a524'a322'a120,
                               0xc746'c544'c342'c140,
                               0xe766'e564'e362'e160,
                               0x8f0e'8d0c'8b0a'8908,
                               0xaf2e'ad2c'ab2a'a928,
                               0xcf4e'cd4c'cb4a'c948,
                               0xef6e'ed6c'eb6a'e968,
                               0x9716'9514'9312'9110,
                               0xb736'b534'b332'b130,
                               0xd756'd554'd352'd150,
                               0xf776'f574'f372'f170,
                               0x9f1e'9d1c'9b1a'9918,
                               0xbf3e'bd3c'bb3a'b938,
                               0xdf5e'dd5c'db5a'd958,
                               0xff7e'fd7c'fb7a'f978});
  TestVssegXeXX<UInt64, 5, 1>(0x8000f427,  // vsseg5e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0xb736'b534'b332'b130,
                               0xc746'c544'c342'c140,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928,
                               0xbf3e'bd3c'bb3a'b938,
                               0xcf4e'cd4c'cb4a'c948});
  TestVssegXeXX<UInt64, 6, 1>(0xa000f427,  // vsseg6e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0xb736'b534'b332'b130,
                               0xc746'c544'c342'c140,
                               0xd756'd554'd352'd150,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928,
                               0xbf3e'bd3c'bb3a'b938,
                               0xcf4e'cd4c'cb4a'c948,
                               0xdf5e'dd5c'db5a'd958});
  TestVssegXeXX<UInt64, 7, 1>(0xc000f427,  // vsseg7e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0xb736'b534'b332'b130,
                               0xc746'c544'c342'c140,
                               0xd756'd554'd352'd150,
                               0xe766'e564'e362'e160,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928,
                               0xbf3e'bd3c'bb3a'b938,
                               0xcf4e'cd4c'cb4a'c948,
                               0xdf5e'dd5c'db5a'd958,
                               0xef6e'ed6c'eb6a'e968});
  TestVssegXeXX<UInt64, 8, 1>(0xe000f427,  // vsseg8e64.v v8, (x1), v0.t
                              {0x8706'8504'8302'8100,
                               0x9716'9514'9312'9110,
                               0xa726'a524'a322'a120,
                               0xb736'b534'b332'b130,
                               0xc746'c544'c342'c140,
                               0xd756'd554'd352'd150,
                               0xe766'e564'e362'e160,
                               0xf776'f574'f372'f170,
                               0x8f0e'8d0c'8b0a'8908,
                               0x9f1e'9d1c'9b1a'9918,
                               0xaf2e'ad2c'ab2a'a928,
                               0xbf3e'bd3c'bb3a'b938,
                               0xcf4e'cd4c'cb4a'c948,
                               0xdf5e'dd5c'db5a'd958,
                               0xef6e'ed6c'eb6a'e968,
                               0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVsssegXeXX) {
  TestVsssegXeXX<UInt8, 1, 1>(0x8208427,  // vsse8.v v8, (x1), x2, v0.t
                              4,
                              {0x5555'5581'5555'5500,
                               0x5555'5583'5555'5502,
                               0x5555'5585'5555'5504,
                               0x5555'5587'5555'5506,
                               0x5555'5589'5555'5508,
                               0x5555'558b'5555'550a,
                               0x5555'558d'5555'550c,
                               0x5555'558f'5555'550e});
  TestVsssegXeXX<UInt8, 1, 2>(0x8208427,  // vsse8.v v8, (x1), x2, v0.t
                              4,
                              {0x5555'5581'5555'5500,
                               0x5555'5583'5555'5502,
                               0x5555'5585'5555'5504,
                               0x5555'5587'5555'5506,
                               0x5555'5589'5555'5508,
                               0x5555'558b'5555'550a,
                               0x5555'558d'5555'550c,
                               0x5555'558f'5555'550e,
                               0x5555'5591'5555'5510,
                               0x5555'5593'5555'5512,
                               0x5555'5595'5555'5514,
                               0x5555'5597'5555'5516,
                               0x5555'5599'5555'5518,
                               0x5555'559b'5555'551a,
                               0x5555'559d'5555'551c,
                               0x5555'559f'5555'551e});
  TestVsssegXeXX<UInt8, 1, 4>(
      0x8208427,  // vlse8.v v8, (x1), x2, v0.t
      4,
      {0x5555'5581'5555'5500, 0x5555'5583'5555'5502, 0x5555'5585'5555'5504, 0x5555'5587'5555'5506,
       0x5555'5589'5555'5508, 0x5555'558b'5555'550a, 0x5555'558d'5555'550c, 0x5555'558f'5555'550e,
       0x5555'5591'5555'5510, 0x5555'5593'5555'5512, 0x5555'5595'5555'5514, 0x5555'5597'5555'5516,
       0x5555'5599'5555'5518, 0x5555'559b'5555'551a, 0x5555'559d'5555'551c, 0x5555'559f'5555'551e,
       0x5555'55a1'5555'5520, 0x5555'55a3'5555'5522, 0x5555'55a5'5555'5524, 0x5555'55a7'5555'5526,
       0x5555'55a9'5555'5528, 0x5555'55ab'5555'552a, 0x5555'55ad'5555'552c, 0x5555'55af'5555'552e,
       0x5555'55b1'5555'5530, 0x5555'55b3'5555'5532, 0x5555'55b5'5555'5534, 0x5555'55b7'5555'5536,
       0x5555'55b9'5555'5538, 0x5555'55bb'5555'553a, 0x5555'55bd'5555'553c, 0x5555'55bf'5555'553e});
  TestVsssegXeXX<UInt8, 1, 8>(
      0x8208427,  // vsse8.v v8, (x1), x2, v0.t
      2,
      {0x5583'5502'5581'5500, 0x5587'5506'5585'5504, 0x558b'550a'5589'5508, 0x558f'550e'558d'550c,
       0x5593'5512'5591'5510, 0x5597'5516'5595'5514, 0x559b'551a'5599'5518, 0x559f'551e'559d'551c,
       0x55a3'5522'55a1'5520, 0x55a7'5526'55a5'5524, 0x55ab'552a'55a9'5528, 0x55af'552e'55ad'552c,
       0x55b3'5532'55b1'5530, 0x55b7'5536'55b5'5534, 0x55bb'553a'55b9'5538, 0x55bf'553e'55bd'553c,
       0x55c3'5542'55c1'5540, 0x55c7'5546'55c5'5544, 0x55cb'554a'55c9'5548, 0x55cf'554e'55cd'554c,
       0x55d3'5552'55d1'5550, 0x55d7'5556'55d5'5554, 0x55db'555a'55d9'5558, 0x55df'555e'55dd'555c,
       0x55e3'5562'55e1'5560, 0x55e7'5566'55e5'5564, 0x55eb'556a'55e9'5568, 0x55ef'556e'55ed'556c,
       0x55f3'5572'55f1'5570, 0x55f7'5576'55f5'5574, 0x55fb'557a'55f9'5578, 0x55ff'557e'55fd'557c});
  TestVsssegXeXX<UInt8, 2, 1>(0x28208427,  // vssseg2e8.v v8, (x1), x2, v0.t
                              4,
                              {0x5555'9181'5555'1000,
                               0x5555'9383'5555'1202,
                               0x5555'9585'5555'1404,
                               0x5555'9787'5555'1606,
                               0x5555'9989'5555'1808,
                               0x5555'9b8b'5555'1a0a,
                               0x5555'9d8d'5555'1c0c,
                               0x5555'9f8f'5555'1e0e});
  TestVsssegXeXX<UInt8, 2, 2>(0x28208427,  // vssseg2e8.v v8, (x1), x2, v0.t
                              4,
                              {0x5555'a181'5555'2000,
                               0x5555'a383'5555'2202,
                               0x5555'a585'5555'2404,
                               0x5555'a787'5555'2606,
                               0x5555'a989'5555'2808,
                               0x5555'ab8b'5555'2a0a,
                               0x5555'ad8d'5555'2c0c,
                               0x5555'af8f'5555'2e0e,
                               0x5555'b191'5555'3010,
                               0x5555'b393'5555'3212,
                               0x5555'b595'5555'3414,
                               0x5555'b797'5555'3616,
                               0x5555'b999'5555'3818,
                               0x5555'bb9b'5555'3a1a,
                               0x5555'bd9d'5555'3c1c,
                               0x5555'bf9f'5555'3e1e});
  TestVsssegXeXX<UInt8, 2, 4>(
      0x28208427,  // vssseg4e8.v v8, (x1), x2, v0.t
      4,
      {0x5555'c181'5555'4000, 0x5555'c383'5555'4202, 0x5555'c585'5555'4404, 0x5555'c787'5555'4606,
       0x5555'c989'5555'4808, 0x5555'cb8b'5555'4a0a, 0x5555'cd8d'5555'4c0c, 0x5555'cf8f'5555'4e0e,
       0x5555'd191'5555'5010, 0x5555'd393'5555'5212, 0x5555'd595'5555'5414, 0x5555'd797'5555'5616,
       0x5555'd999'5555'5818, 0x5555'db9b'5555'5a1a, 0x5555'dd9d'5555'5c1c, 0x5555'df9f'5555'5e1e,
       0x5555'e1a1'5555'6020, 0x5555'e3a3'5555'6222, 0x5555'e5a5'5555'6424, 0x5555'e7a7'5555'6626,
       0x5555'e9a9'5555'6828, 0x5555'ebab'5555'6a2a, 0x5555'edad'5555'6c2c, 0x5555'efaf'5555'6e2e,
       0x5555'f1b1'5555'7030, 0x5555'f3b3'5555'7232, 0x5555'f5b5'5555'7434, 0x5555'f7b7'5555'7636,
       0x5555'f9b9'5555'7838, 0x5555'fbbb'5555'7a3a, 0x5555'fdbd'5555'7c3c, 0x5555'ffbf'5555'7e3e});
  TestVsssegXeXX<UInt8, 3, 1>(0x48208427,  // vssseg3e8.v v8, (x1), x2, v0.t
                              4,
                              {0x55a1'9181'5520'1000,
                               0x55a3'9383'5522'1202,
                               0x55a5'9585'5524'1404,
                               0x55a7'9787'5526'1606,
                               0x55a9'9989'5528'1808,
                               0x55ab'9b8b'552a'1a0a,
                               0x55ad'9d8d'552c'1c0c,
                               0x55af'9f8f'552e'1e0e});
  TestVsssegXeXX<UInt8, 3, 2>(0x48208427,  // vssseg3e8.v v8, (x1), x2, v0.t
                              4,
                              {0x55c1'a181'5540'2000,
                               0x55c3'a383'5542'2202,
                               0x55c5'a585'5544'2404,
                               0x55c7'a787'5546'2606,
                               0x55c9'a989'5548'2808,
                               0x55cb'ab8b'554a'2a0a,
                               0x55cd'ad8d'554c'2c0c,
                               0x55cf'af8f'554e'2e0e,
                               0x55d1'b191'5550'3010,
                               0x55d3'b393'5552'3212,
                               0x55d5'b595'5554'3414,
                               0x55d7'b797'5556'3616,
                               0x55d9'b999'5558'3818,
                               0x55db'bb9b'555a'3a1a,
                               0x55dd'bd9d'555c'3c1c,
                               0x55df'bf9f'555e'3e1e});
  TestVsssegXeXX<UInt8, 4, 1>(0x68208427,  // vssseg4e8.v v8, (x1), x2, v0.t
                              4,
                              {0xb1a1'9181'3020'1000,
                               0xb3a3'9383'3222'1202,
                               0xb5a5'9585'3424'1404,
                               0xb7a7'9787'3626'1606,
                               0xb9a9'9989'3828'1808,
                               0xbbab'9b8b'3a2a'1a0a,
                               0xbdad'9d8d'3c2c'1c0c,
                               0xbfaf'9f8f'3e2e'1e0e});
  TestVsssegXeXX<UInt8, 4, 2>(0x68208427,  // vssseg4e8.v v8, (x1), x2, v0.t
                              4,
                              {0xe1c1'a181'6040'2000,
                               0xe3c3'a383'6242'2202,
                               0xe5c5'a585'6444'2404,
                               0xe7c7'a787'6646'2606,
                               0xe9c9'a989'6848'2808,
                               0xebcb'ab8b'6a4a'2a0a,
                               0xedcd'ad8d'6c4c'2c0c,
                               0xefcf'af8f'6e4e'2e0e,
                               0xf1d1'b191'7050'3010,
                               0xf3d3'b393'7252'3212,
                               0xf5d5'b595'7454'3414,
                               0xf7d7'b797'7656'3616,
                               0xf9d9'b999'7858'3818,
                               0xfbdb'bb9b'7a5a'3a1a,
                               0xfddd'bd9d'7c5c'3c1c,
                               0xffdf'bf9f'7e5e'3e1e});
  TestVsssegXeXX<UInt8, 5, 1>(0x88208427,  // vssseg5e8.v v8, (x1), x2, v0.t
                              8,
                              {0x5555'5540'3020'1000,
                               0x5555'55c1'b1a1'9181,
                               0x5555'5542'3222'1202,
                               0x5555'55c3'b3a3'9383,
                               0x5555'5544'3424'1404,
                               0x5555'55c5'b5a5'9585,
                               0x5555'5546'3626'1606,
                               0x5555'55c7'b7a7'9787,
                               0x5555'5548'3828'1808,
                               0x5555'55c9'b9a9'9989,
                               0x5555'554a'3a2a'1a0a,
                               0x5555'55cb'bbab'9b8b,
                               0x5555'554c'3c2c'1c0c,
                               0x5555'55cd'bdad'9d8d,
                               0x5555'554e'3e2e'1e0e,
                               0x5555'55cf'bfaf'9f8f});
  TestVsssegXeXX<UInt8, 6, 1>(0xa8208427,  // vssseg6e8.v v8, (x1), x2, v0.t
                              8,
                              {0x5555'5040'3020'1000,
                               0x5555'd1c1'b1a1'9181,
                               0x5555'5242'3222'1202,
                               0x5555'd3c3'b3a3'9383,
                               0x5555'5444'3424'1404,
                               0x5555'd5c5'b5a5'9585,
                               0x5555'5646'3626'1606,
                               0x5555'd7c7'b7a7'9787,
                               0x5555'5848'3828'1808,
                               0x5555'd9c9'b9a9'9989,
                               0x5555'5a4a'3a2a'1a0a,
                               0x5555'dbcb'bbab'9b8b,
                               0x5555'5c4c'3c2c'1c0c,
                               0x5555'ddcd'bdad'9d8d,
                               0x5555'5e4e'3e2e'1e0e,
                               0x5555'dfcf'bfaf'9f8f});
  TestVsssegXeXX<UInt8, 7, 1>(0xc8208427,  // vssseg7e8.v v8, (x1), x2, v0.t
                              8,
                              {0x5560'5040'3020'1000,
                               0x55e1'd1c1'b1a1'9181,
                               0x5562'5242'3222'1202,
                               0x55e3'd3c3'b3a3'9383,
                               0x5564'5444'3424'1404,
                               0x55e5'd5c5'b5a5'9585,
                               0x5566'5646'3626'1606,
                               0x55e7'd7c7'b7a7'9787,
                               0x5568'5848'3828'1808,
                               0x55e9'd9c9'b9a9'9989,
                               0x556a'5a4a'3a2a'1a0a,
                               0x55eb'dbcb'bbab'9b8b,
                               0x556c'5c4c'3c2c'1c0c,
                               0x55ed'ddcd'bdad'9d8d,
                               0x556e'5e4e'3e2e'1e0e,
                               0x55ef'dfcf'bfaf'9f8f});
  TestVsssegXeXX<UInt8, 8, 1>(0xe8208427,  // vssseg8e8.v v8, (x1), x2, v0.t
                              8,
                              {0x7060'5040'3020'1000,
                               0xf1e1'd1c1'b1a1'9181,
                               0x7262'5242'3222'1202,
                               0xf3e3'd3c3'b3a3'9383,
                               0x7464'5444'3424'1404,
                               0xf5e5'd5c5'b5a5'9585,
                               0x7666'5646'3626'1606,
                               0xf7e7'd7c7'b7a7'9787,
                               0x7868'5848'3828'1808,
                               0xf9e9'd9c9'b9a9'9989,
                               0x7a6a'5a4a'3a2a'1a0a,
                               0xfbeb'dbcb'bbab'9b8b,
                               0x7c6c'5c4c'3c2c'1c0c,
                               0xfded'ddcd'bdad'9d8d,
                               0x7e6e'5e4e'3e2e'1e0e,
                               0xffef'dfcf'bfaf'9f8f});
  TestVsssegXeXX<UInt16, 1, 1>(0x820d427,  // vsse16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'5555'5555'8100,
                                0x5555'5555'5555'8302,
                                0x5555'5555'5555'8504,
                                0x5555'5555'5555'8706,
                                0x5555'5555'5555'8908,
                                0x5555'5555'5555'8b0a,
                                0x5555'5555'5555'8d0c,
                                0x5555'5555'5555'8f0e});

  TestVsssegXeXX<UInt16, 1, 2>(0x820d427,  // vsse16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'5555'5555'8100,
                                0x5555'5555'5555'8302,
                                0x5555'5555'5555'8504,
                                0x5555'5555'5555'8706,
                                0x5555'5555'5555'8908,
                                0x5555'5555'5555'8b0a,
                                0x5555'5555'5555'8d0c,
                                0x5555'5555'5555'8f0e,
                                0x5555'5555'5555'9110,
                                0x5555'5555'5555'9312,
                                0x5555'5555'5555'9514,
                                0x5555'5555'5555'9716,
                                0x5555'5555'5555'9918,
                                0x5555'5555'5555'9b1a,
                                0x5555'5555'5555'9d1c,
                                0x5555'5555'5555'9f1e});
  TestVsssegXeXX<UInt16, 1, 4>(
      0x820d427,  // vsse16.v v8, (x1), x2, v0.t
      8,
      {0x5555'5555'5555'8100, 0x5555'5555'5555'8302, 0x5555'5555'5555'8504, 0x5555'5555'5555'8706,
       0x5555'5555'5555'8908, 0x5555'5555'5555'8b0a, 0x5555'5555'5555'8d0c, 0x5555'5555'5555'8f0e,
       0x5555'5555'5555'9110, 0x5555'5555'5555'9312, 0x5555'5555'5555'9514, 0x5555'5555'5555'9716,
       0x5555'5555'5555'9918, 0x5555'5555'5555'9b1a, 0x5555'5555'5555'9d1c, 0x5555'5555'5555'9f1e,
       0x5555'5555'5555'a120, 0x5555'5555'5555'a322, 0x5555'5555'5555'a524, 0x5555'5555'5555'a726,
       0x5555'5555'5555'a928, 0x5555'5555'5555'ab2a, 0x5555'5555'5555'ad2c, 0x5555'5555'5555'af2e,
       0x5555'5555'5555'b130, 0x5555'5555'5555'b332, 0x5555'5555'5555'b534, 0x5555'5555'5555'b736,
       0x5555'5555'5555'b938, 0x5555'5555'5555'bb3a, 0x5555'5555'5555'bd3c, 0x5555'5555'5555'bf3e});
  TestVsssegXeXX<UInt16, 1, 8>(
      0x820d427,  // vsse16.v v8, (x1), x2, v0.t
      4,
      {0x5555'8302'5555'8100, 0x5555'8706'5555'8504, 0x5555'8b0a'5555'8908, 0x5555'8f0e'5555'8d0c,
       0x5555'9312'5555'9110, 0x5555'9716'5555'9514, 0x5555'9b1a'5555'9918, 0x5555'9f1e'5555'9d1c,
       0x5555'a322'5555'a120, 0x5555'a726'5555'a524, 0x5555'ab2a'5555'a928, 0x5555'af2e'5555'ad2c,
       0x5555'b332'5555'b130, 0x5555'b736'5555'b534, 0x5555'bb3a'5555'b938, 0x5555'bf3e'5555'bd3c,
       0x5555'c342'5555'c140, 0x5555'c746'5555'c544, 0x5555'cb4a'5555'c948, 0x5555'cf4e'5555'cd4c,
       0x5555'd352'5555'd150, 0x5555'd756'5555'd554, 0x5555'db5a'5555'd958, 0x5555'df5e'5555'dd5c,
       0x5555'e362'5555'e160, 0x5555'e766'5555'e564, 0x5555'eb6a'5555'e968, 0x5555'ef6e'5555'ed6c,
       0x5555'f372'5555'f170, 0x5555'f776'5555'f574, 0x5555'fb7a'5555'f978, 0x5555'ff7e'5555'fd7c});
  TestVsssegXeXX<UInt16, 2, 1>(0x2820d427,  // vssseg2e16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'5555'9110'8100,
                                0x5555'5555'9312'8302,
                                0x5555'5555'9514'8504,
                                0x5555'5555'9716'8706,
                                0x5555'5555'9918'8908,
                                0x5555'5555'9b1a'8b0a,
                                0x5555'5555'9d1c'8d0c,
                                0x5555'5555'9f1e'8f0e});
  TestVsssegXeXX<UInt16, 2, 2>(0x2820d427,  // vssseg2e16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'5555'a120'8100,
                                0x5555'5555'a322'8302,
                                0x5555'5555'a524'8504,
                                0x5555'5555'a726'8706,
                                0x5555'5555'a928'8908,
                                0x5555'5555'ab2a'8b0a,
                                0x5555'5555'ad2c'8d0c,
                                0x5555'5555'af2e'8f0e,
                                0x5555'5555'b130'9110,
                                0x5555'5555'b332'9312,
                                0x5555'5555'b534'9514,
                                0x5555'5555'b736'9716,
                                0x5555'5555'b938'9918,
                                0x5555'5555'bb3a'9b1a,
                                0x5555'5555'bd3c'9d1c,
                                0x5555'5555'bf3e'9f1e});
  TestVsssegXeXX<UInt16, 2, 4>(
      0x2820d427,  // vssseg2e16.v v8, (x1), x2, v0.t
      8,
      {0x5555'5555'c140'8100, 0x5555'5555'c342'8302, 0x5555'5555'c544'8504, 0x5555'5555'c746'8706,
       0x5555'5555'c948'8908, 0x5555'5555'cb4a'8b0a, 0x5555'5555'cd4c'8d0c, 0x5555'5555'cf4e'8f0e,
       0x5555'5555'd150'9110, 0x5555'5555'd352'9312, 0x5555'5555'd554'9514, 0x5555'5555'd756'9716,
       0x5555'5555'd958'9918, 0x5555'5555'db5a'9b1a, 0x5555'5555'dd5c'9d1c, 0x5555'5555'df5e'9f1e,
       0x5555'5555'e160'a120, 0x5555'5555'e362'a322, 0x5555'5555'e564'a524, 0x5555'5555'e766'a726,
       0x5555'5555'e968'a928, 0x5555'5555'eb6a'ab2a, 0x5555'5555'ed6c'ad2c, 0x5555'5555'ef6e'af2e,
       0x5555'5555'f170'b130, 0x5555'5555'f372'b332, 0x5555'5555'f574'b534, 0x5555'5555'f776'b736,
       0x5555'5555'f978'b938, 0x5555'5555'fb7a'bb3a, 0x5555'5555'fd7c'bd3c, 0x5555'5555'ff7e'bf3e});
  TestVsssegXeXX<UInt16, 3, 1>(0x4820d427,  // vssseg3e16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'a120'9110'8100,
                                0x5555'a322'9312'8302,
                                0x5555'a524'9514'8504,
                                0x5555'a726'9716'8706,
                                0x5555'a928'9918'8908,
                                0x5555'ab2a'9b1a'8b0a,
                                0x5555'ad2c'9d1c'8d0c,
                                0x5555'af2e'9f1e'8f0e});
  TestVsssegXeXX<UInt16, 3, 2>(0x4820d427,  // vssseg3e16.v v8, (x1), x2, v0.t
                               8,
                               {0x5555'c140'a120'8100,
                                0x5555'c342'a322'8302,
                                0x5555'c544'a524'8504,
                                0x5555'c746'a726'8706,
                                0x5555'c948'a928'8908,
                                0x5555'cb4a'ab2a'8b0a,
                                0x5555'cd4c'ad2c'8d0c,
                                0x5555'cf4e'af2e'8f0e,
                                0x5555'd150'b130'9110,
                                0x5555'd352'b332'9312,
                                0x5555'd554'b534'9514,
                                0x5555'd756'b736'9716,
                                0x5555'd958'b938'9918,
                                0x5555'db5a'bb3a'9b1a,
                                0x5555'dd5c'bd3c'9d1c,
                                0x5555'df5e'bf3e'9f1e});
  TestVsssegXeXX<UInt16, 4, 1>(0x6820d427,  // vssseg4e16.v v8, (x1), x2, v0.t
                               8,
                               {0xb130'a120'9110'8100,
                                0xb332'a322'9312'8302,
                                0xb534'a524'9514'8504,
                                0xb736'a726'9716'8706,
                                0xb938'a928'9918'8908,
                                0xbb3a'ab2a'9b1a'8b0a,
                                0xbd3c'ad2c'9d1c'8d0c,
                                0xbf3e'af2e'9f1e'8f0e});
  TestVsssegXeXX<UInt16, 4, 2>(0x6820d427,  // vssseg4e16.v v8, (x1), x2, v0.t
                               8,
                               {0xe160'c140'a120'8100,
                                0xe362'c342'a322'8302,
                                0xe564'c544'a524'8504,
                                0xe766'c746'a726'8706,
                                0xe968'c948'a928'8908,
                                0xeb6a'cb4a'ab2a'8b0a,
                                0xed6c'cd4c'ad2c'8d0c,
                                0xef6e'cf4e'af2e'8f0e,
                                0xf170'd150'b130'9110,
                                0xf372'd352'b332'9312,
                                0xf574'd554'b534'9514,
                                0xf776'd756'b736'9716,
                                0xf978'd958'b938'9918,
                                0xfb7a'db5a'bb3a'9b1a,
                                0xfd7c'dd5c'bd3c'9d1c,
                                0xff7e'df5e'bf3e'9f1e});
  TestVsssegXeXX<UInt16, 5, 1>(0x8820d427,  // vssseg5e16.v v8, (x1), x2, v0.t
                               16,
                               {0xb130'a120'9110'8100,
                                0x5555'5555'5555'c140,
                                0xb332'a322'9312'8302,
                                0x5555'5555'5555'c342,
                                0xb534'a524'9514'8504,
                                0x5555'5555'5555'c544,
                                0xb736'a726'9716'8706,
                                0x5555'5555'5555'c746,
                                0xb938'a928'9918'8908,
                                0x5555'5555'5555'c948,
                                0xbb3a'ab2a'9b1a'8b0a,
                                0x5555'5555'5555'cb4a,
                                0xbd3c'ad2c'9d1c'8d0c,
                                0x5555'5555'5555'cd4c,
                                0xbf3e'af2e'9f1e'8f0e,
                                0x5555'5555'5555'cf4e});
  TestVsssegXeXX<UInt16, 6, 1>(0xa820d427,  // vssseg6e16.v v8, (x1), x2, v0.t
                               16,
                               {0xb130'a120'9110'8100,
                                0x5555'5555'd150'c140,
                                0xb332'a322'9312'8302,
                                0x5555'5555'd352'c342,
                                0xb534'a524'9514'8504,
                                0x5555'5555'd554'c544,
                                0xb736'a726'9716'8706,
                                0x5555'5555'd756'c746,
                                0xb938'a928'9918'8908,
                                0x5555'5555'd958'c948,
                                0xbb3a'ab2a'9b1a'8b0a,
                                0x5555'5555'db5a'cb4a,
                                0xbd3c'ad2c'9d1c'8d0c,
                                0x5555'5555'dd5c'cd4c,
                                0xbf3e'af2e'9f1e'8f0e,
                                0x5555'5555'df5e'cf4e});
  TestVsssegXeXX<UInt16, 7, 1>(0xc820d427,  // vssseg7e16.v v8, (x1), x2, v0.t
                               16,
                               {0xb130'a120'9110'8100,
                                0x5555'e160'd150'c140,
                                0xb332'a322'9312'8302,
                                0x5555'e362'd352'c342,
                                0xb534'a524'9514'8504,
                                0x5555'e564'd554'c544,
                                0xb736'a726'9716'8706,
                                0x5555'e766'd756'c746,
                                0xb938'a928'9918'8908,
                                0x5555'e968'd958'c948,
                                0xbb3a'ab2a'9b1a'8b0a,
                                0x5555'eb6a'db5a'cb4a,
                                0xbd3c'ad2c'9d1c'8d0c,
                                0x5555'ed6c'dd5c'cd4c,
                                0xbf3e'af2e'9f1e'8f0e,
                                0x5555'ef6e'df5e'cf4e});
  TestVsssegXeXX<UInt16, 8, 1>(0xe820d427,  // vssseg8e16.v v8, (x1), x2, v0.t
                               16,
                               {0xb130'a120'9110'8100,
                                0xf170'e160'd150'c140,
                                0xb332'a322'9312'8302,
                                0xf372'e362'd352'c342,
                                0xb534'a524'9514'8504,
                                0xf574'e564'd554'c544,
                                0xb736'a726'9716'8706,
                                0xf776'e766'd756'c746,
                                0xb938'a928'9918'8908,
                                0xf978'e968'd958'c948,
                                0xbb3a'ab2a'9b1a'8b0a,
                                0xfb7a'eb6a'db5a'cb4a,
                                0xbd3c'ad2c'9d1c'8d0c,
                                0xfd7c'ed6c'dd5c'cd4c,
                                0xbf3e'af2e'9f1e'8f0e,
                                0xff7e'ef6e'df5e'cf4e});
  TestVsssegXeXX<UInt32, 1, 1>(0x820e427,  // vsse32.v v8, (x1), x2, v0.t
                               16,
                               {0x5555'5555'8302'8100,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8706'8504,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8b0a'8908,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8f0e'8d0c});
  TestVsssegXeXX<UInt32, 1, 2>(0x820e427,  // vsse32.v v8, (x1), x2, v0.t
                               16,
                               {0x5555'5555'8302'8100,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8706'8504,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8b0a'8908,
                                0x5555'5555'5555'5555,
                                0x5555'5555'8f0e'8d0c,
                                0x5555'5555'5555'5555,
                                0x5555'5555'9312'9110,
                                0x5555'5555'5555'5555,
                                0x5555'5555'9716'9514,
                                0x5555'5555'5555'5555,
                                0x5555'5555'9b1a'9918,
                                0x5555'5555'5555'5555,
                                0x5555'5555'9f1e'9d1c});
  TestVsssegXeXX<UInt32, 1, 4>(
      0x820e427,  // vsse32.v v8, (x1), x2, v0.t
      16,
      {0x5555'5555'8302'8100, 0x5555'5555'5555'5555, 0x5555'5555'8706'8504, 0x5555'5555'5555'5555,
       0x5555'5555'8b0a'8908, 0x5555'5555'5555'5555, 0x5555'5555'8f0e'8d0c, 0x5555'5555'5555'5555,
       0x5555'5555'9312'9110, 0x5555'5555'5555'5555, 0x5555'5555'9716'9514, 0x5555'5555'5555'5555,
       0x5555'5555'9b1a'9918, 0x5555'5555'5555'5555, 0x5555'5555'9f1e'9d1c, 0x5555'5555'5555'5555,
       0x5555'5555'a322'a120, 0x5555'5555'5555'5555, 0x5555'5555'a726'a524, 0x5555'5555'5555'5555,
       0x5555'5555'ab2a'a928, 0x5555'5555'5555'5555, 0x5555'5555'af2e'ad2c, 0x5555'5555'5555'5555,
       0x5555'5555'b332'b130, 0x5555'5555'5555'5555, 0x5555'5555'b736'b534, 0x5555'5555'5555'5555,
       0x5555'5555'bb3a'b938, 0x5555'5555'5555'5555, 0x5555'5555'bf3e'bd3c});
  TestVsssegXeXX<UInt32, 1, 8>(
      0x820e427,  // vsse32.v v8, (x1), x2, v0.t
      8,
      {0x5555'5555'8302'8100, 0x5555'5555'8706'8504, 0x5555'5555'8b0a'8908, 0x5555'5555'8f0e'8d0c,
       0x5555'5555'9312'9110, 0x5555'5555'9716'9514, 0x5555'5555'9b1a'9918, 0x5555'5555'9f1e'9d1c,
       0x5555'5555'a322'a120, 0x5555'5555'a726'a524, 0x5555'5555'ab2a'a928, 0x5555'5555'af2e'ad2c,
       0x5555'5555'b332'b130, 0x5555'5555'b736'b534, 0x5555'5555'bb3a'b938, 0x5555'5555'bf3e'bd3c,
       0x5555'5555'c342'c140, 0x5555'5555'c746'c544, 0x5555'5555'cb4a'c948, 0x5555'5555'cf4e'cd4c,
       0x5555'5555'd352'd150, 0x5555'5555'd756'd554, 0x5555'5555'db5a'd958, 0x5555'5555'df5e'dd5c,
       0x5555'5555'e362'e160, 0x5555'5555'e766'e564, 0x5555'5555'eb6a'e968, 0x5555'5555'ef6e'ed6c,
       0x5555'5555'f372'f170, 0x5555'5555'f776'f574, 0x5555'5555'fb7a'f978, 0x5555'5555'ff7e'fd7c});
  TestVsssegXeXX<UInt32, 2, 1>(0x2820e427,  // vssseg2e32.v v8, (x1), x2, v0.t
                               16,
                               {0x9312'9110'8302'8100,
                                0x5555'5555'5555'5555,
                                0x9716'9514'8706'8504,
                                0x5555'5555'5555'5555,
                                0x9b1a'9918'8b0a'8908,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'8f0e'8d0c});
  TestVsssegXeXX<UInt32, 2, 2>(0x2820e427,  // vssseg2e32.v v8, (x1), x2, v0.t
                               16,
                               {0xa322'a120'8302'8100,
                                0x5555'5555'5555'5555,
                                0xa726'a524'8706'8504,
                                0x5555'5555'5555'5555,
                                0xab2a'a928'8b0a'8908,
                                0x5555'5555'5555'5555,
                                0xaf2e'ad2c'8f0e'8d0c,
                                0x5555'5555'5555'5555,
                                0xb332'b130'9312'9110,
                                0x5555'5555'5555'5555,
                                0xb736'b534'9716'9514,
                                0x5555'5555'5555'5555,
                                0xbb3a'b938'9b1a'9918,
                                0x5555'5555'5555'5555,
                                0xbf3e'bd3c'9f1e'9d1c});
  TestVsssegXeXX<UInt32, 2, 4>(
      0x2820e427,  // vssseg2e32.v v8, (x1), x2, v0.t
      16,
      {0xc342'c140'8302'8100, 0x5555'5555'5555'5555, 0xc746'c544'8706'8504, 0x5555'5555'5555'5555,
       0xcb4a'c948'8b0a'8908, 0x5555'5555'5555'5555, 0xcf4e'cd4c'8f0e'8d0c, 0x5555'5555'5555'5555,
       0xd352'd150'9312'9110, 0x5555'5555'5555'5555, 0xd756'd554'9716'9514, 0x5555'5555'5555'5555,
       0xdb5a'd958'9b1a'9918, 0x5555'5555'5555'5555, 0xdf5e'dd5c'9f1e'9d1c, 0x5555'5555'5555'5555,
       0xe362'e160'a322'a120, 0x5555'5555'5555'5555, 0xe766'e564'a726'a524, 0x5555'5555'5555'5555,
       0xeb6a'e968'ab2a'a928, 0x5555'5555'5555'5555, 0xef6e'ed6c'af2e'ad2c, 0x5555'5555'5555'5555,
       0xf372'f170'b332'b130, 0x5555'5555'5555'5555, 0xf776'f574'b736'b534, 0x5555'5555'5555'5555,
       0xfb7a'f978'bb3a'b938, 0x5555'5555'5555'5555, 0xff7e'fd7c'bf3e'bd3c});
  TestVsssegXeXX<UInt32, 3, 1>(0x4820e427,  // vssseg3e32.v v8, (x1), x2, v0.t
                               16,
                               {0x9312'9110'8302'8100,
                                0x5555'5555'a322'a120,
                                0x9716'9514'8706'8504,
                                0x5555'5555'a726'a524,
                                0x9b1a'9918'8b0a'8908,
                                0x5555'5555'ab2a'a928,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0x5555'5555'af2e'ad2c});
  TestVsssegXeXX<UInt32, 3, 2>(0x4820e427,  // vssseg3e32.v v8, (x1), x2, v0.t
                               16,
                               {0xa322'a120'8302'8100,
                                0x5555'5555'c342'c140,
                                0xa726'a524'8706'8504,
                                0x5555'5555'c746'c544,
                                0xab2a'a928'8b0a'8908,
                                0x5555'5555'cb4a'c948,
                                0xaf2e'ad2c'8f0e'8d0c,
                                0x5555'5555'cf4e'cd4c,
                                0xb332'b130'9312'9110,
                                0x5555'5555'd352'd150,
                                0xb736'b534'9716'9514,
                                0x5555'5555'd756'd554,
                                0xbb3a'b938'9b1a'9918,
                                0x5555'5555'db5a'd958,
                                0xbf3e'bd3c'9f1e'9d1c,
                                0x5555'5555'df5e'dd5c});
  TestVsssegXeXX<UInt32, 4, 1>(0x6820e427,  // vssseg4e32.v v8, (x1), x2, v0.t
                               16,
                               {0x9312'9110'8302'8100,
                                0xb332'b130'a322'a120,
                                0x9716'9514'8706'8504,
                                0xb736'b534'a726'a524,
                                0x9b1a'9918'8b0a'8908,
                                0xbb3a'b938'ab2a'a928,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0xbf3e'bd3c'af2e'ad2c});
  TestVsssegXeXX<UInt32, 4, 2>(0x6820e427,  // vssseg4e32.v v8, (x1), x2, v0.t
                               16,
                               {0xa322'a120'8302'8100,
                                0xe362'e160'c342'c140,
                                0xa726'a524'8706'8504,
                                0xe766'e564'c746'c544,
                                0xab2a'a928'8b0a'8908,
                                0xeb6a'e968'cb4a'c948,
                                0xaf2e'ad2c'8f0e'8d0c,
                                0xef6e'ed6c'cf4e'cd4c,
                                0xb332'b130'9312'9110,
                                0xf372'f170'd352'd150,
                                0xb736'b534'9716'9514,
                                0xf776'f574'd756'd554,
                                0xbb3a'b938'9b1a'9918,
                                0xfb7a'f978'db5a'd958,
                                0xbf3e'bd3c'9f1e'9d1c,
                                0xff7e'fd7c'df5e'dd5c});
  TestVsssegXeXX<UInt32, 5, 1>(0x8820e427,  // vssseg5e32.v v8, (x1), x2, v0.t
                               32,
                               {0x9312'9110'8302'8100,
                                0xb332'b130'a322'a120,
                                0x5555'5555'c342'c140,
                                0x5555'5555'5555'5555,
                                0x9716'9514'8706'8504,
                                0xb736'b534'a726'a524,
                                0x5555'5555'c746'c544,
                                0x5555'5555'5555'5555,
                                0x9b1a'9918'8b0a'8908,
                                0xbb3a'b938'ab2a'a928,
                                0x5555'5555'cb4a'c948,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0xbf3e'bd3c'af2e'ad2c,
                                0x5555'5555'cf4e'cd4c});
  TestVsssegXeXX<UInt32, 6, 1>(0xa820e427,  // vssseg6e32.v v8, (x1), x2, v0.t
                               32,
                               {0x9312'9110'8302'8100,
                                0xb332'b130'a322'a120,
                                0xd352'd150'c342'c140,
                                0x5555'5555'5555'5555,
                                0x9716'9514'8706'8504,
                                0xb736'b534'a726'a524,
                                0xd756'd554'c746'c544,
                                0x5555'5555'5555'5555,
                                0x9b1a'9918'8b0a'8908,
                                0xbb3a'b938'ab2a'a928,
                                0xdb5a'd958'cb4a'c948,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0xbf3e'bd3c'af2e'ad2c,
                                0xdf5e'dd5c'cf4e'cd4c});
  TestVsssegXeXX<UInt32, 7, 1>(0xc820e427,  // vssseg7e32.v v8, (x1), x2, v0.t
                               32,
                               {0x9312'9110'8302'8100,
                                0xb332'b130'a322'a120,
                                0xd352'd150'c342'c140,
                                0x5555'5555'e362'e160,
                                0x9716'9514'8706'8504,
                                0xb736'b534'a726'a524,
                                0xd756'd554'c746'c544,
                                0x5555'5555'e766'e564,
                                0x9b1a'9918'8b0a'8908,
                                0xbb3a'b938'ab2a'a928,
                                0xdb5a'd958'cb4a'c948,
                                0x5555'5555'eb6a'e968,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0xbf3e'bd3c'af2e'ad2c,
                                0xdf5e'dd5c'cf4e'cd4c,
                                0x5555'5555'ef6e'ed6c});
  TestVsssegXeXX<UInt32, 8, 1>(0xe820e427,  // vssseg8e32.v v8, (x1), x2, v0.t
                               32,
                               {0x9312'9110'8302'8100,
                                0xb332'b130'a322'a120,
                                0xd352'd150'c342'c140,
                                0xf372'f170'e362'e160,
                                0x9716'9514'8706'8504,
                                0xb736'b534'a726'a524,
                                0xd756'd554'c746'c544,
                                0xf776'f574'e766'e564,
                                0x9b1a'9918'8b0a'8908,
                                0xbb3a'b938'ab2a'a928,
                                0xdb5a'd958'cb4a'c948,
                                0xfb7a'f978'eb6a'e968,
                                0x9f1e'9d1c'8f0e'8d0c,
                                0xbf3e'bd3c'af2e'ad2c,
                                0xdf5e'dd5c'cf4e'cd4c,
                                0xff7e'fd7c'ef6e'ed6c});
  TestVsssegXeXX<UInt64, 1, 1>(0x820f427,  // vsse64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908});
  TestVsssegXeXX<UInt64, 1, 2>(0x820f427,  // vsse64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x9716'9514'9312'9110,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'9b1a'9918});
  TestVsssegXeXX<UInt64, 1, 4>(
      0x820f427,  // vsse64.v v8, (x1), x2, v0.t
      32,
      {0x8706'8504'8302'8100, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0x8f0e'8d0c'8b0a'8908, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0x9716'9514'9312'9110, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0x9f1e'9d1c'9b1a'9918, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0xa726'a524'a322'a120, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0xaf2e'ad2c'ab2a'a928, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0xb736'b534'b332'b130, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555, 0x5555'5555'5555'5555,
       0xbf3e'bd3c'bb3a'b938});
  TestVsssegXeXX<UInt64, 1, 8>(
      0x820f427,  // vsse64.v v8, (x1), x2, v0.t
      16,
      {0x8706'8504'8302'8100, 0x5555'5555'5555'5555, 0x8f0e'8d0c'8b0a'8908, 0x5555'5555'5555'5555,
       0x9716'9514'9312'9110, 0x5555'5555'5555'5555, 0x9f1e'9d1c'9b1a'9918, 0x5555'5555'5555'5555,
       0xa726'a524'a322'a120, 0x5555'5555'5555'5555, 0xaf2e'ad2c'ab2a'a928, 0x5555'5555'5555'5555,
       0xb736'b534'b332'b130, 0x5555'5555'5555'5555, 0xbf3e'bd3c'bb3a'b938, 0x5555'5555'5555'5555,
       0xc746'c544'c342'c140, 0x5555'5555'5555'5555, 0xcf4e'cd4c'cb4a'c948, 0x5555'5555'5555'5555,
       0xd756'd554'd352'd150, 0x5555'5555'5555'5555, 0xdf5e'dd5c'db5a'd958, 0x5555'5555'5555'5555,
       0xe766'e564'e362'e160, 0x5555'5555'5555'5555, 0xef6e'ed6c'eb6a'e968, 0x5555'5555'5555'5555,
       0xf776'f574'f372'f170, 0x5555'5555'5555'5555, 0xff7e'fd7c'fb7a'f978});
  TestVsssegXeXX<UInt64, 2, 1>(0x2820f427,  // vssseg2e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918});
  TestVsssegXeXX<UInt64, 2, 2>(0x2820f427,  // vssseg2e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0xa726'a524'a322'a120,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0xaf2e'ad2c'ab2a'a928,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x9716'9514'9312'9110,
                                0xb736'b534'b332'b130,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'9b1a'9918,
                                0xbf3e'bd3c'bb3a'b938});
  TestVsssegXeXX<UInt64, 2, 4>(0x2820f427,  // vssseg2e64.v v8, (x1), x2, v0.t
                               16,
                               {0x8706'8504'8302'8100,
                                0xc746'c544'c342'c140,
                                0x8f0e'8d0c'8b0a'8908,
                                0xcf4e'cd4c'cb4a'c948,
                                0x9716'9514'9312'9110,
                                0xd756'd554'd352'd150,
                                0x9f1e'9d1c'9b1a'9918,
                                0xdf5e'dd5c'db5a'd958,
                                0xa726'a524'a322'a120,
                                0xe766'e564'e362'e160,
                                0xaf2e'ad2c'ab2a'a928,
                                0xef6e'ed6c'eb6a'e968,
                                0xb736'b534'b332'b130,
                                0xf776'f574'f372'f170,
                                0xbf3e'bd3c'bb3a'b938,
                                0xff7e'fd7c'fb7a'f978});
  TestVsssegXeXX<UInt64, 3, 1>(0x4820f427,  // vssseg3e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928});
  TestVsssegXeXX<UInt64, 3, 2>(0x4820f427,  // vssseg3e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0xa726'a524'a322'a120,
                                0xc746'c544'c342'c140,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0xaf2e'ad2c'ab2a'a928,
                                0xcf4e'cd4c'cb4a'c948,
                                0x5555'5555'5555'5555,
                                0x9716'9514'9312'9110,
                                0xb736'b534'b332'b130,
                                0xd756'd554'd352'd150,
                                0x5555'5555'5555'5555,
                                0x9f1e'9d1c'9b1a'9918,
                                0xbf3e'bd3c'bb3a'b938,
                                0xdf5e'dd5c'db5a'd958});
  TestVsssegXeXX<UInt64, 4, 1>(0x6820f427,  // vssseg4e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0xb736'b534'b332'b130,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928,
                                0xbf3e'bd3c'bb3a'b938});
  TestVsssegXeXX<UInt64, 4, 2>(0x6820f427,  // vssseg4e64.v v8, (x1), x2, v0.t
                               32,
                               {0x8706'8504'8302'8100,
                                0xa726'a524'a322'a120,
                                0xc746'c544'c342'c140,
                                0xe766'e564'e362'e160,
                                0x8f0e'8d0c'8b0a'8908,
                                0xaf2e'ad2c'ab2a'a928,
                                0xcf4e'cd4c'cb4a'c948,
                                0xef6e'ed6c'eb6a'e968,
                                0x9716'9514'9312'9110,
                                0xb736'b534'b332'b130,
                                0xd756'd554'd352'd150,
                                0xf776'f574'f372'f170,
                                0x9f1e'9d1c'9b1a'9918,
                                0xbf3e'bd3c'bb3a'b938,
                                0xdf5e'dd5c'db5a'd958,
                                0xff7e'fd7c'fb7a'f978});
  TestVsssegXeXX<UInt64, 5, 1>(0x8820f427,  // vssseg5e64.v v8, (x1), x2, v0.t
                               64,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0xb736'b534'b332'b130,
                                0xc746'c544'c342'c140,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928,
                                0xbf3e'bd3c'bb3a'b938,
                                0xcf4e'cd4c'cb4a'c948});
  TestVsssegXeXX<UInt64, 6, 1>(0xa820f427,  // vssseg6e64.v v8, (x1), x2, v0.t
                               64,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0xb736'b534'b332'b130,
                                0xc746'c544'c342'c140,
                                0xd756'd554'd352'd150,
                                0x5555'5555'5555'5555,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928,
                                0xbf3e'bd3c'bb3a'b938,
                                0xcf4e'cd4c'cb4a'c948,
                                0xdf5e'dd5c'db5a'd958});
  TestVsssegXeXX<UInt64, 7, 1>(0xc820f427,  // vssseg7e64.v v8, (x1), x2, v0.t
                               64,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0xb736'b534'b332'b130,
                                0xc746'c544'c342'c140,
                                0xd756'd554'd352'd150,
                                0xe766'e564'e362'e160,
                                0x5555'5555'5555'5555,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928,
                                0xbf3e'bd3c'bb3a'b938,
                                0xcf4e'cd4c'cb4a'c948,
                                0xdf5e'dd5c'db5a'd958,
                                0xef6e'ed6c'eb6a'e968});
  TestVsssegXeXX<UInt64, 8, 1>(0xe820f427,  // vssseg8e64.v v8, (x1), x2, v0.t
                               64,
                               {0x8706'8504'8302'8100,
                                0x9716'9514'9312'9110,
                                0xa726'a524'a322'a120,
                                0xb736'b534'b332'b130,
                                0xc746'c544'c342'c140,
                                0xd756'd554'd352'd150,
                                0xe766'e564'e362'e160,
                                0xf776'f574'f372'f170,
                                0x8f0e'8d0c'8b0a'8908,
                                0x9f1e'9d1c'9b1a'9918,
                                0xaf2e'ad2c'ab2a'a928,
                                0xbf3e'bd3c'bb3a'b938,
                                0xcf4e'cd4c'cb4a'c948,
                                0xdf5e'dd5c'db5a'd958,
                                0xef6e'ed6c'eb6a'e968,
                                0xff7e'fd7c'fb7a'f978});
}

TEST_F(Riscv64InterpreterTest, TestVsm) {
  TestVsm(0x2b08427,  // vsm.v v8, (x1)
          {0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143});
}

TEST_F(Riscv64InterpreterTest, TestVectorMaskInstructions) {
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x630c2457,  // vmandn.mm v8, v16, v24
                            {0x8102'8504'8102'8100, 0x8102'8504'890a'8908});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x670c2457,  // vmand.mm v8, v16, v24
                            {0x0604'0000'0200'0000, 0x0e0c'0808'0200'0000});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x6b0c2457,  // vmor.mm v8, v16, v24
                            {0x8f0e'8f0d'8706'8300, 0x9f1e'9f1c'9f1e'9b19});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x6f0c2457,  // vmxor.mm v8, v16, v24
                            {0x890a'8f0d'8506'8300, 0x9112'9714'9d1e'9b19});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x730c2457,  // vmorn.mm v8, v16, v24
                            {0xf7f7'f5f6'fbfb'fdff, 0xefef'edef'ebeb'edee});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x770c2457,  // vmnand.mm v8, v16, v24
                            {0xf9fb'ffff'fdff'ffff, 0xf1f3'f7f7'fdff'ffff});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x7b0c2457,  // vmnor.mm v8, v16, v24
                            {0x70f1'70f2'78f9'7cff, 0x60e1'60e3'60e1'64e6});
  TestVectorMaskInstruction(128,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x7f0c2457,  // vmxnor.mm v8, v16, v24
                            {0x76f5'70f2'7af9'7cff, 0x6eed'68eb'62e1'64e6});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x5300a457,  // vmsbf.m v8, v16
                            {0x0000'0000'0000'00ff, 0x0000'0000'0000'0000});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x53012457,  // vmsof.m v8, v16
                            {0x0000'0000'0000'0100, 0x0000'0000'0000'0000});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x5301a457,  // vmsif.m v8, v16
                            {0x0000'0000'0000'01ff, 0x0000'0000'0000'0000});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x5100a457,  // vmsbf.m v8, v16, v0.t
                            {0xd5ad'd6b5'adff'ffff, 0x6af7'57bb'deed'7bb5});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x51012457,  // vmsof.m v8, v16, v0.t
                            {0xd5ad'd6b5'af6b'b5ad, 0x6af7'57bb'deed'7bb5});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kAgnostic,
                            0x5101a457,  // vmsif.m v8, v16, v0.t
                            {0xd5ad'd6b5'afff'ffff, 0x6af7'57bb'deed'7bb5});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kUndisturbed,
                            0x5100a457,  // vmsbf.m v8, v16, v0.t
                            {0x5505'5415'05d5'5f57, 0x4055'5511'5445'5115});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kUndisturbed,
                            0x51012457,  // vmsof.m v8, v16, v0.t
                            {0x5505'5415'0741'1505, 0x4055'5511'5445'5115});
  TestVectorMaskInstruction(0,
                            intrinsics::InactiveProcessing::kUndisturbed,
                            0x5101a457,  // vmsif.m v8, v16, v0.t
                            {0x5505'5415'07d5'5f57, 0x4055'5511'5445'5115});
}

TEST_F(Riscv64InterpreterTest, TestVmfeq) {
  TestVectorMaskTargetInstruction(0x610c1457,  // Vmfeq.vv v8, v16, v24, v0.t
                                  0x0000'0007,
                                  0x0001,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x6100d457,  // Vmfeq.vf v8, v16, f1, v0.t
                                  0x0000'0040,
                                  0x0020,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmseq) {
  TestVectorMaskTargetInstruction(0x610c0457,  // Vmseq.vv v8, v16, v24, v0.t
                                  {255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  0x0000'0000'0000'00ff,
                                  0x0000'000f,
                                  0x0003,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x6100c457,  // Vmseq.vx v8, v16, x1, v0.t
                                  {0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  0x0000'0000'0f00'0000,
                                  0x0000'3000,
                                  0x0040,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x610ab457,  // Vmseq.vi  v8, v16, -0xb, v0.t
                                  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 0, 0, 0, 0},
                                  0x0000'f000'0000'0000,
                                  0x0000'0000,
                                  0x0000,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmfne) {
  TestVectorMaskTargetInstruction(0x710c1457,  // Vmfne.vv v8, v16, v24, v0.t
                                  0xffff'fff8,
                                  0xfffe,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x7100d457,  // Vmfne.vf v8, v16, f1, v0.t
                                  0xffff'ffbf,
                                  0xffdf,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsne) {
  TestVectorMaskTargetInstruction(
      0x650c0457,  // Vmsne.vv v8, v16, v24, v0.t
      {0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      0xffff'ffff'ffff'ff00,
      0xffff'fff0,
      0xfffc,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x6500c457,  // Vmsne.vx v8, v16, x1, v0.t
      {255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      0xffff'ffff'f0ff'ffff,
      0xffff'cfff,
      0xffbf,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x650ab457,  // Vmsne.vi  v8, v16, -0xb, v0.t
      {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 170, 255, 255, 255, 255},
      0xffff'0fff'ffff'ffff,
      0xffff'ffff,
      0xffff,
      kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmflt) {
  TestVectorMaskTargetInstruction(0x6d0c1457,  // Vmflt.vv v8, v16, v24, v0.t
                                  0x0000'f000,
                                  0x00c0,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x6d00d457,  // Vmflt.vf v8, v16, f1, v0.t
                                  0xff00'ff07,
                                  0xf0d1,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsltu) {
  TestVectorMaskTargetInstruction(0x690c0457,  // Vmsltu.vv v8, v16, v24, v0.t
                                  {0, 0, 0, 3, 255, 255, 0, 255, 0, 0, 0, 0, 255, 255, 255, 255},
                                  0xffff'0000'f0ff'1000,
                                  0xff00'cf00,
                                  0xf0b0,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x6900c457,  // Vmsltu.vx v8, v16, x1, v0.t
      {85, 15, 10, 11, 255, 255, 0, 255, 0, 0, 0, 0, 136, 136, 255, 255},
      0xffaa'0000'f0ff'3330,
      0xff00'cf54,
      0xf0b0,
      kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmslt) {
  TestVectorMaskTargetInstruction(0x6d0c0457,  // Vmslt.vv v8, v16, v24, v0.t
                                  {0, 0, 245, 247, 0, 32, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255},
                                  0xffff'0000'ff40'dc00,
                                  0xff00'f0a0,
                                  0xf0cc,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x6d00c457,  // Vmslt.vx v8, v16, x1, v0.t
                                  {0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 136, 136, 255, 255},
                                  0xffaa'0000'0040'0000,
                                  0xff00'0000,
                                  0xf000,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmfle) {
  TestVectorMaskTargetInstruction(0x650c1457,  // Vmfle.vv v8, v16, v24, v0.t
                                  0x0000'f007,
                                  0x00c1,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x6500d457,  // Vmfle.vf v8, v16, f1, v0.t
                                  0xff00'ff47,
                                  0xf0f1,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsleu) {
  TestVectorMaskTargetInstruction(
      0x710c0457,  // Vmsleu.vv v8, v16, v24, v0.t
      {255, 255, 0, 3, 255, 255, 0, 255, 0, 0, 0, 0, 255, 255, 255, 255},
      0xffff'0000'f0ff'10ff,
      0xff00'cf0f,
      0xf0b3,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x7100c457,  // Vmsleu.vx v8, v16, x1, v0.t
      {85, 15, 10, 11, 255, 255, 255, 255, 0, 0, 0, 0, 136, 136, 255, 255},
      0xffaa'0000'ffff'3330,
      0xff00'ff54,
      0xf0f0,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x710ab457,  // Vmsleu.vi  v8, v16, -0xb, v0.t
      {255, 15, 15, 15, 255, 255, 255, 255, 85, 0, 5, 85, 255, 255, 255, 255},
      0xffff'f30f'ffff'333f,
      0xffff'ff57,
      0xffff,
      kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsle) {
  TestVectorMaskTargetInstruction(
      0x750c0457,  // Vmsle.vv v8, v16, v24, v0.t
      {255, 255, 245, 247, 0, 32, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255},
      0xffff'0000'ff40'dcff,
      0xff00'f0af,
      0xf0cf,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x7500c457,  // Vmsle.vx v8, v16, x1, v0.t
                                  {0, 0, 0, 0, 0, 32, 255, 0, 0, 0, 0, 0, 136, 136, 255, 255},
                                  0xffaa'0000'0f40'0000,
                                  0xff00'3000,
                                  0xf040,
                                  kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x750ab457,  // Vmsle.vi  v8, v16, -0xb
                                  {170, 0, 5, 4, 0, 32, 255, 0, 85, 0, 5, 85, 255, 255, 255, 255},
                                  0xffff'f30f'0f40'000f,
                                  0xffff'3003,
                                  0xff4f,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmfgt) {
  TestVectorMaskTargetInstruction(0x7500d457,  // Vmfgt.vf v8, v16, f1, v0.t
                                  0x0000'0010,
                                  0x0000,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsgtu) {
  TestVectorMaskTargetInstruction(
      0x7900c457,  // Vmsgtu.vx v8, v16, x1, v0.t
      {170, 240, 245, 244, 0, 0, 0, 0, 255, 255, 255, 255, 119, 119, 0, 0},
      0x0055'ffff'0000'cccf,
      0x00ff'00ab,
      0x0f0f,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(0x790ab457,  // Vmsgtu.vi  v8, v16, -0xb, v0.t
                                  {0, 240, 240, 240, 0, 0, 0, 0, 170, 255, 250, 170, 0, 0, 0, 0},
                                  0x0000'0cf0'0000'ccc0,
                                  0x0000'00a8,
                                  0x0000,
                                  kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmsgt) {
  TestVectorMaskTargetInstruction(
      0x7d00c457,  // Vmsgt.vx v8, v16, x1, v0.t
      {255, 255, 255, 255, 255, 223, 0, 255, 255, 255, 255, 255, 119, 119, 0, 0},
      0x0055'ffff'f0bf'ffff,
      0x00ff'cfff,
      0x0fbf,
      kVectorComparisonSource);
  TestVectorMaskTargetInstruction(
      0x7d0ab457,  // Vmsgt.vi  v8, v16, -0xb, v0.t
      {85, 255, 250, 251, 255, 223, 0, 255, 170, 255, 250, 170, 0, 0, 0, 0},
      0x0000'0cf0'f0bf'fff0,
      0x0000'cffc,
      0x00b0,
      kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVmfge) {
  TestVectorMaskTargetInstruction(0x7d00d457,  // Vmfge.vf v8, v16, f1, v0.t
                                  0x0000'0050,
                                  0x0020,
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
      kVectorCalculationsSourceLegacy);
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
      kVectorCalculationsSourceLegacy);
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
      kVectorCalculationsSourceLegacy);
}

TEST_F(Riscv64InterpreterTest, TestVsrl) {
  TestVectorInstruction(0xa10c0457,  // Vsrl.vv v8, v16, v24, v0.t
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
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(0xa100c457,  // Vsrl.vx v8, v16, x1, v0.t
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
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(0xa101b457,  // Vsrl.vi v8, v16, 0x3, v0.t
                        {{0, 16, 0, 16, 0, 16, 0, 16, 1, 17, 1, 17, 1, 17, 1, 17},
                         {2, 18, 2, 18, 2, 18, 2, 18, 3, 19, 3, 19, 3, 19, 3, 19},
                         {4, 20, 4, 20, 4, 20, 4, 20, 5, 21, 5, 21, 5, 21, 5, 21},
                         {6, 22, 6, 22, 6, 22, 6, 22, 7, 23, 7, 23, 7, 23, 7, 23},
                         {8, 24, 8, 24, 8, 24, 8, 24, 9, 25, 9, 25, 9, 25, 9, 25},
                         {10, 26, 10, 26, 10, 26, 10, 26, 11, 27, 11, 27, 11, 27, 11, 27},
                         {12, 28, 12, 28, 12, 28, 12, 28, 13, 29, 13, 29, 13, 29, 13, 29},
                         {14, 30, 14, 30, 14, 30, 14, 30, 15, 31, 15, 31, 15, 31, 15, 31}},
                        {{0x1020, 0x1060, 0x10a0, 0x10e0, 0x1121, 0x1161, 0x11a1, 0x11e1},
                         {0x1222, 0x1262, 0x12a2, 0x12e2, 0x1323, 0x1363, 0x13a3, 0x13e3},
                         {0x1424, 0x1464, 0x14a4, 0x14e4, 0x1525, 0x1565, 0x15a5, 0x15e5},
                         {0x1626, 0x1666, 0x16a6, 0x16e6, 0x1727, 0x1767, 0x17a7, 0x17e7},
                         {0x1828, 0x1868, 0x18a8, 0x18e8, 0x1929, 0x1969, 0x19a9, 0x19e9},
                         {0x1a2a, 0x1a6a, 0x1aaa, 0x1aea, 0x1b2b, 0x1b6b, 0x1bab, 0x1beb},
                         {0x1c2c, 0x1c6c, 0x1cac, 0x1cec, 0x1d2d, 0x1d6d, 0x1dad, 0x1ded},
                         {0x1e2e, 0x1e6e, 0x1eae, 0x1eee, 0x1f2f, 0x1f6f, 0x1faf, 0x1fef}},
                        {{0x1060'5020, 0x10e0'd0a0, 0x1161'5121, 0x11e1'd1a1},
                         {0x1262'5222, 0x12e2'd2a2, 0x1363'5323, 0x13e3'd3a3},
                         {0x1464'5424, 0x14e4'd4a4, 0x1565'5525, 0x15e5'd5a5},
                         {0x1666'5626, 0x16e6'd6a6, 0x1767'5727, 0x17e7'd7a7},
                         {0x1868'5828, 0x18e8'd8a8, 0x1969'5929, 0x19e9'd9a9},
                         {0x1a6a'5a2a, 0x1aea'daaa, 0x1b6b'5b2b, 0x1beb'dbab},
                         {0x1c6c'5c2c, 0x1cec'dcac, 0x1d6d'5d2d, 0x1ded'ddad},
                         {0x1e6e'5e2e, 0x1eee'deae, 0x1f6f'5f2f, 0x1fef'dfaf}},
                        {{0x10e0'd0a0'9060'5020, 0x11e1'd1a1'9161'5121},
                         {0x12e2'd2a2'9262'5222, 0x13e3'd3a3'9363'5323},
                         {0x14e4'd4a4'9464'5424, 0x15e5'd5a5'9565'5525},
                         {0x16e6'd6a6'9666'5626, 0x17e7'd7a7'9767'5727},
                         {0x18e8'd8a8'9868'5828, 0x19e9'd9a9'9969'5929},
                         {0x1aea'daaa'9a6a'5a2a, 0x1beb'dbab'9b6b'5b2b},
                         {0x1cec'dcac'9c6c'5c2c, 0x1ded'ddad'9d6d'5d2d},
                         {0x1eee'deae'9e6e'5e2e, 0x1fef'dfaf'9f6f'5f2f}},
                        kVectorCalculationsSourceLegacy);
}

TEST_F(Riscv64InterpreterTest, TestVsra) {
  TestVectorInstruction(0xa50c0457,  // Vsra.vv v8, v16, v24, v0.t
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
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(0xa500c457,  // Vsra.vx v8, v16, x1, v0.t
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
                        kVectorCalculationsSourceLegacy);
  TestVectorInstruction(0xa501b457,  // Vsra.vi v8, v16, 0x3, v0.t
                        {{0, 240, 0, 240, 0, 240, 0, 240, 1, 241, 1, 241, 1, 241, 1, 241},
                         {2, 242, 2, 242, 2, 242, 2, 242, 3, 243, 3, 243, 3, 243, 3, 243},
                         {4, 244, 4, 244, 4, 244, 4, 244, 5, 245, 5, 245, 5, 245, 5, 245},
                         {6, 246, 6, 246, 6, 246, 6, 246, 7, 247, 7, 247, 7, 247, 7, 247},
                         {8, 248, 8, 248, 8, 248, 8, 248, 9, 249, 9, 249, 9, 249, 9, 249},
                         {10, 250, 10, 250, 10, 250, 10, 250, 11, 251, 11, 251, 11, 251, 11, 251},
                         {12, 252, 12, 252, 12, 252, 12, 252, 13, 253, 13, 253, 13, 253, 13, 253},
                         {14, 254, 14, 254, 14, 254, 14, 254, 15, 255, 15, 255, 15, 255, 15, 255}},
                        {{0xf020, 0xf060, 0xf0a0, 0xf0e0, 0xf121, 0xf161, 0xf1a1, 0xf1e1},
                         {0xf222, 0xf262, 0xf2a2, 0xf2e2, 0xf323, 0xf363, 0xf3a3, 0xf3e3},
                         {0xf424, 0xf464, 0xf4a4, 0xf4e4, 0xf525, 0xf565, 0xf5a5, 0xf5e5},
                         {0xf626, 0xf666, 0xf6a6, 0xf6e6, 0xf727, 0xf767, 0xf7a7, 0xf7e7},
                         {0xf828, 0xf868, 0xf8a8, 0xf8e8, 0xf929, 0xf969, 0xf9a9, 0xf9e9},
                         {0xfa2a, 0xfa6a, 0xfaaa, 0xfaea, 0xfb2b, 0xfb6b, 0xfbab, 0xfbeb},
                         {0xfc2c, 0xfc6c, 0xfcac, 0xfcec, 0xfd2d, 0xfd6d, 0xfdad, 0xfded},
                         {0xfe2e, 0xfe6e, 0xfeae, 0xfeee, 0xff2f, 0xff6f, 0xffaf, 0xffef}},
                        {{0xf060'5020, 0xf0e0'd0a0, 0xf161'5121, 0xf1e1'd1a1},
                         {0xf262'5222, 0xf2e2'd2a2, 0xf363'5323, 0xf3e3'd3a3},
                         {0xf464'5424, 0xf4e4'd4a4, 0xf565'5525, 0xf5e5'd5a5},
                         {0xf666'5626, 0xf6e6'd6a6, 0xf767'5727, 0xf7e7'd7a7},
                         {0xf868'5828, 0xf8e8'd8a8, 0xf969'5929, 0xf9e9'd9a9},
                         {0xfa6a'5a2a, 0xfaea'daaa, 0xfb6b'5b2b, 0xfbeb'dbab},
                         {0xfc6c'5c2c, 0xfcec'dcac, 0xfd6d'5d2d, 0xfded'ddad},
                         {0xfe6e'5e2e, 0xfeee'deae, 0xff6f'5f2f, 0xffef'dfaf}},
                        {{0xf0e0'd0a0'9060'5020, 0xf1e1'd1a1'9161'5121},
                         {0xf2e2'd2a2'9262'5222, 0xf3e3'd3a3'9363'5323},
                         {0xf4e4'd4a4'9464'5424, 0xf5e5'd5a5'9565'5525},
                         {0xf6e6'd6a6'9666'5626, 0xf7e7'd7a7'9767'5727},
                         {0xf8e8'd8a8'9868'5828, 0xf9e9'd9a9'9969'5929},
                         {0xfaea'daaa'9a6a'5a2a, 0xfbeb'dbab'9b6b'5b2b},
                         {0xfcec'dcac'9c6c'5c2c, 0xfded'ddad'9d6d'5d2d},
                         {0xfeee'deae'9e6e'5e2e, 0xffef'dfaf'9f6f'5f2f}},
                        kVectorCalculationsSourceLegacy);
}

TEST_F(Riscv64InterpreterTest, TestVfmacc) {
  TestVectorFloatInstruction(0xb1881457,  // vfmacc.vv v8, v16, v24, v0.t
                             {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x6a1f'cefd, 0x7629'21c4, 0x6232'9db4, 0x6e3c'70f9},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5551, 0xd66b'bbc8, 0x5555'5555, 0x5555'5037},
                              {0xfaad'fde4, 0xff80'0000, 0xf2c2'c69a, 0xfecd'99e3},
                              {0xff80'0000, 0xff80'0000, 0xff80'0000, 0xff80'0000}},
                             {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x75b4'9040'f9f1'ea75, 0x6dcb'c6d1'12f0'a99b},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0xd614'2330'4af7'4c90, 0x5555'5555'5555'5555},
                              {0xfff0'0000'0000'0000, 0xfe5b'5815'60f1'ac51},
                              {0xfff0'0000'0000'0000, 0xfff0'0000'0000'0000}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0xb100d457,  // vfmacc.vf v8, f1, v16, v0.t
                             {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'550e, 0x5555'0ca1},
                              {0x550b'37bf, 0xd895'6354, 0xdc99'df27, 0xe09c'b3a3},
                              {0xe49f'8677, 0xe8a2'594a, 0xeca5'2c1d, 0xf0a7'fef0},
                              {0xf4aa'd1c3, 0xf8ad'a496, 0xfcb0'7768, 0xff80'0000}},
                             {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0xd780'0dff'a493'9082, 0xdf85'b3a5'4a3b'e0d2},
                              {0xe790'194a'efe1'8677, 0xef95'bef0'9587'2c1d},
                              {0xf7a0'2496'3b2c'd1c3, 0xffa5'ca3b'e0d2'7768}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfnmacc) {
  TestVectorFloatInstruction(0xb5881457,  // vfnmacc.vv v8, v16, v24, v0.t
                             {{0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xea1f'cefd, 0xf629'21c4, 0xe232'9db4, 0xee3c'70f9},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5551, 0x566b'bbc8, 0xd555'5555, 0xd555'5037},
                              {0x7aad'fde4, 0x7f80'0000, 0x72c2'c69a, 0x7ecd'99e3},
                              {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                             {{0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xf5b4'9040'f9f1'ea75, 0xedcb'c6d1'12f0'a99b},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0x5614'2330'4af7'4c90, 0xd555'5555'5555'5555},
                              {0x7ff0'0000'0000'0000, 0x7e5b'5815'60f1'ac51},
                              {0x7ff0'0000'0000'0000, 0x7ff0'0000'0000'0000}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0xb500d457,  // vfnmacc.vf v8, f1, v16, v0.t
                             {{0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'550e, 0xd555'0ca1},
                              {0xd50b'37bf, 0x5895'6354, 0x5c99'df27, 0x609c'b3a3},
                              {0x649f'8677, 0x68a2'594a, 0x6ca5'2c1d, 0x70a7'fef0},
                              {0x74aa'd1c3, 0x78ad'a496, 0x7cb0'7768, 0x7f80'0000}},
                             {{0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0x5780'0dff'a493'9082, 0x5f85'b3a5'4a3b'e0d2},
                              {0x6790'194a'efe1'8677, 0x6f95'bef0'9587'2c1d},
                              {0x77a0'2496'3b2c'd1c3, 0x7fa5'ca3b'e0d2'7768}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfwmacc) {
  __m128i dst_result = {0x0000'0000'0000'0000, 0x0000'0000'0000'0000};
  TestWideningVectorFloatInstruction(0xf1881457,  // vfwmacc.vv v8, v16, v24, v0.t
                                     {{0x3330'e53c'6480'0000, 0x34b2'786b'bbc5'4900},
                                      {0x3234'1766'da4a'6200, 0x33b5'cab6'2d6c'4800},
                                      {0x3937'92ba'5bd0'8000, 0x3ab9'666a'779a'0d00},
                                      {0x383b'4565'd61f'6600, 0x39bd'3935'e5bd'8800},
                                      {0x3f3f'423b'5522'0000, 0x40c0'ab36'1ab7'e880},
                                      {0x3e41'bab3'e9fa'b500, 0x3fc2'd4dc'5007'e400},
                                      {0x4543'f9df'a83a'4000, 0x46c5'2438'7aa3'4a80},
                                      {0x4446'53b6'69e6'3700, 0x45c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf100d457,  // vfwmacc.vf v8, f1, v16, v0.t
                                     {{0xb886'f0ad'0000'0000, 0xb907'a561'b400'0000},
                                      {0xb988'5a16'6800'0000, 0xba09'0ecb'1c00'0000},
                                      {0xba89'c37f'd000'0000, 0xbb0a'7834'8400'0000},
                                      {0xbb8b'2ce9'3800'0000, 0xbc0b'e19d'ec00'0000},
                                      {0xbc8c'9652'a000'0000, 0xbd0d'4b07'5400'0000},
                                      {0xbd8d'ffbc'0800'0000, 0xbe0e'b470'bc00'0000},
                                      {0xbe8f'6925'7000'0000, 0xbf10'0eed'1200'0000},
                                      {0xbf90'6947'6c00'0000, 0xc010'c3a1'c600'0000}},
                                     kVectorCalculationsSource,
                                     dst_result);

  dst_result = {0x401c'6666'6666'6666, 0x401c'6666'6666'6666};
  TestWideningVectorFloatInstruction(0xf1881457,  // vfwmacc.vv v8, v16, v24, v0.t
                                     {{0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'66e3'6f53'baee, 0x40c0'aec2'e784'b54d},
                                      {0x401c'6666'66f4'3c05, 0x401c'fd0d'48e6'a586},
                                      {0x4543'f9df'a83a'4000, 0x46c5'2438'7aa3'4a80},
                                      {0x4446'53b6'69e6'3700, 0x45c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf100d457,  // vfwmacc.vf v8, f1, v16, v0.t
                                     {{0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6657},
                                      {0x401c'6666'6666'5766, 0x401c'6666'6657'0c2e},
                                      {0x401c'6666'56b1'd3ae, 0x401c'6656'5779'5466},
                                      {0x401c'55fd'1efa'6666, 0x4007'4589'40cc'cccc}},
                                     kVectorCalculationsSource,
                                     dst_result);
}

TEST_F(Riscv64InterpreterTest, TestVfwnmacc) {
  __m128i dst_result = {0x0000'0000'0000'0000, 0x0000'0000'0000'0000};
  TestWideningVectorFloatInstruction(0xf5881457,  // vfwnmacc.vv v8, v16, v24, v0.t
                                     {{0xb330'e53c'6480'0000, 0xb4b2'786b'bbc5'4900},
                                      {0xb234'1766'da4a'6200, 0xb3b5'cab6'2d6c'4800},
                                      {0xb937'92ba'5bd0'8000, 0xbab9'666a'779a'0d00},
                                      {0xb83b'4565'd61f'6600, 0xb9bd'3935'e5bd'8800},
                                      {0xbf3f'423b'5522'0000, 0xc0c0'ab36'1ab7'e880},
                                      {0xbe41'bab3'e9fa'b500, 0xbfc2'd4dc'5007'e400},
                                      {0xc543'f9df'a83a'4000, 0xc6c5'2438'7aa3'4a80},
                                      {0xc446'53b6'69e6'3700, 0xc5c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf500d457,  // vfwnmacc.vf v8, f1, v16, v0.t
                                     {{0x3886'f0ad'0000'0000, 0x3907'a561'b400'0000},
                                      {0x3988'5a16'6800'0000, 0x3a09'0ecb'1c00'0000},
                                      {0x3a89'c37f'd000'0000, 0x3b0a'7834'8400'0000},
                                      {0x3b8b'2ce9'3800'0000, 0x3c0b'e19d'ec00'0000},
                                      {0x3c8c'9652'a000'0000, 0x3d0d'4b07'5400'0000},
                                      {0x3d8d'ffbc'0800'0000, 0x3e0e'b470'bc00'0000},
                                      {0x3e8f'6925'7000'0000, 0x3f10'0eed'1200'0000},
                                      {0x3f90'6947'6c00'0000, 0x4010'c3a1'c600'0000}},
                                     kVectorCalculationsSource,
                                     dst_result);

  dst_result = {0x401c'6666'6666'6666, 0x401c'6666'6666'6666};
  TestWideningVectorFloatInstruction(0xf5881457,  // vfwnmacc.vv v8, v16, v24, v0.t
                                     {{0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'66e3'6f53'baee, 0xc0c0'aec2'e784'b54d},
                                      {0xc01c'6666'66f4'3c05, 0xc01c'fd0d'48e6'a586},
                                      {0xc543'f9df'a83a'4000, 0xc6c5'2438'7aa3'4a80},
                                      {0xc446'53b6'69e6'3700, 0xc5c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf500d457,  // vfwnmacc.vf v8, f1, v16, v0.t
                                     {{0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6657},
                                      {0xc01c'6666'6666'5766, 0xc01c'6666'6657'0c2e},
                                      {0xc01c'6666'56b1'd3ae, 0xc01c'6656'5779'5466},
                                      {0xc01c'55fd'1efa'6666, 0xc007'4589'40cc'cccc}},
                                     kVectorCalculationsSource,
                                     dst_result);
}

TEST_F(Riscv64InterpreterTest, TestVfwmsac) {
  __m128i dst_result = {0x0000'0000'0000'0000, 0x0000'0000'0000'0000};
  TestWideningVectorFloatInstruction(0xf9881457,  // vfwmsac.vv v8, v16, v24, v0.t
                                     {{0x3330'e53c'6480'0000, 0x34b2'786b'bbc5'4900},
                                      {0x3234'1766'da4a'6200, 0x33b5'cab6'2d6c'4800},
                                      {0x3937'92ba'5bd0'8000, 0x3ab9'666a'779a'0d00},
                                      {0x383b'4565'd61f'6600, 0x39bd'3935'e5bd'8800},
                                      {0x3f3f'423b'5522'0000, 0x40c0'ab36'1ab7'e880},
                                      {0x3e41'bab3'e9fa'b500, 0x3fc2'd4dc'5007'e400},
                                      {0x4543'f9df'a83a'4000, 0x46c5'2438'7aa3'4a80},
                                      {0x4446'53b6'69e6'3700, 0x45c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf900d457,  // vfwmsac.vf v8, f1, v16, v0.t
                                     {{0xb886'f0ad'0000'0000, 0xb907'a561'b400'0000},
                                      {0xb988'5a16'6800'0000, 0xba09'0ecb'1c00'0000},
                                      {0xba89'c37f'd000'0000, 0xbb0a'7834'8400'0000},
                                      {0xbb8b'2ce9'3800'0000, 0xbc0b'e19d'ec00'0000},
                                      {0xbc8c'9652'a000'0000, 0xbd0d'4b07'5400'0000},
                                      {0xbd8d'ffbc'0800'0000, 0xbe0e'b470'bc00'0000},
                                      {0xbe8f'6925'7000'0000, 0xbf10'0eed'1200'0000},
                                      {0xbf90'6947'6c00'0000, 0xc010'c3a1'c600'0000}},
                                     kVectorCalculationsSource,
                                     dst_result);

  dst_result = {0x401c'6666'6666'6666, 0x401c'6666'6666'6666};
  TestWideningVectorFloatInstruction(0xf9881457,  // vfwmsac.vv v8, v16, v24, v0.t
                                     {{0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'65e9'5d79'11de, 0x40c0'a7a9'4deb'1bb3},
                                      {0xc01c'6666'65d8'90c7, 0xc01b'cfbf'83e6'2746},
                                      {0x4543'f9df'a83a'4000, 0x46c5'2438'7aa3'4a80},
                                      {0x4446'53b6'69e6'3700, 0x45c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xf900d457,  // vfwmsac.vf v8, f1, v16, v0.t
                                     {{0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6666},
                                      {0xc01c'6666'6666'6666, 0xc01c'6666'6666'6675},
                                      {0xc01c'6666'6666'7566, 0xc01c'6666'6675'c09e},
                                      {0xc01c'6666'761a'f91e, 0xc01c'6676'7553'7866},
                                      {0xc01c'76cf'add2'6666, 0xc026'9504'1633'3333}},
                                     kVectorCalculationsSource,
                                     dst_result);
}

TEST_F(Riscv64InterpreterTest, TestVfwnmsac) {
  __m128i dst_result = {0x0000'0000'0000'0000, 0x0000'0000'0000'0000};
  TestWideningVectorFloatInstruction(0xfd881457,  // vfwnmsac.vv v8, v16, v24, v0.t
                                     {{0xb330'e53c'6480'0000, 0xb4b2'786b'bbc5'4900},
                                      {0xb234'1766'da4a'6200, 0xb3b5'cab6'2d6c'4800},
                                      {0xb937'92ba'5bd0'8000, 0xbab9'666a'779a'0d00},
                                      {0xb83b'4565'd61f'6600, 0xb9bd'3935'e5bd'8800},
                                      {0xbf3f'423b'5522'0000, 0xc0c0'ab36'1ab7'e880},
                                      {0xbe41'bab3'e9fa'b500, 0xbfc2'd4dc'5007'e400},
                                      {0xc543'f9df'a83a'4000, 0xc6c5'2438'7aa3'4a80},
                                      {0xc446'53b6'69e6'3700, 0xc5c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xfd00d457,  // vfwnmsac.vf v8, f1, v16, v0.t
                                     {{0x3886'f0ad'0000'0000, 0x3907'a561'b400'0000},
                                      {0x3988'5a16'6800'0000, 0x3a09'0ecb'1c00'0000},
                                      {0x3a89'c37f'd000'0000, 0x3b0a'7834'8400'0000},
                                      {0x3b8b'2ce9'3800'0000, 0x3c0b'e19d'ec00'0000},
                                      {0x3c8c'9652'a000'0000, 0x3d0d'4b07'5400'0000},
                                      {0x3d8d'ffbc'0800'0000, 0x3e0e'b470'bc00'0000},
                                      {0x3e8f'6925'7000'0000, 0x3f10'0eed'1200'0000},
                                      {0x3f90'6947'6c00'0000, 0x4010'c3a1'c600'0000}},
                                     kVectorCalculationsSource,
                                     dst_result);

  dst_result = {0x401c'6666'6666'6666, 0x401c'6666'6666'6666};
  TestWideningVectorFloatInstruction(0xfd881457,  // vfwnmsac.vv v8, v16, v24, v0.t
                                     {{0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'65e9'5d79'11de, 0xc0c0'a7a9'4deb'1bb3},
                                      {0x401c'6666'65d8'90c7, 0x401b'cfbf'83e6'2746},
                                      {0xc543'f9df'a83a'4000, 0xc6c5'2438'7aa3'4a80},
                                      {0xc446'53b6'69e6'3700, 0xc5c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource,
                                     dst_result);
  TestWideningVectorFloatInstruction(0xfd00d457,  // vfwnmsac.vf v8, f1, v16, v0.t
                                     {{0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6666},
                                      {0x401c'6666'6666'6666, 0x401c'6666'6666'6675},
                                      {0x401c'6666'6666'7566, 0x401c'6666'6675'c09e},
                                      {0x401c'6666'761a'f91e, 0x401c'6676'7553'7866},
                                      {0x401c'76cf'add2'6666, 0x4026'9504'1633'3333}},
                                     kVectorCalculationsSource,
                                     dst_result);
}

TEST_F(Riscv64InterpreterTest, TestVfmsac) {
  TestVectorFloatInstruction(0xb9881457,  // vfmsac.vv v8, v16, v24, v0.t
                             {{0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0x6a1f'cefd, 0x7629'21c4, 0x6232'9db3, 0x6e3c'70f9},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5559, 0xd6ab'3339, 0xd555'5555, 0xd555'5a73},
                              {0xfaad'fde4, 0xff80'0000, 0xf2c2'c69a, 0xfecd'99e3},
                              {0xff80'0000, 0xff80'0000, 0xff80'0000, 0xff80'0000}},
                             {{0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0x75b4'9040'f9f1'ea75, 0x6dcb'c6d1'12f0'a99b},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd614'25da'f5a1'f73b, 0xd555'5555'5555'5555},
                              {0xfff0'0000'0000'0000, 0xfe5b'5815'60f1'ac51},
                              {0xfff0'0000'0000'0000, 0xfff0'0000'0000'0000}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0xb900d457,  // vfmsac.vf v8, f1, v16, v0.t
                             {{0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'5555, 0xd555'5555},
                              {0xd555'5555, 0xd555'5555, 0xd555'559c, 0xd555'9e09},
                              {0xd58f'b976, 0xd898'b8aa, 0xdc99'e27d, 0xe09c'b3a6},
                              {0xe49f'8678, 0xe8a2'594a, 0xeca5'2c1d, 0xf0a7'fef0},
                              {0xf4aa'd1c3, 0xf8ad'a496, 0xfcb0'7768, 0xff80'0000}},
                             {{0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd555'5555'5555'5555, 0xd555'5555'5555'5555},
                              {0xd780'0dff'a498'e5d7, 0xdf85'b3a5'4a3b'e0d2},
                              {0xe790'194a'efe1'8678, 0xef95'bef0'9587'2c1d},
                              {0xf7a0'2496'3b2c'd1c3, 0xffa5'ca3b'e0d2'7768}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfnmsac) {
  TestVectorFloatInstruction(0xbd881457,  // vfnmsac.vv v8, v16, v24, v0.t
                             {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0xea1f'cefd, 0xf629'21c4, 0xe232'9db3, 0xee3c'70f9},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5559, 0x56ab'3339, 0x5555'5555, 0x5555'5a73},
                              {0x7aad'fde4, 0x7f80'0000, 0x72c2'c69a, 0x7ecd'99e3},
                              {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                             {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0xf5b4'9040'f9f1'ea75, 0xedcb'c6d1'12f0'a99b},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5614'25da'f5a1'f73b, 0x5555'5555'5555'5555},
                              {0x7ff0'0000'0000'0000, 0x7e5b'5815'60f1'ac51},
                              {0x7ff0'0000'0000'0000, 0x7ff0'0000'0000'0000}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0xbd00d457,  // vfnmsac.vf v8, f1, v16, v0.t
                             {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
                              {0x5555'5555, 0x5555'5555, 0x5555'559c, 0x5555'9e09},
                              {0x558f'b976, 0x5898'b8aa, 0x5c99'e27d, 0x609c'b3a6},
                              {0x649f'8678, 0x68a2'594a, 0x6ca5'2c1d, 0x70a7'fef0},
                              {0x74aa'd1c3, 0x78ad'a496, 0x7cb0'7768, 0x7f80'0000}},
                             {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
                              {0x5780'0dff'a498'e5d7, 0x5f85'b3a5'4a3b'e0d2},
                              {0x6790'194a'efe1'8678, 0x6f95'bef0'9587'2c1d},
                              {0x77a0'2496'3b2c'd1c3, 0x7fa5'ca3b'e0d2'7768}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfmadd) {
  TestVectorFloatInstruction(0xa1881457,  // vfmadd.vv v8, v16, v24, v0.t
                             {{0x98dd'a63a, 0x9e28'a06a, 0xa0e6'e462, 0xa4ed'95be},
                              {0xb624'b220, 0xbe2c'ba29, 0xb100'd4ec, 0xb504'308a},
                              {0xd644'd240, 0xde4c'da49, 0xc654'e5df, 0xce5c'ca7c},
                              {0xf664'f260, 0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78},
                              {0xd922'4bb5, 0xdd25'a463, 0xe128'fd11, 0xe52c'55bf},
                              {0xe92f'ae6d, 0xed33'071b, 0xf136'5fc9, 0xf539'b877},
                              {0xf93d'1125, 0xfd40'69d3, 0xff80'0000, 0xff80'0000},
                              {0xff80'0000, 0xff80'0000, 0xff80'0000, 0xff80'0000}},
                             {{0x9e0c'9a09'9d86'3e2c, 0xa474'5e08'5cb1'b0b0},
                              {0xbe2c'ba29'b624'b220, 0xb484'68bd'bcbc'6610},
                              {0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251},
                              {0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271},
                              {0xdcae'5c5b'af03'ac55, 0xe4b4'88dd'dcdc'8630},
                              {0xecbe'71c6'6f19'1715, 0xf4c4'9393'3ce7'3b90},
                              {0xfcce'8731'2f2e'81d5, 0xfff0'0000'0000'0000},
                              {0xfff0'0000'0000'0000, 0xfff0'0000'0000'0000}},

                             kVectorCalculationsSource);

  TestVectorFloatInstruction(0xa100d457,  // vfmadd.vf v8, f1, v16, v0.t
                             {{0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5695'fffe, 0x5695'fe62},
                              {0x5694'5a5d, 0xd70b'd554, 0xdb5a'8e58, 0xdf5e'dd11},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0xd756'd554'd2da'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},

                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfnmadd) {
  TestVectorFloatInstruction(0xa5881457,  // vfnmadd.vv v8, v16, v24, v0.t
                             {{0x18dd'a63a, 0x1e28'a06a, 0x20e6'e462, 0x24ed'95be},
                              {0x3624'b220, 0x3e2c'ba29, 0x3100'd4ec, 0x3504'308a},
                              {0x5644'd240, 0x5e4c'da49, 0x4654'e5df, 0x4e5c'ca7c},
                              {0x7664'f260, 0x7e6c'fa69, 0x6674'e271, 0x6e7c'ea78},
                              {0x5922'4bb5, 0x5d25'a463, 0x6128'fd11, 0x652c'55bf},
                              {0x692f'ae6d, 0x6d33'071b, 0x7136'5fc9, 0x7539'b877},
                              {0x793d'1125, 0x7d40'69d3, 0x7f80'0000, 0x7f80'0000},
                              {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                             {{0x1e0c'9a09'9d86'3e2c, 0x2474'5e08'5cb1'b0b0},
                              {0x3e2c'ba29'b624'b220, 0x3484'68bd'bcbc'6610},
                              {0x5e4c'da49'd644'd240, 0x4e5c'ca58'c654'c251},
                              {0x7e6c'fa69'f664'f260, 0x6e7c'ea78'e674'e271},
                              {0x5cae'5c5b'af03'ac55, 0x64b4'88dd'dcdc'8630},
                              {0x6cbe'71c6'6f19'1715, 0x74c4'9393'3ce7'3b90},
                              {0x7cce'8731'2f2e'81d5, 0x7ff0'0000'0000'0000},
                              {0x7ff0'0000'0000'0000, 0x7ff0'0000'0000'0000}},
                             kVectorCalculationsSource);

  TestVectorFloatInstruction(0xa500d457,  // vfmadd.vf v8, f1, v16, v0.t
                             {{0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd695'fffe, 0xd695'fe62},
                              {0xd694'5a5d, 0x570b'd554, 0x5b5a'8e58, 0x5f5e'dd11},
                              {0x6362'e160, 0x6766'e564, 0x6b6a'e968, 0x6f6e'ed6c},
                              {0x7372'f170, 0x7776'f574, 0x7b7a'f978, 0x7f7e'fd7c}},
                             {{0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0x5756'd554'd2da'd150, 0x5f5e'dd5c'db5a'd958},
                              {0x6766'e564'e362'e160, 0x6f6e'ed6c'eb6a'e968},
                              {0x7776'f574'f372'f170, 0x7f7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfmsub) {
  TestVectorFloatInstruction(0xa9881457,  // vfmsub.vv v8, v16, v24, v0.t
                             {{0x98d5'5d1a, 0x1de1'2750, 0xa0e6'e462, 0xa4ed'95be},
                              {0x3624'b220, 0x3e2c'ba29, 0xb100'd4e6, 0xb504'2aa4},
                              {0x5644'd240, 0x5e4c'da49, 0x4654'9ec3, 0x4e5c'ca34},
                              {0x7664'f260, 0x7e6c'fa69, 0x6674'e271, 0x6e7c'ea78},
                              {0xd922'4bb5, 0xdd25'a463, 0xe128'fd11, 0xe52c'55bf},
                              {0xe92f'ae6d, 0xed33'071b, 0xf136'5fc9, 0xf539'b877},
                              {0xf93d'1125, 0xfd40'69d3, 0xff80'0000, 0xff80'0000},
                              {0xff80'0000, 0xff80'0000, 0xff80'0000, 0xff80'0000}},
                             {{0x1e0c'9a09'8e82'e5d4, 0xa474'5e08'5cb1'b0b0},
                              {0x3e2c'ba29'b624'b220, 0xb484'68bd'bcbc'6610},
                              {0x5e4c'da49'd644'd240, 0x4e5c'ca58'c654'c251},
                              {0x7e6c'fa69'f664'f260, 0x6e7c'ea78'e674'e271},
                              {0xdcae'5c5b'af03'ac55, 0xe4b4'88dd'dcdc'8630},
                              {0xecbe'71c6'6f19'1715, 0xf4c4'9393'3ce7'3b90},
                              {0xfcce'8731'2f2e'81d5, 0xfff0'0000'0000'0000},
                              {0xfff0'0000'0000'0000, 0xfff0'0000'0000'0000}},
                             kVectorCalculationsSource);

  TestVectorFloatInstruction(0xa900d457,  // vfmsub.vf v8, f1, v16, v0.t
                             {{0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0000, 0x5696'0000},
                              {0x5696'0000, 0x5696'0000, 0x5696'0001, 0x5696'019d},
                              {0x5697'a5a2, 0x5790'eaaa, 0x5b5b'2458, 0x5f5e'dda7},
                              {0x6362'e160, 0x6766'e564, 0x6b6a'e968, 0x6f6e'ed6c},
                              {0x7372'f170, 0x7776'f574, 0x7b7a'f978, 0x7f7e'fd7c}},
                             {{0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x557e'0000'0000'0000, 0x557e'0000'0000'0000},
                              {0x5756'd554'd3ca'd150, 0x5f5e'dd5c'db5a'd958},
                              {0x6766'e564'e362'e160, 0x6f6e'ed6c'eb6a'e968},
                              {0x7776'f574'f372'f170, 0x7f7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfnmsub) {
  TestVectorFloatInstruction(0xad881457,  // vfnmsub.vv v8, v16, v24, v0.t
                             {{0x18d5'5d1a, 0x9de1'2750, 0x20e6'e462, 0x24ed'95be},
                              {0xb624'b220, 0xbe2c'ba29, 0x3100'd4e6, 0x3504'2aa4},
                              {0xd644'd240, 0xde4c'da49, 0xc654'9ec3, 0xce5c'ca34},
                              {0xf664'f260, 0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78},
                              {0x5922'4bb5, 0x5d25'a463, 0x6128'fd11, 0x652c'55bf},
                              {0x692f'ae6d, 0x6d33'071b, 0x7136'5fc9, 0x7539'b877},
                              {0x793d'1125, 0x7d40'69d3, 0x7f80'0000, 0x7f80'0000},
                              {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                             {{0x9e0c'9a09'8e82'e5d4, 0x2474'5e08'5cb1'b0b0},
                              {0xbe2c'ba29'b624'b220, 0x3484'68bd'bcbc'6610},
                              {0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251},
                              {0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271},
                              {0x5cae'5c5b'af03'ac55, 0x64b4'88dd'dcdc'8630},
                              {0x6cbe'71c6'6f19'1715, 0x74c4'9393'3ce7'3b90},
                              {0x7cce'8731'2f2e'81d5, 0x7ff0'0000'0000'0000},
                              {0x7ff0'0000'0000'0000, 0x7ff0'0000'0000'0000}},
                             kVectorCalculationsSource);

  TestVectorFloatInstruction(0xad00d457,  // vfnmsub.vf v8, f1, v16, v0.t
                             {{0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0000, 0xd696'0000},
                              {0xd696'0000, 0xd696'0000, 0xd696'0001, 0xd696'019d},
                              {0xd697'a5a2, 0xd790'eaaa, 0xdb5b'2458, 0xdf5e'dda7},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd57e'0000'0000'0000, 0xd57e'0000'0000'0000},
                              {0xd756'd554'd3ca'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVnmsac) {
  TestVectorInstruction(0xbd882457,  // vnmsac.vv v8, v16, v24, v0.t
                        {{85, 195, 77, 147, 49, 83, 13, 3, 205, 195, 141, 147, 53, 83, 205, 3},
                         {85, 131, 205, 211, 33, 19, 141, 67, 189, 131, 13, 211, 53, 19, 77, 67},
                         {85, 67, 77, 19, 17, 211, 13, 131, 173, 67, 141, 19, 53, 211, 205, 131},
                         {85, 3, 205, 83, 1, 147, 141, 195, 157, 3, 13, 83, 53, 147, 77, 195},
                         {85, 195, 77, 147, 241, 83, 13, 3, 141, 195, 141, 147, 53, 83, 205, 3},
                         {85, 131, 205, 211, 225, 19, 141, 67, 125, 131, 13, 211, 53, 19, 77, 67},
                         {85, 67, 77, 19, 209, 211, 13, 131, 109, 67, 141, 19, 53, 211, 205, 131},
                         {85, 3, 205, 83, 193, 147, 141, 195, 93, 3, 13, 83, 53, 147, 77, 195}},
                        {{0x5555, 0x1d4d, 0x4031, 0x4d0d, 0x2bcd, 0x3c8d, 0xa435, 0xebcd},
                         {0x1355, 0xdacd, 0xed21, 0x098d, 0xd7bd, 0xf80d, 0x5f35, 0xa64d},
                         {0xcd55, 0x944d, 0x9611, 0xc20d, 0x7fad, 0xaf8d, 0x1635, 0x5ccd},
                         {0x8355, 0x49cd, 0x3b01, 0x768d, 0x239d, 0x630d, 0xc935, 0x0f4d},
                         {0x3555, 0xfb4d, 0xdbf1, 0x270d, 0xc38d, 0x128d, 0x7835, 0xbdcd},
                         {0xe355, 0xa8cd, 0x78e1, 0xd38d, 0x5f7d, 0xbe0d, 0x2335, 0x684d},
                         {0x8d55, 0x524d, 0x11d1, 0x7c0d, 0xf76d, 0x658d, 0xca35, 0x0ecd},
                         {0x3355, 0xf7cd, 0xa6c1, 0x208d, 0x8b5d, 0x090d, 0x6d35, 0xb14d}},
                        {{0xe3c3'5555, 0xf5e6'4031, 0xdb6c'2bcd, 0xfe9f'a435},
                         {0x43e6'1355, 0x42f6'ed21, 0x277c'd7bd, 0x57be'5f35},
                         {0x9800'cd55, 0x83ff'9611, 0x6785'7fad, 0xa4d5'1635},
                         {0xe013'8355, 0xb900'3b01, 0x9b86'239d, 0xe5e3'c935},
                         {0xdd5e'3555, 0xa73c'dbf1, 0x8cc6'c38d, 0xe836'7835},
                         {0x1d70'e355, 0xd43d'78e1, 0xb8c7'5f7d, 0x2145'2335},
                         {0x517b'8d55, 0xf536'11d1, 0xd8bf'f76d, 0x4e4b'ca35},
                         {0x797e'3355, 0x0a26'a6c1, 0xecb0'8b5d, 0x6f4a'6d35}},
                        {{0xbbf2'b5fd'e3c3'5555, 0xb840'0f97'db6c'2bcd},
                         {0x4d47'a732'43e6'1355, 0x418e'fac8'277c'd7bd},
                         {0xc284'8456'9800'cd55, 0xaec5'd1e8'6785'7fad},
                         {0x1ba9'4d6a'e013'8355, 0xffe4'94f8'9b86'239d},
                         {0xa27d'86f1'dd5e'3555, 0x96ca'd88b'8cc6'c38d},
                         {0xf3a2'4806'1d70'e355, 0xdfe9'939b'b8c7'5f7d},
                         {0x28ae'f50a'517b'8d55, 0x0cf0'3a9b'd8bf'f76d},
                         {0x41a3'8dfe'797e'3355, 0x1dde'cd8b'ecb0'8b5d}},
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

TEST_F(Riscv64InterpreterTest, TestVnmsub) {
  TestVectorInstruction(
      0xad882457,  // vnmsub.vv v8, v16, v24, v0.t
      {{0, 189, 90, 23, 181, 113, 14, 203, 105, 5, 194, 95, 28, 185, 118, 19},
       {208, 141, 42, 231, 133, 65, 222, 155, 57, 213, 146, 47, 236, 137, 70, 227},
       {160, 93, 250, 183, 85, 17, 174, 107, 9, 165, 98, 255, 188, 89, 22, 179},
       {112, 45, 202, 135, 37, 225, 126, 59, 217, 117, 50, 207, 140, 41, 230, 131},
       {64, 253, 154, 87, 245, 177, 78, 11, 169, 69, 2, 159, 92, 249, 182, 83},
       {16, 205, 106, 39, 197, 129, 30, 219, 121, 21, 210, 111, 44, 201, 134, 35},
       {224, 157, 58, 247, 149, 81, 238, 171, 73, 229, 162, 63, 252, 153, 86, 243},
       {176, 109, 10, 199, 101, 33, 190, 123, 25, 181, 114, 15, 204, 105, 38, 195}},
      {{0xbd00, 0x6c5a, 0x1bb5, 0xcb0e, 0x5a69, 0x09c2, 0xb91c, 0x6876},
       {0x37d0, 0xe72a, 0x9685, 0x45de, 0xd539, 0x8492, 0x33ec, 0xe346},
       {0xb2a0, 0x61fa, 0x1155, 0xc0ae, 0x5009, 0xff62, 0xaebc, 0x5e16},
       {0x2d70, 0xdcca, 0x8c25, 0x3b7e, 0xcad9, 0x7a32, 0x298c, 0xd8e6},
       {0xa840, 0x579a, 0x06f5, 0xb64e, 0x45a9, 0xf502, 0xa45c, 0x53b6},
       {0x2310, 0xd26a, 0x81c5, 0x311e, 0xc079, 0x6fd2, 0x1f2c, 0xce86},
       {0x9de0, 0x4d3a, 0xfc95, 0xabee, 0x3b49, 0xeaa2, 0x99fc, 0x4956},
       {0x18b0, 0xc80a, 0x7765, 0x26be, 0xb619, 0x6572, 0x14cc, 0xc426}},
      {{0x6c5a'bd00, 0x2064'1bb5, 0xb46d'5a69, 0x6876'b91c},
       {0x3c80'37d0, 0xf089'9685, 0x8492'd539, 0x389c'33ec},
       {0x0ca5'b2a0, 0xc0af'1155, 0x54b8'5009, 0x08c1'aebc},
       {0xdccb'2d70, 0x90d4'8c25, 0x24dd'cad9, 0xd8e7'298c},
       {0xacef'a840, 0x60f9'06f5, 0xf502'45a9, 0xa90b'a45c},
       {0x7d15'2310, 0x311e'81c5, 0xc527'c079, 0x7931'1f2c},
       {0x4d3a'9de0, 0x0143'fc95, 0x954d'3b49, 0x4956'99fc},
       {0x1d60'18b0, 0xd169'7765, 0x6572'b619, 0x197c'14cc}},
      {{0xcb0e'c660'6c5a'bd00, 0x6876'b91c'b46d'5a69},
       {0x45de'ebdb'3c80'37d0, 0xe346'de97'8492'd539},
       {0xc0af'1156'0ca5'b2a0, 0x5e17'0412'54b8'5009},
       {0x3b7f'36d0'dccb'2d70, 0xd8e7'298d'24dd'cad9},
       {0xb64e'5c4a'acef'a840, 0x53b6'4f06'f502'45a9},
       {0x311e'81c5'7d15'2310, 0xce86'7481'c527'c079},
       {0xabee'a740'4d3a'9de0, 0x4956'99fc'954d'3b49},
       {0x26be'ccbb'1d60'18b0, 0xc426'bf77'6572'b619}},
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

TEST_F(Riscv64InterpreterTest, TestVfmin) {
  TestVectorFloatInstruction(0x1100d457,  // vfmin.vf v8, v16, f1, v0.t
                             {{0xf005'f005, 0xf005'f005, 0x4040'4040, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
                              {0xaaaa'aaaa, 0xaaaa'aaaa, 0x1111'1111, 0x1111'1111},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb},
                              {0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9}},
                             {{0xf005'f005'f005'f005, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
                              {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
                              {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9}},
                             kVectorComparisonSource);
  TestVectorFloatInstruction(0x110c1457,  // vfmin.vv v8,v16,v24,v0.t
                             {{0xf005'f005, 0xf005'f005, 0x4040'4040, 0x7fc0'0000},
                              {0x1111'1111, 0x1111'1111, 0x1111'1111, 0x1111'1111},
                              {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
                              {0xaaaa'aaaa, 0xaaaa'aaaa, 0x1111'1111, 0x1111'1111},
                              {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
                              {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
                              {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
                              {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8}},
                             {{0xf005'f005'f005'f005, 0x7ff8'0000'0000'0000},
                              {0x1111'1111'1111'1111, 0x1111'1111'1111'1111},
                              {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
                              {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
                              {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
                              {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
                              {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
                              {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1}},
                             kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVfmax) {
  TestVectorFloatInstruction(0x1900d457,  // vfmax.vf v8, v16, f1, v0.t
                             {{0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'40b4, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                              {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000}},
                             {{0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                              {0x4016'8000'0000'0000, 0x4016'8000'0000'0000}},
                             kVectorComparisonSource);
  TestVectorFloatInstruction(0x190c1457,  // vfmax.vv v8,v16,v24,v0.t
                             {{0xf005'f005, 0xf005'f005, 0x4040'4040, 0x7fc0'0000},
                              {0x40b4'40b4, 0x1111'1111, 0x40b4'0000, 0x1111'1111},
                              {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
                              {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
                              {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
                              {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
                              {0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb},
                              {0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9}},
                             {{0xf005'f005'f005'f005, 0x7ff8'0000'0000'0000},
                              {0x1111'1111'1111'1111, 0x1111'1111'1111'1111},
                              {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
                              {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271},
                              {0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
                              {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
                              {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
                              {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9}},
                             kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVfsgnj) {
  TestVectorFloatInstruction(0x210c1457,  // vfsgnj.vv v8, v16, v24, v0.t
                             {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
                              {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
                              {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
                              {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
                              {0x4342'c140, 0x4746'c544, 0x4b4a'c948, 0x4f4e'cd4c},
                              {0x5352'd150, 0x5756'd554, 0x5b5a'd958, 0x5f5e'dd5c},
                              {0x6362'e160, 0x6766'e564, 0x6b6a'e968, 0x6f6e'ed6c},
                              {0x7372'f170, 0x7776'f574, 0x7b7a'f978, 0x7f7e'fd7c}},
                             {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
                              {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
                              {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
                              {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
                              {0x4746'c544'c342'c140, 0x4f4e'cd4c'cb4a'c948},
                              {0x5756'd554'd352'd150, 0x5f5e'dd5c'db5a'd958},
                              {0x6766'e564'e362'e160, 0x6f6e'ed6c'eb6a'e968},
                              {0x7776'f574'f372'f170, 0x7f7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0x2100d457,  // vfsgnj.vf v8, v16, f1, v0.t
                             {{0x0302'8100, 0x0706'8504, 0x0b0a'8908, 0x0f0e'8d0c},
                              {0x1312'9110, 0x1716'9514, 0x1b1a'9918, 0x1f1e'9d1c},
                              {0x2322'a120, 0x2726'a524, 0x2b2a'a928, 0x2f2e'ad2c},
                              {0x3332'b130, 0x3736'b534, 0x3b3a'b938, 0x3f3e'bd3c},
                              {0x4342'c140, 0x4746'c544, 0x4b4a'c948, 0x4f4e'cd4c},
                              {0x5352'd150, 0x5756'd554, 0x5b5a'd958, 0x5f5e'dd5c},
                              {0x6362'e160, 0x6766'e564, 0x6b6a'e968, 0x6f6e'ed6c},
                              {0x7372'f170, 0x7776'f574, 0x7b7a'f978, 0x7f7e'fd7c}},
                             {{0x0706'8504'8302'8100, 0x0f0e'8d0c'8b0a'8908},
                              {0x1716'9514'9312'9110, 0x1f1e'9d1c'9b1a'9918},
                              {0x2726'a524'a322'a120, 0x2f2e'ad2c'ab2a'a928},
                              {0x3736'b534'b332'b130, 0x3f3e'bd3c'bb3a'b938},
                              {0x4746'c544'c342'c140, 0x4f4e'cd4c'cb4a'c948},
                              {0x5756'd554'd352'd150, 0x5f5e'dd5c'db5a'd958},
                              {0x6766'e564'e362'e160, 0x6f6e'ed6c'eb6a'e968},
                              {0x7776'f574'f372'f170, 0x7f7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0x250c1457,  // vfsgnjn.vv v8, v16, v24, v0.t
                             {{0x0302'8100, 0x0706'8504, 0x0b0a'8908, 0x0f0e'8d0c},
                              {0x1312'9110, 0x1716'9514, 0x1b1a'9918, 0x1f1e'9d1c},
                              {0x2322'a120, 0x2726'a524, 0x2b2a'a928, 0x2f2e'ad2c},
                              {0x3332'b130, 0x3736'b534, 0x3b3a'b938, 0x3f3e'bd3c},
                              {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x0706'8504'8302'8100, 0x0f0e'8d0c'8b0a'8908},
                              {0x1716'9514'9312'9110, 0x1f1e'9d1c'9b1a'9918},
                              {0x2726'a524'a322'a120, 0x2f2e'ad2c'ab2a'a928},
                              {0x3736'b534'b332'b130, 0x3f3e'bd3c'bb3a'b938},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0x2500d457,  // vfsgnjn.vf v8, v16, f1, v0.t
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
  TestVectorFloatInstruction(0x290c1457,  // vfsgnjx.vv v8, v16, v24, v0.t
                             {{0x0302'8100, 0x0706'8504, 0x0b0a'8908, 0x0f0e'8d0c},
                              {0x1312'9110, 0x1716'9514, 0x1b1a'9918, 0x1f1e'9d1c},
                              {0x2322'a120, 0x2726'a524, 0x2b2a'a928, 0x2f2e'ad2c},
                              {0x3332'b130, 0x3736'b534, 0x3b3a'b938, 0x3f3e'bd3c},
                              {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
                              {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
                              {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
                              {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
                             {{0x0706'8504'8302'8100, 0x0f0e'8d0c'8b0a'8908},
                              {0x1716'9514'9312'9110, 0x1f1e'9d1c'9b1a'9918},
                              {0x2726'a524'a322'a120, 0x2f2e'ad2c'ab2a'a928},
                              {0x3736'b534'b332'b130, 0x3f3e'bd3c'bb3a'b938},
                              {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
                              {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
                              {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
                              {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
                             kVectorCalculationsSource);
  TestVectorFloatInstruction(0x2900d457,  // vfsgnjx.vf v8, v16, f1, v0.t
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
}

TEST_F(Riscv64InterpreterTest, TestVredsum) {
  TestVectorReductionInstruction(
      0x1882457,  // vredsum.vs v8,v24,v16,v0.t
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

TEST_F(Riscv64InterpreterTest, TestVfredosum) {
  TestVectorReductionInstruction(0xd881457,  // vfredosum.vs v8, v24, v16, v0.t
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

// Currently Vfredusum is implemented as Vfredosum (as explicitly permitted by RVV 1.0).
// If we would implement some speedups which would change results then we may need to alter tests.
TEST_F(Riscv64InterpreterTest, TestVfredusum) {
  TestVectorReductionInstruction(0x5881457,  // vfredusum.vs v8, v24, v16, v0.t
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

TEST_F(Riscv64InterpreterTest, TestVredand) {
  TestVectorReductionInstruction(
      0x5882457,  // vredand.vs v8,v24,v16,v0.t
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

TEST_F(Riscv64InterpreterTest, TestVredor) {
  TestVectorReductionInstruction(
      0x9882457,  // vredor.vs v8,v24,v16,v0.t
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

TEST_F(Riscv64InterpreterTest, TestVredxor) {
  TestVectorReductionInstruction(
      0xd882457,  // vredxor.vs v8,v24,v16,v0.t
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

TEST_F(Riscv64InterpreterTest, TestVredminu) {
  TestVectorReductionInstruction(
      0x11882457,  // vredminu.vs v8,v24,v16,v0.t
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

TEST_F(Riscv64InterpreterTest, TestVredmin) {
  TestVectorReductionInstruction(
      0x15882457,  // vredmin.vs v8,v24,v16,v0.t
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

TEST_F(Riscv64InterpreterTest, TestVfredmin) {
  TestVectorReductionInstruction(0x15881457,  // vfredmin.vs v8, v24, v16, v0.t
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

TEST_F(Riscv64InterpreterTest, TestVredmaxu) {
  TestVectorReductionInstruction(
      0x19882457,  // vredmaxu.vs v8,v24,v16,v0.t
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

TEST_F(Riscv64InterpreterTest, TestVredmax) {
  TestVectorReductionInstruction(
      0x1d882457,  // vredmax.vs v8,v24,v16,v0.t
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

TEST_F(Riscv64InterpreterTest, TestVfredmax) {
  TestVectorReductionInstruction(0x1d881457,  // vfredmax.vs v8, v24, v16, v0.t
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

// Note that the expected test outputs for v[f]merge.vXm are identical to those for v[f]mv.v.X.
// This happens because v[f]merge.vXm is just a v[f]mv.v.X with mask (second operand is not used
// by v[f]mv.v.X but the difference between v[f]merge.vXm and v[f]mv.v.X is captured in masking
// logic within TestVectorInstruction itself via the parameter TestVectorInstructionMode::kVMerge
// for V[f]merge/V[f]mv).
TEST_F(Riscv64InterpreterTest, TestVmerge) {
  TestVectorMergeFloatInstruction(0x5d00d457,  // Vfmerge.vfm v8, v16, f1, v0
                                  {{0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000},
                                   {0x40b4'0000, 0x40b4'0000, 0x40b4'0000, 0x40b4'0000}},
                                  {{0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000},
                                   {0x4016'8000'0000'0000, 0x4016'8000'0000'0000}},
                                  kVectorCalculationsSource);
  TestVectorMergeInstruction(
      0x5d0c0457,  // Vmerge.vvm v8, v16, v24, v0
      {{0, 146, 4, 150, 9, 154, 12, 158, 17, 130, 20, 134, 24, 138, 28, 142},
       {32, 178, 36, 182, 41, 186, 44, 190, 49, 162, 52, 166, 56, 170, 60, 174},
       {64, 210, 68, 214, 73, 218, 76, 222, 81, 194, 84, 198, 88, 202, 92, 206},
       {96, 242, 100, 246, 105, 250, 108, 254, 113, 226, 116, 230, 120, 234, 124, 238},
       {128, 18, 132, 22, 137, 26, 140, 30, 145, 2, 148, 6, 152, 10, 156, 14},
       {160, 50, 164, 54, 169, 58, 172, 62, 177, 34, 180, 38, 184, 42, 188, 46},
       {192, 82, 196, 86, 201, 90, 204, 94, 209, 66, 212, 70, 216, 74, 220, 78},
       {224, 114, 228, 118, 233, 122, 236, 126, 241, 98, 244, 102, 248, 106, 252, 110}},
      {{0x9200, 0x9604, 0x9a09, 0x9e0c, 0x8211, 0x8614, 0x8a18, 0x8e1c},
       {0xb220, 0xb624, 0xba29, 0xbe2c, 0xa231, 0xa634, 0xaa38, 0xae3c},
       {0xd240, 0xd644, 0xda49, 0xde4c, 0xc251, 0xc654, 0xca58, 0xce5c},
       {0xf260, 0xf664, 0xfa69, 0xfe6c, 0xe271, 0xe674, 0xea78, 0xee7c},
       {0x1280, 0x1684, 0x1a89, 0x1e8c, 0x0291, 0x0694, 0x0a98, 0x0e9c},
       {0x32a0, 0x36a4, 0x3aa9, 0x3eac, 0x22b1, 0x26b4, 0x2ab8, 0x2ebc},
       {0x52c0, 0x56c4, 0x5ac9, 0x5ecc, 0x42d1, 0x46d4, 0x4ad8, 0x4edc},
       {0x72e0, 0x76e4, 0x7ae9, 0x7eec, 0x62f1, 0x66f4, 0x6af8, 0x6efc}},
      {{0x9604'9200, 0x9e0c'9a09, 0x8614'8211, 0x8e1c'8a18},
       {0xb624'b220, 0xbe2c'ba29, 0xa634'a231, 0xae3c'aa38},
       {0xd644'd240, 0xde4c'da49, 0xc654'c251, 0xce5c'ca58},
       {0xf664'f260, 0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78},
       {0x1684'1280, 0x1e8c'1a89, 0x0694'0291, 0x0e9c'0a98},
       {0x36a4'32a0, 0x3eac'3aa9, 0x26b4'22b1, 0x2ebc'2ab8},
       {0x56c4'52c0, 0x5ecc'5ac9, 0x46d4'42d1, 0x4edc'4ad8},
       {0x76e4'72e0, 0x7eec'7ae9, 0x66f4'62f1, 0x6efc'6af8}},
      {{0x9e0c'9a09'9604'9200, 0x8e1c'8a18'8614'8211},
       {0xbe2c'ba29'b624'b220, 0xae3c'aa38'a634'a231},
       {0xde4c'da49'd644'd240, 0xce5c'ca58'c654'c251},
       {0xfe6c'fa69'f664'f260, 0xee7c'ea78'e674'e271},
       {0x1e8c'1a89'1684'1280, 0x0e9c'0a98'0694'0291},
       {0x3eac'3aa9'36a4'32a0, 0x2ebc'2ab8'26b4'22b1},
       {0x5ecc'5ac9'56c4'52c0, 0x4edc'4ad8'46d4'42d1},
       {0x7eec'7ae9'76e4'72e0, 0x6efc'6af8'66f4'62f1}},
      kVectorCalculationsSource);
  TestVectorMergeInstruction(
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
      kVectorCalculationsSource);
  TestVectorMergeInstruction(
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
      kVectorCalculationsSourceLegacy);
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
      kVectorCalculationsSourceLegacy);
  TestVectorFloatInstruction(0x910c1457,  // vfmul.vv v8, v16, v24, v0.t
                             {{0x8000'0000, 0x8000'0000, 0x8000'0000, 0x8000'0000},
                              {0x8000'02f0, 0x85ca'89ec, 0x91d9'a3e9, 0x9de9'3ee6},
                              {0xa9f9'5ae5, 0xb604'fbf4, 0xc20d'8af5, 0xce16'5a77},
                              {0xda1f'6a7a, 0xe628'bafe, 0xf232'4c02, 0xfe3c'1d87},
                              {0x0a49'9dd9, 0x165a'3ee4, 0x226b'60ef, 0x2e7d'03f9},
                              {0x3a87'9403, 0x4690'e68c, 0x529a'7994, 0x5ea4'4d1d},
                              {0x6aae'6126, 0x76b8'b5b2, 0x7f80'0000, 0x7f80'0000},
                              {0x7f80'0000, 0x7f80'0000, 0x7f80'0000, 0x7f80'0000}},
                             {{0x8000'0000'0000'0000, 0x8000'0000'0000'0000},
                              {0x8553'e032'b59e'2bf7, 0x9d6b'012b'925d'8532},
                              {0xb584'0511'cdec'af2c, 0xcd9b'2e22'd263'd03f},
                              {0xe5b4'2a11'269b'b302, 0xfdcb'5b3a'52ca'9bed},
                              {0x15e4'4f30'bfab'3779, 0x2dfb'8872'1391'e83b},
                              {0x4614'7470'991b'3c90, 0x5e2b'b5ca'14b9'b52b},
                              {0x7644'99d0'b2eb'c249, 0x7ff0'0000'0000'0000},
                              {0x7ff0'0000'0000'0000, 0x7ff0'0000'0000'0000}},
                             kVectorCalculationsSourceLegacy);
  TestVectorFloatInstruction(0x9100d457,  // vfmul.vf v8, v16, f1, v0.t
                             {{0x8437'8568, 0x883d'2b0e, 0x8c42'd0b3, 0x9048'7659},
                              {0x944e'1bfe, 0x9853'c1a4, 0x9c59'674a, 0xa05f'0cef},
                              {0xa464'b295, 0xa86a'583b, 0xac6f'fde0, 0xb075'a386},
                              {0xb47b'492c, 0xb880'7769, 0xbc83'4a3b, 0xc086'1d0e},
                              {0xc488'efe1, 0xc88b'c2b4, 0xcc8e'9587, 0xd091'6859},
                              {0xd494'3b2c, 0xd897'0dff, 0xdc99'e0d2, 0xe09c'b3a5},
                              {0xe49f'8678, 0xe8a2'594a, 0xeca5'2c1d, 0xf0a7'fef0},
                              {0xf4aa'd1c3, 0xf8ad'a496, 0xfcb0'7768, 0xff80'0000}},
                             {{0x872f'ab0e'583b'8568, 0x8f35'7b2c'd1c3'685a},
                              {0x973f'c1a4'eed2'1bfe, 0x9f45'8678'1d0e'b3a5},
                              {0xa74f'd83b'8568'b295, 0xaf55'91c3'6859'fef0},
                              {0xb75f'eed2'1bff'492c, 0xbf65'9d0e'b3a5'4a3b},
                              {0xc770'02b4'594a'efe1, 0xcf75'a859'fef0'9587},
                              {0xd780'0dff'a496'3b2c, 0xdf85'b3a5'4a3b'e0d2},
                              {0xe790'194a'efe1'8678, 0xef95'bef0'9587'2c1d},
                              {0xf7a0'2496'3b2c'd1c3, 0xffa5'ca3b'e0d2'7768}},
                             kVectorCalculationsSourceLegacy);
  TestWideningVectorFloatInstruction(0xe10c1457,  // vfwmul.vv v8, v16, v24, v0.t
                                     {{0x3330'e53c'6480'0000, 0x34b2'786b'bbc5'4900},
                                      {0x3234'1766'da4a'6200, 0x33b5'cab6'2d6c'4800},
                                      {0x3937'92ba'5bd0'8000, 0x3ab9'666a'779a'0d00},
                                      {0x383b'4565'd61f'6600, 0x39bd'3935'e5bd'8800},
                                      {0x3f3f'423b'5522'0000, 0x40c0'ab36'1ab7'e880},
                                      {0x3e41'bab3'e9fa'b500, 0x3fc2'd4dc'5007'e400},
                                      {0x4543'f9df'a83a'4000, 0x46c5'2438'7aa3'4a80},
                                      {0x4446'53b6'69e6'3700, 0x45c7'8e1f'2e31'8400}},
                                     kVectorCalculationsSource);
  TestWideningVectorFloatInstruction(0xe100d457,  // vfwmul.vf v8, v16, f1, v0.t
                                     {{0xb886'f0ad'0000'0000, 0xb907'a561'b400'0000},
                                      {0xb988'5a16'6800'0000, 0xba09'0ecb'1c00'0000},
                                      {0xba89'c37f'd000'0000, 0xbb0a'7834'8400'0000},
                                      {0xbb8b'2ce9'3800'0000, 0xbc0b'e19d'ec00'0000},
                                      {0xbc8c'9652'a000'0000, 0xbd0d'4b07'5400'0000},
                                      {0xbd8d'ffbc'0800'0000, 0xbe0e'b470'bc00'0000},
                                      {0xbe8f'6925'7000'0000, 0xbf10'0eed'1200'0000},
                                      {0xbf90'6947'6c00'0000, 0xc010'c3a1'c600'0000}},
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
      kVectorCalculationsSourceLegacy);
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
      kVectorCalculationsSourceLegacy);
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
      kVectorCalculationsSourceLegacy);
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
      kVectorCalculationsSourceLegacy);
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
      kVectorCalculationsSourceLegacy);
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
      kVectorCalculationsSourceLegacy);
}

TEST_F(Riscv64InterpreterTest, TestVrem) {
  TestVectorInstruction(
      0x890c2457,  // vremu.vv v8, v16, v24, v0.t
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {10, 13, 10, 13, 0, 0, 0, 0, 0, 0, 10, 13, 0, 0, 0, 0},
       {22, 64, 22, 64, 22, 64, 22, 64, 0, 0, 0, 0, 0, 128, 22, 64},
       {74, 72, 70, 68, 65, 64, 62, 60, 17, 17, 17, 17, 17, 17, 17, 17},
       {116, 125, 112, 121, 107, 117, 104, 113, 101, 109, 98, 105, 94, 101, 90, 97},
       {84, 93, 80, 89, 79, 85, 76, 81, 68, 77, 65, 73, 61, 69, 57, 65},
       {187, 187, 187, 169, 187, 187, 187, 169, 187, 187, 187, 169, 187, 187, 187, 169},
       {169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0d81, 0x0d81, 0x0000, 0x0000, 0x0000, 0x0d81, 0x0000, 0x0000},
       {0x4016, 0x4016, 0x4016, 0x4016, 0x0000, 0x0000, 0x8000, 0x4016},
       {0x484a, 0x4446, 0x4041, 0x3c3e, 0x1111, 0x1111, 0x1111, 0x1111},
       {0x7d74, 0x7970, 0x756b, 0x7168, 0x6d65, 0x6962, 0x655e, 0x615a},
       {0x5d54, 0x5950, 0x554f, 0x514c, 0x4d44, 0x4941, 0x453d, 0x4139},
       {0xbbbb, 0xa9bb, 0xbbbb, 0xa9bb, 0xbbbb, 0xa9bb, 0xbbbb, 0xa9bb},
       {0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0d81'0d81, 0x0000'0000, 0x0d80'cccd, 0x0000'0000},
       {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
       {0x4446'484a, 0x3c3e'4041, 0x1111'1111, 0x1111'1111},
       {0x7970'7d74, 0x7168'756b, 0x6962'6d65, 0x615a'655e},
       {0x5950'5d54, 0x514c'554f, 0x4941'4d44, 0x4139'453d},
       {0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb},
       {0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x1111'1110'51c5'51c6, 0x1111'1110'51c5'1112},
       {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
       {0x3c3e'4041'4446'484a, 0x1111'1111'1111'1111},
       {0x7168'756b'7970'7d74, 0x615a'655e'6962'6d65},
       {0x514c'554f'5950'5d54, 0x4139'453d'4941'4d44},
       {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
       {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9}},
      kVectorComparisonSource);

  TestVectorInstruction(
      0x8900e457,  // vremu.vx v8, v16, x1, v0.t
      {{5, 70, 5, 70, 5, 70, 5, 70, 64, 64, 64, 64, 85, 85, 85, 85},
       {10, 64, 10, 64, 85, 85, 85, 85, 0, 0, 10, 64, 85, 85, 85, 85},
       {22, 64, 22, 64, 22, 64, 22, 64, 0, 0, 0, 0, 0, 128, 22, 64},
       {0, 0, 0, 0, 0, 0, 0, 0, 17, 17, 17, 17, 17, 17, 17, 17},
       {74, 85, 74, 85, 74, 85, 74, 85, 76, 85, 76, 85, 76, 85, 76, 85},
       {74, 85, 74, 85, 78, 85, 78, 85, 75, 85, 75, 85, 75, 85, 75, 85},
       {17, 17, 17, 169, 17, 17, 17, 169, 17, 17, 17, 169, 17, 17, 17, 169},
       {169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169}},
      {{0x455b, 0x455b, 0x455b, 0x455b, 0x4040, 0x4040, 0x5555, 0x5555},
       {0x40b4, 0x40b4, 0x5555, 0x5555, 0x0000, 0x40b4, 0x5555, 0x5555},
       {0x4016, 0x4016, 0x4016, 0x4016, 0x0000, 0x0000, 0x8000, 0x4016},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x1111, 0x1111, 0x1111, 0x1111},
       {0x554a, 0x554a, 0x554a, 0x554a, 0x554c, 0x554c, 0x554c, 0x554c},
       {0x554a, 0x554a, 0x554e, 0x554e, 0x554b, 0x554b, 0x554b, 0x554b},
       {0x1111, 0xa9bb, 0x1111, 0xa9bb, 0x1111, 0xa9bb, 0x1111, 0xa9bb},
       {0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9, 0xa9a9}},
      {{0x455b'455b, 0x455b'455b, 0x4040'4040, 0x5555'5555},
       {0x40b4'40b4, 0x5555'5555, 0x40b4'0000, 0x5555'5555},
       {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
       {0x0000'0000, 0x0000'0000, 0x1111'1111, 0x1111'1111},
       {0x554a'554a, 0x554a'554a, 0x554c'554c, 0x554c'554c},
       {0x554a'554a, 0x554e'554e, 0x554b'554b, 0x554b'554b},
       {0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb, 0xa9bb'bbbb},
       {0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9, 0xa9a9'a9a9}},
      {{0x455b'455b'455b'455b, 0x5555'5554'9595'9596},
       {0x5555'5554'9609'960a, 0x5555'5554'9609'5556},
       {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
       {0x0000'0000'0000'0000, 0x1111'1111'1111'1111},
       {0x554a'554a'554a'554a, 0x554c'554c'554c'554c},
       {0x554e'554e'554a'554a, 0x554b'554b'554b'554b},
       {0xa9bb'bbbb'a9bb'bbbb, 0xa9bb'bbbb'a9bb'bbbb},
       {0xa9a9'a9a9'a9a9'a9a9, 0xa9a9'a9a9'a9a9'a9a9}},
      kVectorComparisonSource);

  TestVectorInstruction(
      0x8d0c2457,  // vrem.vv v8, v16, v24, v0.t
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {248, 13, 248, 13, 255, 255, 255, 255, 0, 0, 248, 13, 255, 255, 255, 255},
       {7, 0, 7, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0},
       {170, 170, 170, 170, 170, 170, 170, 170, 17, 17, 17, 17, 17, 17, 17, 17},
       {244, 255, 244, 255, 244, 255, 244, 255, 246, 255, 246, 255, 246, 255, 246, 255},
       {244, 255, 244, 255, 248, 255, 248, 255, 245, 255, 245, 255, 245, 255, 245, 255},
       {251, 249, 247, 227, 242, 241, 239, 219, 234, 233, 231, 253, 227, 225, 223, 237},
       {233, 229, 253, 247, 238, 235, 249, 241, 244, 253, 253, 249, 249, 253, 253, 255}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0d81, 0x0d81, 0xffff, 0xffff, 0x0000, 0x0d81, 0xffff, 0xffff},
       {0x000b, 0x000b, 0x000b, 0x000b, 0x0000, 0x0000, 0xfff8, 0x000b},
       {0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0x1111, 0x1111, 0x1111, 0x1111},
       {0xfff4, 0xfff4, 0xfff4, 0xfff4, 0xfff6, 0xfff6, 0xfff6, 0xfff6},
       {0xfff4, 0xfff4, 0xfff8, 0xfff8, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xf8fb, 0xe2f7, 0xf0f2, 0xdaef, 0xe8ea, 0xfc13, 0xe0e3, 0xec03},
       {0xe3e9, 0xf4fd, 0xfe05, 0xff0d, 0xf803, 0xfb15, 0xff31, 0xfffd}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0d81'0d81, 0xffff'ffff, 0x0d80'cccd, 0xffff'ffff},
       {0x000b'fb79, 0x000b'fb79, 0x0000'0000, 0x000c'3b63},
       {0xaaaa'aaaa, 0xaaaa'aaaa, 0x1111'1111, 0x1111'1111},
       {0xfff4'fff4, 0xfff4'fff4, 0xfff6'fff6, 0xfff6'fff6},
       {0xfff4'fff4, 0xfff8'fff8, 0xfff5'fff5, 0xfff5'fff5},
       {0xe2f6'f8fb, 0xdaee'f0f2, 0xfc12'1619, 0xec02'060b},
       {0xf4fb'0109, 0xff09'131c, 0xfb0d'1f30, 0xffaa'5551}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0xffff'ffff'40b4'40b4, 0xffff'ffff'40b4'0000},
       {0x000c'000c'000b'fb79, 0x000c'3ff5'bff5'bb63},
       {0xaaaa'aaaa'aaaa'aaaa, 0x1111'1111'1111'1111},
       {0xfff4'fff4'fff4'fff4, 0xfff6'fff6'fff6'fff6},
       {0xfff8'fff8'fff4'fff4, 0xfff5'fff5'fff5'fff5},
       {0xdaee'f0f1'e2f6'f8fb, 0xec02'0609'fc12'1619},
       {0xff09'1318'2731'3b49, 0xffaa'54ff'aa54'ffa4}},
      kVectorComparisonSource);

  TestVectorInstruction(
      0x8d00e457,  // vrem.vx v8, v16, x1, v0.t
      {{5, 240, 5, 240, 5, 240, 5, 240, 64, 64, 64, 64, 255, 255, 255, 255},
       {180, 64, 180, 64, 255, 255, 255, 255, 0, 0, 180, 64, 255, 255, 255, 255},
       {22, 64, 22, 64, 22, 64, 22, 64, 0, 0, 0, 0, 0, 214, 22, 64},
       {0, 0, 0, 0, 0, 0, 0, 0, 17, 17, 17, 17, 17, 17, 17, 17},
       {244, 255, 244, 255, 244, 255, 244, 255, 246, 255, 246, 255, 246, 255, 246, 255},
       {244, 255, 244, 255, 248, 255, 248, 255, 245, 255, 245, 255, 245, 255, 245, 255},
       {187, 187, 187, 255, 187, 187, 187, 255, 187, 187, 187, 255, 187, 187, 187, 255},
       {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}},
      {{0xf005, 0xf005, 0xf005, 0xf005, 0x4040, 0x4040, 0xffff, 0xffff},
       {0x40b4, 0x40b4, 0xffff, 0xffff, 0x0000, 0x40b4, 0xffff, 0xffff},
       {0x4016, 0x4016, 0x4016, 0x4016, 0x0000, 0x0000, 0xd556, 0x4016},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x1111, 0x1111, 0x1111, 0x1111},
       {0xfff4, 0xfff4, 0xfff4, 0xfff4, 0xfff6, 0xfff6, 0xfff6, 0xfff6},
       {0xfff4, 0xfff4, 0xfff8, 0xfff8, 0xfff5, 0xfff5, 0xfff5, 0xfff5},
       {0xbbbb, 0xff11, 0xbbbb, 0xff11, 0xbbbb, 0xff11, 0xbbbb, 0xff11},
       {0xfeff, 0xfeff, 0xfeff, 0xfeff, 0xfeff, 0xfeff, 0xfeff, 0xfeff}},
      {{0xf005'f005, 0xf005'f005, 0x4040'4040, 0xffff'ffff},
       {0x40b4'40b4, 0xffff'ffff, 0x40b4'0000, 0xffff'ffff},
       {0x4016'4016, 0x4016'4016, 0x0000'0000, 0x4016'8000},
       {0x0000'0000, 0x0000'0000, 0x1111'1111, 0x1111'1111},
       {0xfff4'fff4, 0xfff4'fff4, 0xfff6'fff6, 0xfff6'fff6},
       {0xfff4'fff4, 0xfff8'fff8, 0xfff5'fff5, 0xfff5'fff5},
       {0xff11'1111, 0xff11'1111, 0xff11'1111, 0xff11'1111},
       {0xfefe'feff, 0xfefe'feff, 0xfefe'feff, 0xfefe'feff}},
      {{0xf005'f005'f005'f005, 0xffff'ffff'4040'4040},
       {0xffff'ffff'40b4'40b4, 0xffff'ffff'40b4'0000},
       {0x4016'4016'4016'4016, 0x4016'8000'0000'0000},
       {0x0000'0000'0000'0000, 0x1111'1111'1111'1111},
       {0xfff4'fff4'fff4'fff4, 0xfff6'fff6'fff6'fff6},
       {0xfff8'fff8'fff4'fff4, 0xfff5'fff5'fff5'fff5},
       {0xff11'1110'ff11'1111, 0xff11'1110'ff11'1111},
       {0xfefe'fefe'fefe'feff, 0xfefe'fefe'fefe'feff}},
      kVectorComparisonSource);
}

TEST_F(Riscv64InterpreterTest, TestVslideup) {
  // With slide offset equal zero, this is equivalent to Vmv.
  TestVectorInstruction(
      0x39803457,  // vslideup.vi v8, v24, 0, v0.t
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
      kVectorCalculationsSourceLegacy);

  // VLMUL = 0.
  TestVectorPermutationInstruction(
      0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
      {{85, 0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28}, {}, {}, {}, {}, {}, {}, {}},
      {{0x5555, 0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18},
       {},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x5555'5555, 0x0604'0200, 0x0e0c'0a09, 0x1614'1211}, {}, {}, {}, {}, {}, {}, {}},
      {{0x5555'5555'5555'5555, 0x0e0c'0a09'0604'0200}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/0,
      /*regx1=*/1,
      /*skip=*/1);
  TestVectorPermutationInstruction(
      0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
      {{85, 85, 85, 85, 85, 85, 85, 85, 0, 2, 4, 6, 9, 10, 12, 14}, {}, {}, {}, {}, {}, {}, {}},
      {{0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555},
       {},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555}, {}, {}, {}, {}, {}, {}, {}},
      {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/0,
      /*regx1=*/8,
      /*skip=*/8);

  // VLMUL = 1
  TestVectorPermutationInstruction(
      0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
      {{85, 0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28},
       {30, 32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x5555, 0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18},
       {0x1e1c, 0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x5555'5555, 0x0604'0200, 0x0e0c'0a09, 0x1614'1211},
       {0x1e1c'1a18, 0x2624'2220, 0x2e2c'2a29, 0x3634'3231},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x5555'5555'5555'5555, 0x0e0c'0a09'0604'0200},
       {0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {},
       {},
       {},
       {},
       {},
       {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/1,
      /*regx1=*/1,
      /*skip=*/1);
  TestVectorPermutationInstruction(
      0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
      {{85, 85, 85, 85, 85, 85, 85, 85, 0, 2, 4, 6, 9, 10, 12, 14},
       {17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555},
       {0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {},
       {},
       {},
       {},
       {},
       {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/1,
      /*regx1=*/8,
      /*skip=*/8);

  // VLMUL = 2
  TestVectorPermutationInstruction(
      0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
      {{85, 0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28},
       {30, 32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60},
       {62, 64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92},
       {94, 96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124},
       {},
       {},
       {},
       {}},
      {{0x5555, 0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18},
       {0x1e1c, 0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38},
       {0x3e3c, 0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58},
       {0x5e5c, 0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78},
       {},
       {},
       {},
       {}},
      {{0x5555'5555, 0x0604'0200, 0x0e0c'0a09, 0x1614'1211},
       {0x1e1c'1a18, 0x2624'2220, 0x2e2c'2a29, 0x3634'3231},
       {0x3e3c'3a38, 0x4644'4240, 0x4e4c'4a49, 0x5654'5251},
       {0x5e5c'5a58, 0x6664'6260, 0x6e6c'6a69, 0x7674'7271},
       {},
       {},
       {},
       {}},
      {{0x5555'5555'5555'5555, 0x0e0c'0a09'0604'0200},
       {0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {},
       {},
       {},
       {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/2,
      /*regx1=*/1,
      /*skip=*/1);

  TestVectorPermutationInstruction(
      0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
      {{85, 85, 85, 85, 85, 85, 85, 85, 0, 2, 4, 6, 9, 10, 12, 14},
       {17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 73, 74, 76, 78},
       {81, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 105, 106, 108, 110},
       {},
       {},
       {},
       {}},
      {{0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555},
       {0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {},
       {},
       {},
       {}},
      {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {},
       {},
       {},
       {}},
      {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {},
       {},
       {},
       {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/2,
      /*regx1=*/8,
      /*skip=*/8);

  // VLMUL = 3
  TestVectorPermutationInstruction(
      0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
      {{85, 0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28},
       {30, 32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60},
       {62, 64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92},
       {94, 96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124},
       {126, 128, 130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156},
       {158, 160, 162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188},
       {190, 192, 194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220},
       {222, 224, 226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252}},
      {{0x5555, 0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18},
       {0x1e1c, 0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38},
       {0x3e3c, 0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58},
       {0x5e5c, 0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78},
       {0x7e7c, 0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98},
       {0x9e9c, 0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8},
       {0xbebc, 0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8},
       {0xdedc, 0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8}},
      {{0x5555'5555, 0x0604'0200, 0x0e0c'0a09, 0x1614'1211},
       {0x1e1c'1a18, 0x2624'2220, 0x2e2c'2a29, 0x3634'3231},
       {0x3e3c'3a38, 0x4644'4240, 0x4e4c'4a49, 0x5654'5251},
       {0x5e5c'5a58, 0x6664'6260, 0x6e6c'6a69, 0x7674'7271},
       {0x7e7c'7a78, 0x8684'8280, 0x8e8c'8a89, 0x9694'9291},
       {0x9e9c'9a98, 0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1},
       {0xbebc'bab8, 0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1},
       {0xdedc'dad8, 0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1}},
      {{0x5555'5555'5555'5555, 0x0e0c'0a09'0604'0200},
       {0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0x8e8c'8a89'8684'8280},
       {0x9e9c'9a98'9694'9291, 0xaeac'aaa9'a6a4'a2a0},
       {0xbebc'bab8'b6b4'b2b1, 0xcecc'cac9'c6c4'c2c0},
       {0xdedc'dad8'd6d4'd2d1, 0xeeec'eae9'e6e4'e2e0}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/3,
      /*regx1=*/1,
      /*skip=*/1);

  TestVectorPermutationInstruction(
      0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
      {{85, 85, 85, 85, 85, 85, 85, 85, 0, 2, 4, 6, 9, 10, 12, 14},
       {17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 73, 74, 76, 78},
       {81, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 105, 106, 108, 110},
       {113, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 137, 138, 140, 142},
       {145, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 169, 170, 172, 174},
       {177, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 201, 202, 204, 206},
       {209, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 233, 234, 236, 238}},
      {{0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555},
       {0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c},
       {0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc}},
      {{0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x5555'5555, 0x5555'5555, 0x5555'5555, 0x5555'5555},
       {0x0604'0200, 0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18},
       {0x2624'2220, 0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38},
       {0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8}},
      {{0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x5555'5555'5555'5555, 0x5555'5555'5555'5555},
       {0x0e0c'0a09'0604'0200, 0x1e1c'1a18'1614'1211},
       {0x2e2c'2a29'2624'2220, 0x3e3c'3a38'3634'3231},
       {0x4e4c'4a49'4644'4240, 0x5e5c'5a58'5654'5251},
       {0x6e6c'6a69'6664'6260, 0x7e7c'7a78'7674'7271}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/3,
      /*regx1=*/8,
      /*skip=*/8);

  // VLMUL = 4
  TestVectorPermutationInstruction(0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/4,
                                   /*regx1=*/1,
                                   /*skip=*/1);

  TestVectorPermutationInstruction(0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/4,
                                   /*regx1=*/8,
                                   /*skip=*/8);

  // VLMUL = 5
  TestVectorPermutationInstruction(0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
                                   {{85, 0}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/5,
                                   /*regx1=*/1,
                                   /*skip=*/1);

  TestVectorPermutationInstruction(0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
                                   {{85, 85}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/5,
                                   /*regx1=*/8,
                                   /*skip=*/8);

  // VLMUL = 6
  TestVectorPermutationInstruction(0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
                                   {{85, 0, 2, 4}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555, 0x0200}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555'5555}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/6,
                                   /*regx1=*/1,
                                   /*skip=*/1);

  TestVectorPermutationInstruction(0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
                                   {{85, 85, 85, 85}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555, 0x5555}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555'5555}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/6,
                                   /*regx1=*/8,
                                   /*skip=*/8);

  // VLMUL = 7
  TestVectorPermutationInstruction(0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
                                   {{85, 0, 2, 4, 6, 9, 10, 12}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555, 0x0200, 0x0604, 0x0a09}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555'5555, 0x0604'0200}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555'5555'5555'5555}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/7,
                                   /*regx1=*/1,
                                   /*skip=*/1);

  TestVectorPermutationInstruction(0x3980c457,  // vslideup.vx v8, v24, x1, v0.t
                                   {{85, 85, 85, 85, 85, 85, 85, 85}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555, 0x5555, 0x5555, 0x5555}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555'5555, 0x5555'5555}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x5555'5555'5555'5555}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/7,
                                   /*regx1=*/8,
                                   /*skip=*/8);
}

TEST_F(Riscv64InterpreterTest, TestVslidedown) {
  // With slide offset equal zero, this is equivalent to Vmv.
  TestVectorInstruction(
      0x3d803457,  // vslidedown.vi v8, v24, 0, v0.t
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
      kVectorCalculationsSourceLegacy);

  // VLMUL = 0.
  TestVectorPermutationInstruction(
      0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0x1e1c'1a18'1614'1211, 0}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/0,
      /*regx1=*/1,
      /*skip=*/0);

  TestVectorPermutationInstruction(
      0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
      {{17, 18, 20, 22, 24, 26, 28, 30, 0, 0, 0, 0, 0, 0, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0, 0, 0, 0, 0, 0, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0, 0, 0, 0}, {}, {}, {}, {}, {}, {}, {}},
      {{0, 0}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/0,
      /*regx1=*/8,
      /*skip=*/0);

  // VLMUL = 1
  TestVectorPermutationInstruction(
      0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 0},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0},
       {},
       {},
       {},
       {},
       {},
       {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/1,
      /*regx1=*/1,
      /*skip=*/0);
  TestVectorPermutationInstruction(
      0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
      {{17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {49, 50, 52, 54, 56, 58, 60, 62, 0, 0, 0, 0, 0, 0, 0, 0},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {}, {}, {}, {}, {}, {}},
      {{0, 0}, {0, 0}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/1,
      /*regx1=*/8,
      /*skip=*/0);

  // VLMUL = 2
  TestVectorPermutationInstruction(
      0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 64},
       {66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94, 96},
       {98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126, 0},
       {},
       {},
       {},
       {}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0x4240},
       {0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c, 0x6260},
       {0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c, 0x0000},
       {},
       {},
       {},
       {}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0x4644'4240},
       {0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58, 0x6664'6260},
       {0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78, 0x0000'0000},
       {},
       {},
       {},
       {}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0x0000'0000'0000'0000},
       {},
       {},
       {},
       {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/2,
      /*regx1=*/1,
      /*skip=*/0);

  TestVectorPermutationInstruction(
      0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
      {{17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 73, 74, 76, 78},
       {81, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 105, 106, 108, 110},
       {113, 114, 116, 118, 120, 122, 124, 126, 0, 0, 0, 0, 0, 0, 0, 0},
       {},
       {},
       {},
       {}},
      {{0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {},
       {},
       {},
       {}},
      {{0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {},
       {},
       {},
       {}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {},
       {},
       {},
       {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/2,
      /*regx1=*/8,
      /*skip=*/0);

  // VLMUL = 3
  TestVectorPermutationInstruction(
      0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 64},
       {66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94, 96},
       {98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126, 128},
       {130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156, 158, 160},
       {162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188, 190, 192},
       {194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220, 222, 224},
       {226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252, 254, 0}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0x4240},
       {0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c, 0x6260},
       {0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c, 0x8280},
       {0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c, 0xa2a0},
       {0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc, 0xc2c0},
       {0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc, 0xe2e0},
       {0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc, 0x0000}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0x4644'4240},
       {0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58, 0x6664'6260},
       {0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78, 0x8684'8280},
       {0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98, 0xa6a4'a2a0},
       {0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8, 0xc6c4'c2c0},
       {0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8, 0xe6e4'e2e0},
       {0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8, 0x0000'0000}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0x8e8c'8a89'8684'8280},
       {0x9e9c'9a98'9694'9291, 0xaeac'aaa9'a6a4'a2a0},
       {0xbebc'bab8'b6b4'b2b1, 0xcecc'cac9'c6c4'c2c0},
       {0xdedc'dad8'd6d4'd2d1, 0xeeec'eae9'e6e4'e2e0},
       {0xfefc'faf8'f6f4'f2f1, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/3,
      /*regx1=*/1,
      /*skip=*/0);

  TestVectorPermutationInstruction(
      0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
      {{17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 42, 44, 46},
       {49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 73, 74, 76, 78},
       {81, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 105, 106, 108, 110},
       {113, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 137, 138, 140, 142},
       {145, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 169, 170, 172, 174},
       {177, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 201, 202, 204, 206},
       {209, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 233, 234, 236, 238},
       {241, 242, 244, 246, 248, 250, 252, 254, 0, 0, 0, 0, 0, 0, 0, 0}},
      {{0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c},
       {0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c},
       {0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c},
       {0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c},
       {0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc},
       {0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc},
       {0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {{0x4644'4240, 0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58},
       {0x6664'6260, 0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78},
       {0x8684'8280, 0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98},
       {0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8},
       {0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8},
       {0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {{0x8e8c'8a89'8684'8280, 0x9e9c'9a98'9694'9291},
       {0xaeac'aaa9'a6a4'a2a0, 0xbebc'bab8'b6b4'b2b1},
       {0xcecc'cac9'c6c4'c2c0, 0xdedc'dad8'd6d4'd2d1},
       {0xeeec'eae9'e6e4'e2e0, 0xfefc'faf8'f6f4'f2f1},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/3,
      /*regx1=*/8,
      /*skip=*/0);

  // VLMUL = 4
  TestVectorPermutationInstruction(0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/4,
                                   /*regx1=*/1,
                                   /*skip=*/0);

  TestVectorPermutationInstruction(0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/4,
                                   /*regx1=*/8,
                                   /*skip=*/0);

  // VLMUL = 5
  TestVectorPermutationInstruction(0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
                                   {{2, 4}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0604}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/5,
                                   /*regx1=*/1,
                                   /*skip=*/0);

  TestVectorPermutationInstruction(0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
                                   {{17, 18}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0000}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/5,
                                   /*regx1=*/8,
                                   /*skip=*/0);

  // VLMUL = 6
  TestVectorPermutationInstruction(0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
                                   {{2, 4, 6, 9}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0604, 0x0a09}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0e0c'0a09}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/6,
                                   /*regx1=*/1,
                                   /*skip=*/0);

  TestVectorPermutationInstruction(0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
                                   {{17, 18, 20, 22}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0000, 0x0000}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0000'0000}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/6,
                                   /*regx1=*/8,
                                   /*skip=*/0);

  // VLMUL = 7
  TestVectorPermutationInstruction(0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
                                   {{2, 4, 6, 9, 10, 12, 14, 17}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0604, 0x0a09, 0x0e0c, 0x1211}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0e0c'0a09, 0x1614'1211}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x1e1c'1a18'1614'1211}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/7,
                                   /*regx1=*/1,
                                   /*skip=*/0);

  TestVectorPermutationInstruction(0x3d80c457,  // vslidedown.vx v8, v24, x1, v0.t
                                   {{17, 18, 20, 22, 24, 26, 28, 30}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0000, 0x0000, 0x0000, 0x0000}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0000'0000, 0x0000'0000}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0000'0000'0000'0000}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/7,
                                   /*regx1=*/8,
                                   /*skip=*/0);
}

TEST_F(Riscv64InterpreterTest, TestVslide1up) {
  TestVectorInstruction(
      0x3980e457,  // vslide1up.vx v8, v24, x1, v0.t
      {{0xaa, 0, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28},
       {30, 32, 34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60},
       {62, 64, 66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92},
       {94, 96, 98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124},
       {126, 128, 130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156},
       {158, 160, 162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188},
       {190, 192, 194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220},
       {222, 224, 226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252}},
      {{0xaaaa, 0x0200, 0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18},
       {0x1e1c, 0x2220, 0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38},
       {0x3e3c, 0x4240, 0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58},
       {0x5e5c, 0x6260, 0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78},
       {0x7e7c, 0x8280, 0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98},
       {0x9e9c, 0xa2a0, 0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8},
       {0xbebc, 0xc2c0, 0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8},
       {0xdedc, 0xe2e0, 0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8}},
      {{0xaaaa'aaaa, 0x0604'0200, 0x0e0c'0a09, 0x1614'1211},
       {0x1e1c'1a18, 0x2624'2220, 0x2e2c'2a29, 0x3634'3231},
       {0x3e3c'3a38, 0x4644'4240, 0x4e4c'4a49, 0x5654'5251},
       {0x5e5c'5a58, 0x6664'6260, 0x6e6c'6a69, 0x7674'7271},
       {0x7e7c'7a78, 0x8684'8280, 0x8e8c'8a89, 0x9694'9291},
       {0x9e9c'9a98, 0xa6a4'a2a0, 0xaeac'aaa9, 0xb6b4'b2b1},
       {0xbebc'bab8, 0xc6c4'c2c0, 0xcecc'cac9, 0xd6d4'd2d1},
       {0xdedc'dad8, 0xe6e4'e2e0, 0xeeec'eae9, 0xf6f4'f2f1}},
      {{0xaaaa'aaaa'aaaa'aaaa, 0x0e0c'0a09'0604'0200},
       {0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0x8e8c'8a89'8684'8280},
       {0x9e9c'9a98'9694'9291, 0xaeac'aaa9'a6a4'a2a0},
       {0xbebc'bab8'b6b4'b2b1, 0xcecc'cac9'c6c4'c2c0},
       {0xdedc'dad8'd6d4'd2d1, 0xeeec'eae9'e6e4'e2e0}},
      kVectorCalculationsSourceLegacy);
}

TEST_F(Riscv64InterpreterTest, TestVslide1down) {
  // Where the element at the top gets inserted will depend on VLMUL so we use
  // TestVectorPermutationInstruction instead of TestVectorInstruction.

  // VLMUL = 0
  TestVectorPermutationInstruction(
      0x3d80e457,  // vslide1down.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 0xaa}, {}, {}, {}, {}, {}, {}, {}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0xaaaa},
       {},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0xaaaa'aaaa}, {}, {}, {}, {}, {}, {}, {}},
      {{0x1e1c'1a18'1614'1211, 0xaaaa'aaaa'aaaa'aaaa}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/0,
      /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_x1=*/true);

  // VLMUL = 1
  TestVectorPermutationInstruction(
      0x3d80e457,  // vslide1down.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 0xaa},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0xaaaa},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0xaaaa'aaaa},
       {},
       {},
       {},
       {},
       {},
       {}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0xaaaa'aaaa'aaaa'aaaa},
       {},
       {},
       {},
       {},
       {},
       {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/1,
      /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_x1=*/true);

  // VLMUL = 2
  TestVectorPermutationInstruction(
      0x3d80e457,  // vslide1down.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 64},
       {66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94, 96},
       {98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126, 0xaa},
       {},
       {},
       {},
       {}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0x4240},
       {0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c, 0x6260},
       {0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c, 0xaaaa},
       {},
       {},
       {},
       {}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0x4644'4240},
       {0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58, 0x6664'6260},
       {0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78, 0xaaaa'aaaa},
       {},
       {},
       {},
       {}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0xaaaa'aaaa'aaaa'aaaa},
       {},
       {},
       {},
       {}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/2,
      /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_x1=*/true);

  // VLMUL = 3
  TestVectorPermutationInstruction(
      0x3d80e457,  // vslide1down.vx v8, v24, x1, v0.t
      {{2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32},
       {34, 36, 38, 41, 42, 44, 46, 49, 50, 52, 54, 56, 58, 60, 62, 64},
       {66, 68, 70, 73, 74, 76, 78, 81, 82, 84, 86, 88, 90, 92, 94, 96},
       {98, 100, 102, 105, 106, 108, 110, 113, 114, 116, 118, 120, 122, 124, 126, 128},
       {130, 132, 134, 137, 138, 140, 142, 145, 146, 148, 150, 152, 154, 156, 158, 160},
       {162, 164, 166, 169, 170, 172, 174, 177, 178, 180, 182, 184, 186, 188, 190, 192},
       {194, 196, 198, 201, 202, 204, 206, 209, 210, 212, 214, 216, 218, 220, 222, 224},
       {226, 228, 230, 233, 234, 236, 238, 241, 242, 244, 246, 248, 250, 252, 254, 0xaa}},
      {{0x0604, 0x0a09, 0x0e0c, 0x1211, 0x1614, 0x1a18, 0x1e1c, 0x2220},
       {0x2624, 0x2a29, 0x2e2c, 0x3231, 0x3634, 0x3a38, 0x3e3c, 0x4240},
       {0x4644, 0x4a49, 0x4e4c, 0x5251, 0x5654, 0x5a58, 0x5e5c, 0x6260},
       {0x6664, 0x6a69, 0x6e6c, 0x7271, 0x7674, 0x7a78, 0x7e7c, 0x8280},
       {0x8684, 0x8a89, 0x8e8c, 0x9291, 0x9694, 0x9a98, 0x9e9c, 0xa2a0},
       {0xa6a4, 0xaaa9, 0xaeac, 0xb2b1, 0xb6b4, 0xbab8, 0xbebc, 0xc2c0},
       {0xc6c4, 0xcac9, 0xcecc, 0xd2d1, 0xd6d4, 0xdad8, 0xdedc, 0xe2e0},
       {0xe6e4, 0xeae9, 0xeeec, 0xf2f1, 0xf6f4, 0xfaf8, 0xfefc, 0xaaaa}},
      {{0x0e0c'0a09, 0x1614'1211, 0x1e1c'1a18, 0x2624'2220},
       {0x2e2c'2a29, 0x3634'3231, 0x3e3c'3a38, 0x4644'4240},
       {0x4e4c'4a49, 0x5654'5251, 0x5e5c'5a58, 0x6664'6260},
       {0x6e6c'6a69, 0x7674'7271, 0x7e7c'7a78, 0x8684'8280},
       {0x8e8c'8a89, 0x9694'9291, 0x9e9c'9a98, 0xa6a4'a2a0},
       {0xaeac'aaa9, 0xb6b4'b2b1, 0xbebc'bab8, 0xc6c4'c2c0},
       {0xcecc'cac9, 0xd6d4'd2d1, 0xdedc'dad8, 0xe6e4'e2e0},
       {0xeeec'eae9, 0xf6f4'f2f1, 0xfefc'faf8, 0xaaaa'aaaa}},
      {{0x1e1c'1a18'1614'1211, 0x2e2c'2a29'2624'2220},
       {0x3e3c'3a38'3634'3231, 0x4e4c'4a49'4644'4240},
       {0x5e5c'5a58'5654'5251, 0x6e6c'6a69'6664'6260},
       {0x7e7c'7a78'7674'7271, 0x8e8c'8a89'8684'8280},
       {0x9e9c'9a98'9694'9291, 0xaeac'aaa9'a6a4'a2a0},
       {0xbebc'bab8'b6b4'b2b1, 0xcecc'cac9'c6c4'c2c0},
       {0xdedc'dad8'd6d4'd2d1, 0xeeec'eae9'e6e4'e2e0},
       {0xfefc'faf8'f6f4'f2f1, 0xaaaa'aaaa'aaaa'aaaa}},
      kVectorCalculationsSourceLegacy,
      /*vlmul=*/3,
      /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_x1=*/true);

  // VLMUL = 4
  TestVectorPermutationInstruction(0x3d80e457,  // vslide1down.vx v8, v24, x1, v0.t
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/4,
                                   /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
                                   /*skip=*/0,
                                   /*ignore_vma_for_last=*/true,
                                   /*last_elem_is_x1=*/true);

  // VLMUL = 5
  TestVectorPermutationInstruction(0x3d80e457,  // vslide1down.vx v8, v24, x1, v0.t
                                   {{2, 0xaa}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0xaaaa}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/5,
                                   /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
                                   /*skip=*/0,
                                   /*ignore_vma_for_last=*/true,
                                   /*last_elem_is_x1=*/true);

  // VLMUL = 6
  TestVectorPermutationInstruction(0x3d80e457,  // vslide1down.vx v8, v24, x1, v0.t
                                   {{2, 4, 6, 0xaa}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0604, 0xaaaa}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0xaaaa'aaaa}, {}, {}, {}, {}, {}, {}, {}},
                                   {{}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/6,
                                   /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
                                   /*skip=*/0,
                                   /*ignore_vma_for_last=*/true,
                                   /*last_elem_is_x1=*/true);

  // VLMUL = 7
  TestVectorPermutationInstruction(0x3d80e457,  // vslide1down.vx v8, v24, x1, v0.t
                                   {{2, 4, 6, 9, 10, 12, 14, 0xaa}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0604, 0x0a09, 0x0e0c, 0xaaaa}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0x0e0c'0a09, 0xaaaa'aaaa}, {}, {}, {}, {}, {}, {}, {}},
                                   {{0xaaaa'aaaa'aaaa'aaaa}, {}, {}, {}, {}, {}, {}, {}},
                                   kVectorCalculationsSourceLegacy,
                                   /*vlmul=*/7,
                                   /*regx1=*/0xaaaa'aaaa'aaaa'aaaa,
                                   /*skip=*/0,
                                   /*ignore_vma_for_last=*/true,
                                   /*last_elem_is_x1=*/true);
}

TEST_F(Riscv64InterpreterTest, TestVfslide1up) {
  TestVectorFloatInstruction(0x3980d457,  // vfslide1up.vf v8, v24, f1, v0.t
                             {{0x40b4'0000, 0x9604'9200, 0x9e0c'9a09, 0x8614'8211},
                              {0x8e1c'8a18, 0xb624'b220, 0xbe2c'ba29, 0xa634'a231},
                              {0xae3c'aa38, 0xd644'd240, 0xde4c'da49, 0xc654'c251},
                              {0xce5c'ca58, 0xf664'f260, 0xfe6c'fa69, 0xe674'e271},
                              {0xee7c'ea78, 0x1684'1280, 0x1e8c'1a89, 0x0694'0291},
                              {0x0e9c'0a98, 0x36a4'32a0, 0x3eac'3aa9, 0x26b4'22b1},
                              {0x2ebc'2ab8, 0x56c4'52c0, 0x5ecc'5ac9, 0x46d4'42d1},
                              {0x4edc'4ad8, 0x76e4'72e0, 0x7eec'7ae9, 0x66f4'62f1}},
                             {{0x4016'8000'0000'0000, 0x9e0c'9a09'9604'9200},
                              {0x8e1c'8a18'8614'8211, 0xbe2c'ba29'b624'b220},
                              {0xae3c'aa38'a634'a231, 0xde4c'da49'd644'd240},
                              {0xce5c'ca58'c654'c251, 0xfe6c'fa69'f664'f260},
                              {0xee7c'ea78'e674'e271, 0x1e8c'1a89'1684'1280},
                              {0x0e9c'0a98'0694'0291, 0x3eac'3aa9'36a4'32a0},
                              {0x2ebc'2ab8'26b4'22b1, 0x5ecc'5ac9'56c4'52c0},
                              {0x4edc'4ad8'46d4'42d1, 0x7eec'7ae9'76e4'72e0}},
                             kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVfslide1down) {
  // Where the element at the top gets inserted will depend on VLMUL so we use
  // TestVectorFloatPermutationInstruction instead of TestVectorFloatInstruction.

  // VLMUL = 0
  TestVectorFloatPermutationInstruction(
      0x3d80d457,  // vfslide1down.vf v8, v24, f1, v0.t
      {{0x9e0c'9a09, 0x8614'8211, 0x8e1c'8a18, 0x40b4'0000}, {}, {}, {}, {}, {}, {}, {}},
      {{0x8e1c'8a18'8614'8211, 0x4016'8000'0000'0000}, {}, {}, {}, {}, {}, {}, {}},
      kVectorCalculationsSource,
      /*vlmul=*/0,
      /*skip=*/0,
      /*ignore_vma_for_last=*/true,
      /*last_elem_is_f1=*/true);

  // VLMUL = 1
  TestVectorFloatPermutationInstruction(0x3d80d457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{0x9e0c'9a09, 0x8614'8211, 0x8e1c'8a18, 0xb624'b220},
                                         {0xbe2c'ba29, 0xa634'a231, 0xae3c'aa38, 0x40b4'0000},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {}},
                                        {{0x8e1c'8a18'8614'8211, 0xbe2c'ba29'b624'b220},
                                         {0xae3c'aa38'a634'a231, 0x4016'8000'0000'0000},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/1,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 2
  TestVectorFloatPermutationInstruction(0x3d80d457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{0x9e0c'9a09, 0x8614'8211, 0x8e1c'8a18, 0xb624'b220},
                                         {0xbe2c'ba29, 0xa634'a231, 0xae3c'aa38, 0xd644'd240},
                                         {0xde4c'da49, 0xc654'c251, 0xce5c'ca58, 0xf664'f260},
                                         {0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78, 0x40b4'0000},
                                         {},
                                         {},
                                         {},
                                         {}},
                                        {{0x8e1c'8a18'8614'8211, 0xbe2c'ba29'b624'b220},
                                         {0xae3c'aa38'a634'a231, 0xde4c'da49'd644'd240},
                                         {0xce5c'ca58'c654'c251, 0xfe6c'fa69'f664'f260},
                                         {0xee7c'ea78'e674'e271, 0x4016'8000'0000'0000},
                                         {},
                                         {},
                                         {},
                                         {}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/2,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 3
  TestVectorFloatPermutationInstruction(0x3d80d457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{0x9e0c'9a09, 0x8614'8211, 0x8e1c'8a18, 0xb624'b220},
                                         {0xbe2c'ba29, 0xa634'a231, 0xae3c'aa38, 0xd644'd240},
                                         {0xde4c'da49, 0xc654'c251, 0xce5c'ca58, 0xf664'f260},
                                         {0xfe6c'fa69, 0xe674'e271, 0xee7c'ea78, 0x1684'1280},
                                         {0x1e8c'1a89, 0x0694'0291, 0x0e9c'0a98, 0x36a4'32a0},
                                         {0x3eac'3aa9, 0x26b4'22b1, 0x2ebc'2ab8, 0x56c4'52c0},
                                         {0x5ecc'5ac9, 0x46d4'42d1, 0x4edc'4ad8, 0x76e4'72e0},
                                         {0x7eec'7ae9, 0x66f4'62f1, 0x6efc'6af8, 0x40b4'0000}},
                                        {{0x8e1c'8a18'8614'8211, 0xbe2c'ba29'b624'b220},
                                         {0xae3c'aa38'a634'a231, 0xde4c'da49'd644'd240},
                                         {0xce5c'ca58'c654'c251, 0xfe6c'fa69'f664'f260},
                                         {0xee7c'ea78'e674'e271, 0x1e8c'1a89'1684'1280},
                                         {0x0e9c'0a98'0694'0291, 0x3eac'3aa9'36a4'32a0},
                                         {0x2ebc'2ab8'26b4'22b1, 0x5ecc'5ac9'56c4'52c0},
                                         {0x4edc'4ad8'46d4'42d1, 0x7eec'7ae9'76e4'72e0},
                                         {0x6efc'6af8'66f4'62f1, 0x4016'8000'0000'0000}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/3,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 4
  TestVectorFloatPermutationInstruction(0x3d80d457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{}, {}, {}, {}, {}, {}, {}, {}},
                                        {{}, {}, {}, {}, {}, {}, {}, {}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/4,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 5
  TestVectorFloatPermutationInstruction(0x3d80d457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{}, {}, {}, {}, {}, {}, {}, {}},
                                        {{}, {}, {}, {}, {}, {}, {}, {}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/5,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 6
  TestVectorFloatPermutationInstruction(0x3d80d457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{0x40b4'0000}, {}, {}, {}, {}, {}, {}, {}},
                                        {{}, {}, {}, {}, {}, {}, {}, {}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/6,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);

  // VLMUL = 7
  TestVectorFloatPermutationInstruction(0x3d80d457,  // vfslide1down.vf v8, v24, f1, v0.t
                                        {{0x9e0c'9a09, 0x40b4'0000}, {}, {}, {}, {}, {}, {}, {}},
                                        {{0x4016'8000'0000'0000}, {}, {}, {}, {}, {}, {}, {}},
                                        kVectorCalculationsSource,
                                        /*vlmul=*/7,
                                        /*skip=*/0,
                                        /*ignore_vma_for_last=*/true,
                                        /*last_elem_is_f1=*/true);
}

TEST_F(Riscv64InterpreterTest, TestVwadd) {
  TestWideningVectorInstruction(0xc50c2457,  // vwadd.vv v8,v16,v24,v0.t
                                {{0x0000, 0xff13, 0x0006, 0xff19, 0x000d, 0xff1f, 0x0012, 0xff25},
                                 {0x0019, 0xff0b, 0x001e, 0xff11, 0x0024, 0xff17, 0x002a, 0xff1d},
                                 {0x0030, 0xff43, 0x0036, 0xff49, 0x003d, 0xff4f, 0x0042, 0xff55},
                                 {0x0049, 0xff3b, 0x004e, 0xff41, 0x0054, 0xff47, 0x005a, 0xff4d},
                                 {0x0060, 0xff73, 0x0066, 0xff79, 0x006d, 0xff7f, 0x0072, 0xff85},
                                 {0x0079, 0xff6b, 0x007e, 0xff71, 0x0084, 0xff77, 0x008a, 0xff7d},
                                 {0x0090, 0xffa3, 0x0096, 0xffa9, 0x009d, 0xffaf, 0x00a2, 0xffb5},
                                 {0x00a9, 0xff9b, 0x00ae, 0xffa1, 0x00b4, 0xffa7, 0x00ba, 0xffad}},
                                {{0xffff'1300, 0xffff'1906, 0xffff'1f0d, 0xffff'2512},
                                 {0xffff'0b19, 0xffff'111e, 0xffff'1724, 0xffff'1d2a},
                                 {0xffff'4330, 0xffff'4936, 0xffff'4f3d, 0xffff'5542},
                                 {0xffff'3b49, 0xffff'414e, 0xffff'4754, 0xffff'4d5a},
                                 {0xffff'7360, 0xffff'7966, 0xffff'7f6d, 0xffff'8572},
                                 {0xffff'6b79, 0xffff'717e, 0xffff'7784, 0xffff'7d8a},
                                 {0xffff'a390, 0xffff'a996, 0xffff'af9d, 0xffff'b5a2},
                                 {0xffff'9ba9, 0xffff'a1ae, 0xffff'a7b4, 0xffff'adba}},
                                {{0xffff'ffff'1907'1300, 0xffff'ffff'2513'1f0d},
                                 {0xffff'ffff'111f'0b19, 0xffff'ffff'1d2b'1724},
                                 {0xffff'ffff'4937'4330, 0xffff'ffff'5543'4f3d},
                                 {0xffff'ffff'414f'3b49, 0xffff'ffff'4d5b'4754},
                                 {0xffff'ffff'7967'7360, 0xffff'ffff'8573'7f6d},
                                 {0xffff'ffff'717f'6b79, 0xffff'ffff'7d8b'7784},
                                 {0xffff'ffff'a997'a390, 0xffff'ffff'b5a3'af9d},
                                 {0xffff'ffff'a1af'9ba9, 0xffff'ffff'adbb'a7b4}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwaddu) {
  TestWideningVectorInstruction(0xc10c2457,  // vwaddu.vv v8, v16, v24, v0.t
                                {{0x0000, 0x0113, 0x0006, 0x0119, 0x000d, 0x011f, 0x0012, 0x0125},
                                 {0x0019, 0x010b, 0x001e, 0x0111, 0x0024, 0x0117, 0x002a, 0x011d},
                                 {0x0030, 0x0143, 0x0036, 0x0149, 0x003d, 0x014f, 0x0042, 0x0155},
                                 {0x0049, 0x013b, 0x004e, 0x0141, 0x0054, 0x0147, 0x005a, 0x014d},
                                 {0x0060, 0x0173, 0x0066, 0x0179, 0x006d, 0x017f, 0x0072, 0x0185},
                                 {0x0079, 0x016b, 0x007e, 0x0171, 0x0084, 0x0177, 0x008a, 0x017d},
                                 {0x0090, 0x01a3, 0x0096, 0x01a9, 0x009d, 0x01af, 0x00a2, 0x01b5},
                                 {0x00a9, 0x019b, 0x00ae, 0x01a1, 0x00b4, 0x01a7, 0x00ba, 0x01ad}},
                                {{0x0001'1300, 0x0001'1906, 0x0001'1f0d, 0x0001'2512},
                                 {0x0001'0b19, 0x0001'111e, 0x0001'1724, 0x0001'1d2a},
                                 {0x0001'4330, 0x0001'4936, 0x0001'4f3d, 0x0001'5542},
                                 {0x0001'3b49, 0x0001'414e, 0x0001'4754, 0x0001'4d5a},
                                 {0x0001'7360, 0x0001'7966, 0x0001'7f6d, 0x0001'8572},
                                 {0x0001'6b79, 0x0001'717e, 0x0001'7784, 0x0001'7d8a},
                                 {0x0001'a390, 0x0001'a996, 0x0001'af9d, 0x0001'b5a2},
                                 {0x0001'9ba9, 0x0001'a1ae, 0x0001'a7b4, 0x0001'adba}},
                                {{0x0000'0001'1907'1300, 0x0000'0001'2513'1f0d},
                                 {0x0000'0001'111f'0b19, 0x0000'0001'1d2b'1724},
                                 {0x0000'0001'4937'4330, 0x0000'0001'5543'4f3d},
                                 {0x0000'0001'414f'3b49, 0x0000'0001'4d5b'4754},
                                 {0x0000'0001'7967'7360, 0x0000'0001'8573'7f6d},
                                 {0x0000'0001'717f'6b79, 0x0000'0001'7d8b'7784},
                                 {0x0000'0001'a997'a390, 0x0000'0001'b5a3'af9d},
                                 {0x0000'0001'a1af'9ba9, 0x0000'0001'adbb'a7b4}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwaddwx) {
  TestWideningVectorInstruction(0xd500e457,  //  vwadd.wx v8,v16,ra,v0.t
                                {{0x80aa, 0x82ac, 0x84ae, 0x86b0, 0x88b2, 0x8ab4, 0x8cb6, 0x8eb8},
                                 {0x90ba, 0x92bc, 0x94be, 0x96c0, 0x98c2, 0x9ac4, 0x9cc6, 0x9ec8},
                                 {0xa0ca, 0xa2cc, 0xa4ce, 0xa6d0, 0xa8d2, 0xaad4, 0xacd6, 0xaed8},
                                 {0xb0da, 0xb2dc, 0xb4de, 0xb6e0, 0xb8e2, 0xbae4, 0xbce6, 0xbee8},
                                 {0xc0ea, 0xc2ec, 0xc4ee, 0xc6f0, 0xc8f2, 0xcaf4, 0xccf6, 0xcef8},
                                 {0xd0fa, 0xd2fc, 0xd4fe, 0xd700, 0xd902, 0xdb04, 0xdd06, 0xdf08},
                                 {0xe10a, 0xe30c, 0xe50e, 0xe710, 0xe912, 0xeb14, 0xed16, 0xef18},
                                 {0xf11a, 0xf31c, 0xf51e, 0xf720, 0xf922, 0xfb24, 0xfd26, 0xff28}},
                                {{0x8302'2baa, 0x8706'2fae, 0x8b0a'33b2, 0x8f0e'37b6},
                                 {0x9312'3bba, 0x9716'3fbe, 0x9b1a'43c2, 0x9f1e'47c6},
                                 {0xa322'4bca, 0xa726'4fce, 0xab2a'53d2, 0xaf2e'57d6},
                                 {0xb332'5bda, 0xb736'5fde, 0xbb3a'63e2, 0xbf3e'67e6},
                                 {0xc342'6bea, 0xc746'6fee, 0xcb4a'73f2, 0xcf4e'77f6},
                                 {0xd352'7bfa, 0xd756'7ffe, 0xdb5a'8402, 0xdf5e'8806},
                                 {0xe362'8c0a, 0xe766'900e, 0xeb6a'9412, 0xef6e'9816},
                                 {0xf372'9c1a, 0xf776'a01e, 0xfb7a'a422, 0xff7e'a826}},
                                {{0x8706'8504'2dad'2baa, 0x8f0e'8d0c'35b5'33b2},
                                 {0x9716'9514'3dbd'3bba, 0x9f1e'9d1c'45c5'43c2},
                                 {0xa726'a524'4dcd'4bca, 0xaf2e'ad2c'55d5'53d2},
                                 {0xb736'b534'5ddd'5bda, 0xbf3e'bd3c'65e5'63e2},
                                 {0xc746'c544'6ded'6bea, 0xcf4e'cd4c'75f5'73f2},
                                 {0xd756'd554'7dfd'7bfa, 0xdf5e'dd5c'8605'8402},
                                 {0xe766'e564'8e0d'8c0a, 0xef6e'ed6c'9615'9412},
                                 {0xf776'f574'9e1d'9c1a, 0xff7e'fd7c'a625'a422}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwadduwx) {
  TestWideningVectorInstruction(0xd100e457,  //  vwaddu.wx v8,v16,ra,v0.t
                                {{0x81aa, 0x83ac, 0x85ae, 0x87b0, 0x89b2, 0x8bb4, 0x8db6, 0x8fb8},
                                 {0x91ba, 0x93bc, 0x95be, 0x97c0, 0x99c2, 0x9bc4, 0x9dc6, 0x9fc8},
                                 {0xa1ca, 0xa3cc, 0xa5ce, 0xa7d0, 0xa9d2, 0xabd4, 0xadd6, 0xafd8},
                                 {0xb1da, 0xb3dc, 0xb5de, 0xb7e0, 0xb9e2, 0xbbe4, 0xbde6, 0xbfe8},
                                 {0xc1ea, 0xc3ec, 0xc5ee, 0xc7f0, 0xc9f2, 0xcbf4, 0xcdf6, 0xcff8},
                                 {0xd1fa, 0xd3fc, 0xd5fe, 0xd800, 0xda02, 0xdc04, 0xde06, 0xe008},
                                 {0xe20a, 0xe40c, 0xe60e, 0xe810, 0xea12, 0xec14, 0xee16, 0xf018},
                                 {0xf21a, 0xf41c, 0xf61e, 0xf820, 0xfa22, 0xfc24, 0xfe26, 0x0028}},
                                {{0x8303'2baa, 0x8707'2fae, 0x8b0b'33b2, 0x8f0f'37b6},
                                 {0x9313'3bba, 0x9717'3fbe, 0x9b1b'43c2, 0x9f1f'47c6},
                                 {0xa323'4bca, 0xa727'4fce, 0xab2b'53d2, 0xaf2f'57d6},
                                 {0xb333'5bda, 0xb737'5fde, 0xbb3b'63e2, 0xbf3f'67e6},
                                 {0xc343'6bea, 0xc747'6fee, 0xcb4b'73f2, 0xcf4f'77f6},
                                 {0xd353'7bfa, 0xd757'7ffe, 0xdb5b'8402, 0xdf5f'8806},
                                 {0xe363'8c0a, 0xe767'900e, 0xeb6b'9412, 0xef6f'9816},
                                 {0xf373'9c1a, 0xf777'a01e, 0xfb7b'a422, 0xff7f'a826}},
                                {{0x8706'8505'2dad'2baa, 0x8f0e'8d0d'35b5'33b2},
                                 {0x9716'9515'3dbd'3bba, 0x9f1e'9d1d'45c5'43c2},
                                 {0xa726'a525'4dcd'4bca, 0xaf2e'ad2d'55d5'53d2},
                                 {0xb736'b535'5ddd'5bda, 0xbf3e'bd3d'65e5'63e2},
                                 {0xc746'c545'6ded'6bea, 0xcf4e'cd4d'75f5'73f2},
                                 {0xd756'd555'7dfd'7bfa, 0xdf5e'dd5d'8605'8402},
                                 {0xe766'e565'8e0d'8c0a, 0xef6e'ed6d'9615'9412},
                                 {0xf776'f575'9e1d'9c1a, 0xff7e'fd7d'a625'a422}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwadduvx) {
  TestWideningVectorInstruction(0xc100e457,  // vwaddu.vx v8, v16, x1, v0.t
                                {{0x00aa, 0x012b, 0x00ac, 0x012d, 0x00ae, 0x012f, 0x00b0, 0x0131},
                                 {0x00b2, 0x0133, 0x00b4, 0x0135, 0x00b6, 0x0137, 0x00b8, 0x0139},
                                 {0x00ba, 0x013b, 0x00bc, 0x013d, 0x00be, 0x013f, 0x00c0, 0x0141},
                                 {0x00c2, 0x0143, 0x00c4, 0x0145, 0x00c6, 0x0147, 0x00c8, 0x0149},
                                 {0x00ca, 0x014b, 0x00cc, 0x014d, 0x00ce, 0x014f, 0x00d0, 0x0151},
                                 {0x00d2, 0x0153, 0x00d4, 0x0155, 0x00d6, 0x0157, 0x00d8, 0x0159},
                                 {0x00da, 0x015b, 0x00dc, 0x015d, 0x00de, 0x015f, 0x00e0, 0x0161},
                                 {0x00e2, 0x0163, 0x00e4, 0x0165, 0x00e6, 0x0167, 0x00e8, 0x0169}},
                                {{0x0001'2baa, 0x0001'2dac, 0x0001'2fae, 0x0001'31b0},
                                 {0x0001'33b2, 0x0001'35b4, 0x0001'37b6, 0x0001'39b8},
                                 {0x0001'3bba, 0x0001'3dbc, 0x0001'3fbe, 0x0001'41c0},
                                 {0x0001'43c2, 0x0001'45c4, 0x0001'47c6, 0x0001'49c8},
                                 {0x0001'4bca, 0x0001'4dcc, 0x0001'4fce, 0x0001'51d0},
                                 {0x0001'53d2, 0x0001'55d4, 0x0001'57d6, 0x0001'59d8},
                                 {0x0001'5bda, 0x0001'5ddc, 0x0001'5fde, 0x0001'61e0},
                                 {0x0001'63e2, 0x0001'65e4, 0x0001'67e6, 0x0001'69e8}},
                                {{0x0000'0001'2dad'2baa, 0x0000'0001'31b1'2fae},
                                 {0x0000'0001'35b5'33b2, 0x0000'0001'39b9'37b6},
                                 {0x0000'0001'3dbd'3bba, 0x0000'0001'41c1'3fbe},
                                 {0x0000'0001'45c5'43c2, 0x0000'0001'49c9'47c6},
                                 {0x0000'0001'4dcd'4bca, 0x0000'0001'51d1'4fce},
                                 {0x0000'0001'55d5'53d2, 0x0000'0001'59d9'57d6},
                                 {0x0000'0001'5ddd'5bda, 0x0000'0001'61e1'5fde},
                                 {0x0000'0001'65e5'63e2, 0x0000'0001'69e9'67e6}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwaddvx) {
  TestWideningVectorInstruction(0xc500e457,  // vwadd.vx v8, v16, x1, v0.t
                                {{0xffaa, 0xff2b, 0xffac, 0xff2d, 0xffae, 0xff2f, 0xffb0, 0xff31},
                                 {0xffb2, 0xff33, 0xffb4, 0xff35, 0xffb6, 0xff37, 0xffb8, 0xff39},
                                 {0xffba, 0xff3b, 0xffbc, 0xff3d, 0xffbe, 0xff3f, 0xffc0, 0xff41},
                                 {0xffc2, 0xff43, 0xffc4, 0xff45, 0xffc6, 0xff47, 0xffc8, 0xff49},
                                 {0xffca, 0xff4b, 0xffcc, 0xff4d, 0xffce, 0xff4f, 0xffd0, 0xff51},
                                 {0xffd2, 0xff53, 0xffd4, 0xff55, 0xffd6, 0xff57, 0xffd8, 0xff59},
                                 {0xffda, 0xff5b, 0xffdc, 0xff5d, 0xffde, 0xff5f, 0xffe0, 0xff61},
                                 {0xffe2, 0xff63, 0xffe4, 0xff65, 0xffe6, 0xff67, 0xffe8, 0xff69}},
                                {{0xffff'2baa, 0xffff'2dac, 0xffff'2fae, 0xffff'31b0},
                                 {0xffff'33b2, 0xffff'35b4, 0xffff'37b6, 0xffff'39b8},
                                 {0xffff'3bba, 0xffff'3dbc, 0xffff'3fbe, 0xffff'41c0},
                                 {0xffff'43c2, 0xffff'45c4, 0xffff'47c6, 0xffff'49c8},
                                 {0xffff'4bca, 0xffff'4dcc, 0xffff'4fce, 0xffff'51d0},
                                 {0xffff'53d2, 0xffff'55d4, 0xffff'57d6, 0xffff'59d8},
                                 {0xffff'5bda, 0xffff'5ddc, 0xffff'5fde, 0xffff'61e0},
                                 {0xffff'63e2, 0xffff'65e4, 0xffff'67e6, 0xffff'69e8}},
                                {{0xffff'ffff'2dad'2baa, 0xffff'ffff'31b1'2fae},
                                 {0xffff'ffff'35b5'33b2, 0xffff'ffff'39b9'37b6},
                                 {0xffff'ffff'3dbd'3bba, 0xffff'ffff'41c1'3fbe},
                                 {0xffff'ffff'45c5'43c2, 0xffff'ffff'49c9'47c6},
                                 {0xffff'ffff'4dcd'4bca, 0xffff'ffff'51d1'4fce},
                                 {0xffff'ffff'55d5'53d2, 0xffff'ffff'59d9'57d6},
                                 {0xffff'ffff'5ddd'5bda, 0xffff'ffff'61e1'5fde},
                                 {0xffff'ffff'65e5'63e2, 0xffff'ffff'69e9'67e6}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwsubuvv) {
  TestWideningVectorInstruction(0xc90c2457,  // vwsubu.vv v8, v16, v24, v0.t
                                {{0x0000, 0xffef, 0xfffe, 0xffed, 0xfffb, 0xffeb, 0xfffa, 0xffe9},
                                 {0xfff7, 0x0007, 0xfff6, 0x0005, 0xfff4, 0x0003, 0xfff2, 0x0001},
                                 {0xfff0, 0xffdf, 0xffee, 0xffdd, 0xffeb, 0xffdb, 0xffea, 0xffd9},
                                 {0xffe7, 0xfff7, 0xffe6, 0xfff5, 0xffe4, 0xfff3, 0xffe2, 0xfff1},
                                 {0xffe0, 0xffcf, 0xffde, 0xffcd, 0xffdb, 0xffcb, 0xffda, 0xffc9},
                                 {0xffd7, 0xffe7, 0xffd6, 0xffe5, 0xffd4, 0xffe3, 0xffd2, 0xffe1},
                                 {0xffd0, 0xffbf, 0xffce, 0xffbd, 0xffcb, 0xffbb, 0xffca, 0xffb9},
                                 {0xffc7, 0xffd7, 0xffc6, 0xffd5, 0xffc4, 0xffd3, 0xffc2, 0xffd1}},
                                {{0xffff'ef00, 0xffff'ecfe, 0xffff'eafb, 0xffff'e8fa},
                                 {0x0000'06f7, 0x0000'04f6, 0x0000'02f4, 0x0000'00f2},
                                 {0xffff'def0, 0xffff'dcee, 0xffff'daeb, 0xffff'd8ea},
                                 {0xffff'f6e7, 0xffff'f4e6, 0xffff'f2e4, 0xffff'f0e2},
                                 {0xffff'cee0, 0xffff'ccde, 0xffff'cadb, 0xffff'c8da},
                                 {0xffff'e6d7, 0xffff'e4d6, 0xffff'e2d4, 0xffff'e0d2},
                                 {0xffff'bed0, 0xffff'bcce, 0xffff'bacb, 0xffff'b8ca},
                                 {0xffff'd6c7, 0xffff'd4c6, 0xffff'd2c4, 0xffff'd0c2}},
                                {{0xffff'ffff'ecfd'ef00, 0xffff'ffff'e8f9'eafb},
                                 {0x0000'0000'04f6'06f7, 0x0000'0000'00f2'02f4},
                                 {0xffff'ffff'dced'def0, 0xffff'ffff'd8e9'daeb},
                                 {0xffff'ffff'f4e5'f6e7, 0xffff'ffff'f0e1'f2e4},
                                 {0xffff'ffff'ccdd'cee0, 0xffff'ffff'c8d9'cadb},
                                 {0xffff'ffff'e4d5'e6d7, 0xffff'ffff'e0d1'e2d4},
                                 {0xffff'ffff'bccd'bed0, 0xffff'ffff'b8c9'bacb},
                                 {0xffff'ffff'd4c5'd6c7, 0xffff'ffff'd0c1'd2c4}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwsubvv) {
  TestWideningVectorInstruction(0xcd0c2457,  // vwsub.vv v8, v16, v24, v0.t
                                {{0x0000, 0xffef, 0xfffe, 0xffed, 0xfffb, 0xffeb, 0xfffa, 0xffe9},
                                 {0xfff7, 0x0007, 0xfff6, 0x0005, 0xfff4, 0x0003, 0xfff2, 0x0001},
                                 {0xfff0, 0xffdf, 0xffee, 0xffdd, 0xffeb, 0xffdb, 0xffea, 0xffd9},
                                 {0xffe7, 0xfff7, 0xffe6, 0xfff5, 0xffe4, 0xfff3, 0xffe2, 0xfff1},
                                 {0xffe0, 0xffcf, 0xffde, 0xffcd, 0xffdb, 0xffcb, 0xffda, 0xffc9},
                                 {0xffd7, 0xffe7, 0xffd6, 0xffe5, 0xffd4, 0xffe3, 0xffd2, 0xffe1},
                                 {0xffd0, 0xffbf, 0xffce, 0xffbd, 0xffcb, 0xffbb, 0xffca, 0xffb9},
                                 {0xffc7, 0xffd7, 0xffc6, 0xffd5, 0xffc4, 0xffd3, 0xffc2, 0xffd1}},
                                {{0xffff'ef00, 0xffff'ecfe, 0xffff'eafb, 0xffff'e8fa},
                                 {0x0000'06f7, 0x0000'04f6, 0x0000'02f4, 0x0000'00f2},
                                 {0xffff'def0, 0xffff'dcee, 0xffff'daeb, 0xffff'd8ea},
                                 {0xffff'f6e7, 0xffff'f4e6, 0xffff'f2e4, 0xffff'f0e2},
                                 {0xffff'cee0, 0xffff'ccde, 0xffff'cadb, 0xffff'c8da},
                                 {0xffff'e6d7, 0xffff'e4d6, 0xffff'e2d4, 0xffff'e0d2},
                                 {0xffff'bed0, 0xffff'bcce, 0xffff'bacb, 0xffff'b8ca},
                                 {0xffff'd6c7, 0xffff'd4c6, 0xffff'd2c4, 0xffff'd0c2}},
                                {{0xffff'ffff'ecfd'ef00, 0xffff'ffff'e8f9'eafb},
                                 {0x0000'0000'04f6'06f7, 0x0000'0000'00f2'02f4},
                                 {0xffff'ffff'dced'def0, 0xffff'ffff'd8e9'daeb},
                                 {0xffff'ffff'f4e5'f6e7, 0xffff'ffff'f0e1'f2e4},
                                 {0xffff'ffff'ccdd'cee0, 0xffff'ffff'c8d9'cadb},
                                 {0xffff'ffff'e4d5'e6d7, 0xffff'ffff'e0d1'e2d4},
                                 {0xffff'ffff'bccd'bed0, 0xffff'ffff'b8c9'bacb},
                                 {0xffff'ffff'd4c5'd6c7, 0xffff'ffff'd0c1'd2c4}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwsubuvx) {
  TestWideningVectorInstruction(0xc900e457,  // vwsubu.vx v8, v16, x1, v0.t
                                {{0xff56, 0xffd7, 0xff58, 0xffd9, 0xff5a, 0xffdb, 0xff5c, 0xffdd},
                                 {0xff5e, 0xffdf, 0xff60, 0xffe1, 0xff62, 0xffe3, 0xff64, 0xffe5},
                                 {0xff66, 0xffe7, 0xff68, 0xffe9, 0xff6a, 0xffeb, 0xff6c, 0xffed},
                                 {0xff6e, 0xffef, 0xff70, 0xfff1, 0xff72, 0xfff3, 0xff74, 0xfff5},
                                 {0xff76, 0xfff7, 0xff78, 0xfff9, 0xff7a, 0xfffb, 0xff7c, 0xfffd},
                                 {0xff7e, 0xffff, 0xff80, 0x0001, 0xff82, 0x0003, 0xff84, 0x0005},
                                 {0xff86, 0x0007, 0xff88, 0x0009, 0xff8a, 0x000b, 0xff8c, 0x000d},
                                 {0xff8e, 0x000f, 0xff90, 0x0011, 0xff92, 0x0013, 0xff94, 0x0015}},
                                {{0xffff'd656, 0xffff'd858, 0xffff'da5a, 0xffff'dc5c},
                                 {0xffff'de5e, 0xffff'e060, 0xffff'e262, 0xffff'e464},
                                 {0xffff'e666, 0xffff'e868, 0xffff'ea6a, 0xffff'ec6c},
                                 {0xffff'ee6e, 0xffff'f070, 0xffff'f272, 0xffff'f474},
                                 {0xffff'f676, 0xffff'f878, 0xffff'fa7a, 0xffff'fc7c},
                                 {0xffff'fe7e, 0x0000'0080, 0x0000'0282, 0x0000'0484},
                                 {0x0000'0686, 0x0000'0888, 0x0000'0a8a, 0x0000'0c8c},
                                 {0x0000'0e8e, 0x0000'1090, 0x0000'1292, 0x0000'1494}},
                                {{0xffff'ffff'd857'd656, 0xffff'ffff'dc5b'da5a},
                                 {0xffff'ffff'e05f'de5e, 0xffff'ffff'e463'e262},
                                 {0xffff'ffff'e867'e666, 0xffff'ffff'ec6b'ea6a},
                                 {0xffff'ffff'f06f'ee6e, 0xffff'ffff'f473'f272},
                                 {0xffff'ffff'f877'f676, 0xffff'ffff'fc7b'fa7a},
                                 {0x0000'0000'007f'fe7e, 0x0000'0000'0484'0282},
                                 {0x0000'0000'0888'0686, 0x0000'0000'0c8c'0a8a},
                                 {0x0000'0000'1090'0e8e, 0x0000'0000'1494'1292}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwsubvx) {
  TestWideningVectorInstruction(0xcd00e457,  // vwsub.vx v8, v16, x1, v0.t
                                {{0x0056, 0xffd7, 0x0058, 0xffd9, 0x005a, 0xffdb, 0x005c, 0xffdd},
                                 {0x005e, 0xffdf, 0x0060, 0xffe1, 0x0062, 0xffe3, 0x0064, 0xffe5},
                                 {0x0066, 0xffe7, 0x0068, 0xffe9, 0x006a, 0xffeb, 0x006c, 0xffed},
                                 {0x006e, 0xffef, 0x0070, 0xfff1, 0x0072, 0xfff3, 0x0074, 0xfff5},
                                 {0x0076, 0xfff7, 0x0078, 0xfff9, 0x007a, 0xfffb, 0x007c, 0xfffd},
                                 {0x007e, 0xffff, 0x0080, 0x0001, 0x0082, 0x0003, 0x0084, 0x0005},
                                 {0x0086, 0x0007, 0x0088, 0x0009, 0x008a, 0x000b, 0x008c, 0x000d},
                                 {0x008e, 0x000f, 0x0090, 0x0011, 0x0092, 0x0013, 0x0094, 0x0015}},
                                {{0xffff'd656, 0xffff'd858, 0xffff'da5a, 0xffff'dc5c},
                                 {0xffff'de5e, 0xffff'e060, 0xffff'e262, 0xffff'e464},
                                 {0xffff'e666, 0xffff'e868, 0xffff'ea6a, 0xffff'ec6c},
                                 {0xffff'ee6e, 0xffff'f070, 0xffff'f272, 0xffff'f474},
                                 {0xffff'f676, 0xffff'f878, 0xffff'fa7a, 0xffff'fc7c},
                                 {0xffff'fe7e, 0x0000'0080, 0x0000'0282, 0x0000'0484},
                                 {0x0000'0686, 0x0000'0888, 0x0000'0a8a, 0x0000'0c8c},
                                 {0x0000'0e8e, 0x0000'1090, 0x0000'1292, 0x0000'1494}},
                                {{0xffff'ffff'd857'd656, 0xffff'ffff'dc5b'da5a},
                                 {0xffff'ffff'e05f'de5e, 0xffff'ffff'e463'e262},
                                 {0xffff'ffff'e867'e666, 0xffff'ffff'ec6b'ea6a},
                                 {0xffff'ffff'f06f'ee6e, 0xffff'ffff'f473'f272},
                                 {0xffff'ffff'f877'f676, 0xffff'ffff'fc7b'fa7a},
                                 {0x0000'0000'007f'fe7e, 0x0000'0000'0484'0282},
                                 {0x0000'0000'0888'0686, 0x0000'0000'0c8c'0a8a},
                                 {0x0000'0000'1090'0e8e, 0x0000'0000'1494'1292}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwsubwx) {
  TestWideningVectorInstruction(0xdd00e457,  //  vwsub.wx v8,v16,ra,v0.t
                                {{0x8156, 0x8358, 0x855a, 0x875c, 0x895e, 0x8b60, 0x8d62, 0x8f64},
                                 {0x9166, 0x9368, 0x956a, 0x976c, 0x996e, 0x9b70, 0x9d72, 0x9f74},
                                 {0xa176, 0xa378, 0xa57a, 0xa77c, 0xa97e, 0xab80, 0xad82, 0xaf84},
                                 {0xb186, 0xb388, 0xb58a, 0xb78c, 0xb98e, 0xbb90, 0xbd92, 0xbf94},
                                 {0xc196, 0xc398, 0xc59a, 0xc79c, 0xc99e, 0xcba0, 0xcda2, 0xcfa4},
                                 {0xd1a6, 0xd3a8, 0xd5aa, 0xd7ac, 0xd9ae, 0xdbb0, 0xddb2, 0xdfb4},
                                 {0xe1b6, 0xe3b8, 0xe5ba, 0xe7bc, 0xe9be, 0xebc0, 0xedc2, 0xefc4},
                                 {0xf1c6, 0xf3c8, 0xf5ca, 0xf7cc, 0xf9ce, 0xfbd0, 0xfdd2, 0xffd4}},
                                {{0x8302'd656, 0x8706'da5a, 0x8b0a'de5e, 0x8f0e'e262},
                                 {0x9312'e666, 0x9716'ea6a, 0x9b1a'ee6e, 0x9f1e'f272},
                                 {0xa322'f676, 0xa726'fa7a, 0xab2a'fe7e, 0xaf2f'0282},
                                 {0xb333'0686, 0xb737'0a8a, 0xbb3b'0e8e, 0xbf3f'1292},
                                 {0xc343'1696, 0xc747'1a9a, 0xcb4b'1e9e, 0xcf4f'22a2},
                                 {0xd353'26a6, 0xd757'2aaa, 0xdb5b'2eae, 0xdf5f'32b2},
                                 {0xe363'36b6, 0xe767'3aba, 0xeb6b'3ebe, 0xef6f'42c2},
                                 {0xf373'46c6, 0xf777'4aca, 0xfb7b'4ece, 0xff7f'52d2}},
                                {{0x8706'8504'd857'd656, 0x8f0e'8d0c'e05f'de5e},
                                 {0x9716'9514'e867'e666, 0x9f1e'9d1c'f06f'ee6e},
                                 {0xa726'a524'f877'f676, 0xaf2e'ad2d'007f'fe7e},
                                 {0xb736'b535'0888'0686, 0xbf3e'bd3d'1090'0e8e},
                                 {0xc746'c545'1898'1696, 0xcf4e'cd4d'20a0'1e9e},
                                 {0xd756'd555'28a8'26a6, 0xdf5e'dd5d'30b0'2eae},
                                 {0xe766'e565'38b8'36b6, 0xef6e'ed6d'40c0'3ebe},
                                 {0xf776'f575'48c8'46c6, 0xff7e'fd7d'50d0'4ece}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwsubuwx) {
  TestWideningVectorInstruction(0xd900e457,  //  vwsubu.wx v8,v16,ra,v0.t
                                {{0x8056, 0x8258, 0x845a, 0x865c, 0x885e, 0x8a60, 0x8c62, 0x8e64},
                                 {0x9066, 0x9268, 0x946a, 0x966c, 0x986e, 0x9a70, 0x9c72, 0x9e74},
                                 {0xa076, 0xa278, 0xa47a, 0xa67c, 0xa87e, 0xaa80, 0xac82, 0xae84},
                                 {0xb086, 0xb288, 0xb48a, 0xb68c, 0xb88e, 0xba90, 0xbc92, 0xbe94},
                                 {0xc096, 0xc298, 0xc49a, 0xc69c, 0xc89e, 0xcaa0, 0xcca2, 0xcea4},
                                 {0xd0a6, 0xd2a8, 0xd4aa, 0xd6ac, 0xd8ae, 0xdab0, 0xdcb2, 0xdeb4},
                                 {0xe0b6, 0xe2b8, 0xe4ba, 0xe6bc, 0xe8be, 0xeac0, 0xecc2, 0xeec4},
                                 {0xf0c6, 0xf2c8, 0xf4ca, 0xf6cc, 0xf8ce, 0xfad0, 0xfcd2, 0xfed4}},
                                {{0x8301'd656, 0x8705'da5a, 0x8b09'de5e, 0x8f0d'e262},
                                 {0x9311'e666, 0x9715'ea6a, 0x9b19'ee6e, 0x9f1d'f272},
                                 {0xa321'f676, 0xa725'fa7a, 0xab29'fe7e, 0xaf2e'0282},
                                 {0xb332'0686, 0xb736'0a8a, 0xbb3a'0e8e, 0xbf3e'1292},
                                 {0xc342'1696, 0xc746'1a9a, 0xcb4a'1e9e, 0xcf4e'22a2},
                                 {0xd352'26a6, 0xd756'2aaa, 0xdb5a'2eae, 0xdf5e'32b2},
                                 {0xe362'36b6, 0xe766'3aba, 0xeb6a'3ebe, 0xef6e'42c2},
                                 {0xf372'46c6, 0xf776'4aca, 0xfb7a'4ece, 0xff7e'52d2}},
                                {{0x8706'8503'd857'd656, 0x8f0e'8d0b'e05f'de5e},
                                 {0x9716'9513'e867'e666, 0x9f1e'9d1b'f06f'ee6e},
                                 {0xa726'a523'f877'f676, 0xaf2e'ad2c'007f'fe7e},
                                 {0xb736'b534'0888'0686, 0xbf3e'bd3c'1090'0e8e},
                                 {0xc746'c544'1898'1696, 0xcf4e'cd4c'20a0'1e9e},
                                 {0xd756'd554'28a8'26a6, 0xdf5e'dd5c'30b0'2eae},
                                 {0xe766'e564'38b8'36b6, 0xef6e'ed6c'40c0'3ebe},
                                 {0xf776'f574'48c8'46c6, 0xff7e'fd7c'50d0'4ece}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwmul) {
  TestWideningVectorInstruction(0xed0c2457,
                                {{0x0000, 0x3692, 0x0008, 0x33c2, 0x0024, 0x3102, 0x0048, 0x2e52},
                                 {0x0088, 0x3a92, 0x00c8, 0x37c2, 0x0120, 0x3502, 0x0188, 0x3252},
                                 {0x0200, 0x21d2, 0x0288, 0x1f82, 0x0334, 0x1d42, 0x03c8, 0x1b12},
                                 {0x0498, 0x25d2, 0x0548, 0x2382, 0x0620, 0x2142, 0x0708, 0x1f12},
                                 {0x0800, 0x1112, 0x0908, 0x0f42, 0x0a44, 0x0d82, 0x0b48, 0x0bd2},
                                 {0x0ca8, 0x1512, 0x0dc8, 0x1342, 0x0f20, 0x1182, 0x1088, 0x0fd2},
                                 {0x1200, 0x0452, 0x1388, 0x0302, 0x1554, 0x01c2, 0x16c8, 0x0092},
                                 {0x18b8, 0x0852, 0x1a48, 0x0702, 0x1c20, 0x05c2, 0x1e08, 0x0492}},
                                {{0x3692'0000, 0x33bf'3808, 0x30fc'1524, 0x2e4a'0848},
                                 {0x3a86'2988, 0x37b4'18c8, 0x34f1'b120, 0x323f'6988},
                                 {0x21bf'4200, 0x1f6d'7a88, 0x1d2b'6834, 0x1afa'4bc8},
                                 {0x25b5'7d98, 0x2364'5d48, 0x2122'f620, 0x1ef1'af08},
                                 {0x10f4'8800, 0x0f23'c108, 0x0d62'bf44, 0x0bb2'9348},
                                 {0x14ec'd5a8, 0x131c'a5c8, 0x115c'3f20, 0x0fab'f888},
                                 {0x0431'd200, 0x02e2'0b88, 0x01a2'1a54, 0x0072'dec8},
                                 {0x082c'31b8, 0x06dc'f248, 0x059d'8c20, 0x046e'4608}},
                                {{0x33be'bb57'7192'0000, 0x2e49'8c98'5f6f'1524},
                                 {0x37b3'9c18'79e9'2988, 0x323e'eddb'56b5'b120},
                                 {0x1f6d'04e3'116f'4200, 0x1af9'd928'125e'6834},
                                 {0x2363'e7a8'2dd8'7d98, 0x1ef1'3c6e'fd96'f620},
                                 {0x0f23'5a7e'bd54'8800, 0x0bb2'31c7'd155'bf44},
                                 {0x131c'3f47'edcf'd5a8, 0x0fab'9712'b080'3f20},
                                 {0x02e1'bc2a'7541'd200, 0x0072'9677'9c55'1a54},
                                 {0x06dc'a2f7'b9cf'31b8, 0x046d'fdc6'6f71'8c20}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwmulu) {
  TestWideningVectorInstruction(0xe10c2457,
                                {{0x0000, 0x4992, 0x0008, 0x4cc2, 0x0024, 0x5002, 0x0048, 0x5352},
                                 {0x0088, 0x4592, 0x00c8, 0x48c2, 0x0120, 0x4c02, 0x0188, 0x4f52},
                                 {0x0200, 0x64d2, 0x0288, 0x6882, 0x0334, 0x6c42, 0x03c8, 0x7012},
                                 {0x0498, 0x60d2, 0x0548, 0x6482, 0x0620, 0x6842, 0x0708, 0x6c12},
                                 {0x0800, 0x8412, 0x0908, 0x8842, 0x0a44, 0x8c82, 0x0b48, 0x90d2},
                                 {0x0ca8, 0x8012, 0x0dc8, 0x8442, 0x0f20, 0x8882, 0x1088, 0x8cd2},
                                 {0x1200, 0xa752, 0x1388, 0xac02, 0x1554, 0xb0c2, 0x16c8, 0xb592},
                                 {0x18b8, 0xa352, 0x1a48, 0xa802, 0x1c20, 0xacc2, 0x1e08, 0xb192}},
                                {{0x4992'0000, 0x4cc5'3808, 0x5009'1524, 0x535c'0848},
                                 {0x459f'2988, 0x48d2'18c8, 0x4c15'b120, 0x4f69'6988},
                                 {0x64ef'4200, 0x68a3'7a88, 0x6c68'6834, 0x703c'4bc8},
                                 {0x60fe'7d98, 0x64b2'5d48, 0x6876'f620, 0x6c4b'af08},
                                 {0x8454'8800, 0x8889'c108, 0x8ccf'bf44, 0x9124'9348},
                                 {0x8065'd5a8, 0x849a'a5c8, 0x88e0'3f20, 0x8d35'f888},
                                 {0xa7c1'd200, 0xac78'0b88, 0xb13f'1a54, 0xb614'dec8},
                                 {0xa3d5'31b8, 0xa88a'f248, 0xad51'8c20, 0xb228'4608}},
                                {{0x4cc5'ce57'7192'0000, 0x535c'aba5'5f6f'1524},
                                 {0x48d2'a731'79e9'2988, 0x4f6a'04ff'56b5'b120},
                                 {0x68a4'4813'116f'4200, 0x703d'2865'125e'6834},
                                 {0x64b3'22f1'2dd8'7d98, 0x6c4c'83c2'fd96'f620},
                                 {0x888a'cdde'bd54'8800, 0x9125'b134'd155'bf44},
                                 {0x849b'aac0'edcf'd5a8, 0x8d37'0e96'b080'3f20},
                                 {0xac79'5fba'7541'd200, 0xb616'4614'9c55'1a54},
                                 {0xa88c'3ea0'b9cf'31b8, 0xb229'a57a'6f71'8c20}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwmulsu) {
  TestWideningVectorInstruction(0xe90c2457,
                                {{0x0000, 0xb792, 0x0008, 0xb6c2, 0x0024, 0xb602, 0x0048, 0xb552},
                                 {0x0088, 0xc392, 0x00c8, 0xc2c2, 0x0120, 0xc202, 0x0188, 0xc152},
                                 {0x0200, 0xb2d2, 0x0288, 0xb282, 0x0334, 0xb242, 0x03c8, 0xb212},
                                 {0x0498, 0xbed2, 0x0548, 0xbe82, 0x0620, 0xbe42, 0x0708, 0xbe12},
                                 {0x0800, 0xb212, 0x0908, 0xb242, 0x0a44, 0xb282, 0x0b48, 0xb2d2},
                                 {0x0ca8, 0xbe12, 0x0dc8, 0xbe42, 0x0f20, 0xbe82, 0x1088, 0xbed2},
                                 {0x1200, 0xb552, 0x1388, 0xb602, 0x1554, 0xb6c2, 0x16c8, 0xb792},
                                 {0x18b8, 0xc152, 0x1a48, 0xc202, 0x1c20, 0xc2c2, 0x1e08, 0xc392}},
                                {{0xb792'0000, 0xb6c1'3808, 0xb600'1524, 0xb550'0848},
                                 {0xc38e'2988, 0xc2be'18c8, 0xc1fd'b120, 0xc14d'6988},
                                 {0xb2cf'4200, 0xb27f'7a88, 0xb23f'6834, 0xb210'4bc8},
                                 {0xbecd'7d98, 0xbe7e'5d48, 0xbe3e'f620, 0xbe0f'af08},
                                 {0xb214'8800, 0xb245'c108, 0xb286'bf44, 0xb2d8'9348},
                                 {0xbe14'd5a8, 0xbe46'a5c8, 0xbe88'3f20, 0xbed9'f888},
                                 {0xb561'd200, 0xb614'0b88, 0xb6d6'1a54, 0xb7a8'dec8},
                                 {0xc164'31b8, 0xc216'f248, 0xc2d9'8c20, 0xc3ac'4608}},
                                {{0xb6c1'3c57'7192'0000, 0xb550'119c'5f6f'1524},
                                 {0xc2be'2520'79e9'2988, 0xc14d'7ae7'56b5'b120},
                                 {0xb27f'95f3'116f'4200, 0xb210'6e3c'125e'6834},
                                 {0xbe7e'80c0'2dd8'7d98, 0xbe0f'd98a'fd96'f620},
                                 {0xb245'fb9e'bd54'8800, 0xb2d8'd6eb'd155'bf44},
                                 {0xbe46'e86f'edcf'd5a8, 0xbeda'443e'b080'3f20},
                                 {0xb614'6d5a'7541'd200, 0xb7a9'4bab'9c55'1a54},
                                 {0xc217'5c2f'b9cf'31b8, 0xc3ac'bb02'6f71'8c20}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwmulvx) {
  TestWideningVectorInstruction(0xed00e457,  // Vwmul.vx v8, v16, x1, v0.t
                                {{0x0000, 0x2aaa, 0xff54, 0x29fe, 0xfea8, 0x2952, 0xfdfc, 0x28a6},
                                 {0xfd50, 0x27fa, 0xfca4, 0x274e, 0xfbf8, 0x26a2, 0xfb4c, 0x25f6},
                                 {0xfaa0, 0x254a, 0xf9f4, 0x249e, 0xf948, 0x23f2, 0xf89c, 0x2346},
                                 {0xf7f0, 0x229a, 0xf744, 0x21ee, 0xf698, 0x2142, 0xf5ec, 0x2096},
                                 {0xf540, 0x1fea, 0xf494, 0x1f3e, 0xf3e8, 0x1e92, 0xf33c, 0x1de6},
                                 {0xf290, 0x1d3a, 0xf1e4, 0x1c8e, 0xf138, 0x1be2, 0xf08c, 0x1b36},
                                 {0xefe0, 0x1a8a, 0xef34, 0x19de, 0xee88, 0x1932, 0xeddc, 0x1886},
                                 {0xed30, 0x17da, 0xec84, 0x172e, 0xebd8, 0x1682, 0xeb2c, 0x15d6}},
                                {{0x2a55'aa00, 0x29aa'5354, 0x28fe'fca8, 0x2853'a5fc},
                                 {0x27a8'4f50, 0x26fc'f8a4, 0x2651'a1f8, 0x25a6'4b4c},
                                 {0x24fa'f4a0, 0x244f'9df4, 0x23a4'4748, 0x22f8'f09c},
                                 {0x224d'99f0, 0x21a2'4344, 0x20f6'ec98, 0x204b'95ec},
                                 {0x1fa0'3f40, 0x1ef4'e894, 0x1e49'91e8, 0x1d9e'3b3c},
                                 {0x1cf2'e490, 0x1c47'8de4, 0x1b9c'3738, 0x1af0'e08c},
                                 {0x1a45'89e0, 0x199a'3334, 0x18ee'dc88, 0x1843'85dc},
                                 {0x1798'2f30, 0x16ec'd884, 0x1641'81d8, 0x1596'2b2c}},
                                {{0x29a9'd500'5353'aa00, 0x2853'28fe'fb50'fca8},
                                 {0x26fc'7cfd'a34e'4f50, 0x25a5'd0fc'4b4b'a1f8},
                                 {0x244f'24fa'f348'f4a0, 0x22f8'78f9'9b46'4748},
                                 {0x21a1'ccf8'4343'99f0, 0x204b'20f6'eb40'ec98},
                                 {0x1ef4'74f5'933e'3f40, 0x1d9d'c8f4'3b3b'91e8},
                                 {0x1c47'1cf2'e338'e490, 0x1af0'70f1'8b36'3738},
                                 {0x1999'c4f0'3333'89e0, 0x1843'18ee'db30'dc88},
                                 {0x16ec'6ced'832e'2f30, 0x1595'c0ec'2b2b'81d8}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwmuluvx) {
  TestWideningVectorInstruction(0xe100e457,  // Vwmulu.vx v8, v16, x1, v0.t
                                {{0x0000, 0x55aa, 0x0154, 0x56fe, 0x02a8, 0x5852, 0x03fc, 0x59a6},
                                 {0x0550, 0x5afa, 0x06a4, 0x5c4e, 0x07f8, 0x5da2, 0x094c, 0x5ef6},
                                 {0x0aa0, 0x604a, 0x0bf4, 0x619e, 0x0d48, 0x62f2, 0x0e9c, 0x6446},
                                 {0x0ff0, 0x659a, 0x1144, 0x66ee, 0x1298, 0x6842, 0x13ec, 0x6996},
                                 {0x1540, 0x6aea, 0x1694, 0x6c3e, 0x17e8, 0x6d92, 0x193c, 0x6ee6},
                                 {0x1a90, 0x703a, 0x1be4, 0x718e, 0x1d38, 0x72e2, 0x1e8c, 0x7436},
                                 {0x1fe0, 0x758a, 0x2134, 0x76de, 0x2288, 0x7832, 0x23dc, 0x7986},
                                 {0x2530, 0x7ada, 0x2684, 0x7c2e, 0x27d8, 0x7d82, 0x292c, 0x7ed6}},
                                {{0x55ff'aa00, 0x5756'5354, 0x58ac'fca8, 0x5a03'a5fc},
                                 {0x5b5a'4f50, 0x5cb0'f8a4, 0x5e07'a1f8, 0x5f5e'4b4c},
                                 {0x60b4'f4a0, 0x620b'9df4, 0x6362'4748, 0x64b8'f09c},
                                 {0x660f'99f0, 0x6766'4344, 0x68bc'ec98, 0x6a13'95ec},
                                 {0x6b6a'3f40, 0x6cc0'e894, 0x6e17'91e8, 0x6f6e'3b3c},
                                 {0x70c4'e490, 0x721b'8de4, 0x7372'3738, 0x74c8'e08c},
                                 {0x761f'89e0, 0x7776'3334, 0x78cc'dc88, 0x7a23'85dc},
                                 {0x7b7a'2f30, 0x7cd0'd884, 0x7e27'81d8, 0x7f7e'2b2c}},
                                {{0x5757'00aa'5353'aa00, 0x5a04'58ac'fb50'fca8},
                                 {0x5cb1'b0af'a34e'4f50, 0x5f5f'08b2'4b4b'a1f8},
                                 {0x620c'60b4'f348'f4a0, 0x64b9'b8b7'9b46'4748},
                                 {0x6767'10ba'4343'99f0, 0x6a14'68bc'eb40'ec98},
                                 {0x6cc1'c0bf'933e'3f40, 0x6f6f'18c2'3b3b'91e8},
                                 {0x721c'70c4'e338'e490, 0x74c9'c8c7'8b36'3738},
                                 {0x7777'20ca'3333'89e0, 0x7a24'78cc'db30'dc88},
                                 {0x7cd1'd0cf'832e'2f30, 0x7f7f'28d2'2b2b'81d8}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwmulsuvx) {
  TestWideningVectorInstruction(0xe900e457,  // Vwmulsu.vx v8, v16, x1, v0.t
                                {{0x0000, 0xabaa, 0x0154, 0xacfe, 0x02a8, 0xae52, 0x03fc, 0xafa6},
                                 {0x0550, 0xb0fa, 0x06a4, 0xb24e, 0x07f8, 0xb3a2, 0x094c, 0xb4f6},
                                 {0x0aa0, 0xb64a, 0x0bf4, 0xb79e, 0x0d48, 0xb8f2, 0x0e9c, 0xba46},
                                 {0x0ff0, 0xbb9a, 0x1144, 0xbcee, 0x1298, 0xbe42, 0x13ec, 0xbf96},
                                 {0x1540, 0xc0ea, 0x1694, 0xc23e, 0x17e8, 0xc392, 0x193c, 0xc4e6},
                                 {0x1a90, 0xc63a, 0x1be4, 0xc78e, 0x1d38, 0xc8e2, 0x1e8c, 0xca36},
                                 {0x1fe0, 0xcb8a, 0x2134, 0xccde, 0x2288, 0xce32, 0x23dc, 0xcf86},
                                 {0x2530, 0xd0da, 0x2684, 0xd22e, 0x27d8, 0xd382, 0x292c, 0xd4d6}},
                                {{0xab55'aa00, 0xacac'5354, 0xae02'fca8, 0xaf59'a5fc},
                                 {0xb0b0'4f50, 0xb206'f8a4, 0xb35d'a1f8, 0xb4b4'4b4c},
                                 {0xb60a'f4a0, 0xb761'9df4, 0xb8b8'4748, 0xba0e'f09c},
                                 {0xbb65'99f0, 0xbcbc'4344, 0xbe12'ec98, 0xbf69'95ec},
                                 {0xc0c0'3f40, 0xc216'e894, 0xc36d'91e8, 0xc4c4'3b3c},
                                 {0xc61a'e490, 0xc771'8de4, 0xc8c8'3738, 0xca1e'e08c},
                                 {0xcb75'89e0, 0xcccc'3334, 0xce22'dc88, 0xcf79'85dc},
                                 {0xd0d0'2f30, 0xd226'd884, 0xd37d'81d8, 0xd4d4'2b2c}},
                                {{0xacac'5600'5353'aa00, 0xaf59'ae02'fb50'fca8},
                                 {0xb207'0605'a34e'4f50, 0xb4b4'5e08'4b4b'a1f8},
                                 {0xb761'b60a'f348'f4a0, 0xba0f'0e0d'9b46'4748},
                                 {0xbcbc'6610'4343'99f0, 0xbf69'be12'eb40'ec98},
                                 {0xc217'1615'933e'3f40, 0xc4c4'6e18'3b3b'91e8},
                                 {0xc771'c61a'e338'e490, 0xca1f'1e1d'8b36'3738},
                                 {0xcccc'7620'3333'89e0, 0xcf79'ce22'db30'dc88},
                                 {0xd227'2625'832e'2f30, 0xd4d4'7e28'2b2b'81d8}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwaddwv) {
  TestWideningVectorInstruction(0xd50c2457,  // vwadd.wv v8, v16, v24, v0.t
                                {{0x8100, 0x8294, 0x8508, 0x869c, 0x8911, 0x8aa4, 0x8d18, 0x8eac},
                                 {0x9121, 0x9294, 0x9528, 0x969c, 0x9930, 0x9aa4, 0x9d38, 0x9eac},
                                 {0xa140, 0xa2d4, 0xa548, 0xa6dc, 0xa951, 0xaae4, 0xad58, 0xaeec},
                                 {0xb161, 0xb2d4, 0xb568, 0xb6dc, 0xb970, 0xbae4, 0xbd78, 0xbeec},
                                 {0xc180, 0xc314, 0xc588, 0xc71c, 0xc991, 0xcb24, 0xcd98, 0xcf2c},
                                 {0xd1a1, 0xd314, 0xd5a8, 0xd71c, 0xd9b0, 0xdb24, 0xddb8, 0xdf2c},
                                 {0xe1c0, 0xe354, 0xe5c8, 0xe75c, 0xe9d1, 0xeb64, 0xedd8, 0xef6c},
                                 {0xf1e1, 0xf354, 0xf5e8, 0xf75c, 0xf9f0, 0xfb64, 0xfdf8, 0xff6c}},
                                {{0x8302'1300, 0x8706'1b08, 0x8b0a'2311, 0x8f0e'2b18},
                                 {0x9312'1321, 0x9716'1b28, 0x9b1a'2330, 0x9f1e'2b38},
                                 {0xa322'5340, 0xa726'5b48, 0xab2a'6351, 0xaf2e'6b58},
                                 {0xb332'5361, 0xb736'5b68, 0xbb3a'6370, 0xbf3e'6b78},
                                 {0xc342'9380, 0xc746'9b88, 0xcb4a'a391, 0xcf4e'ab98},
                                 {0xd352'93a1, 0xd756'9ba8, 0xdb5a'a3b0, 0xdf5e'abb8},
                                 {0xe362'd3c0, 0xe766'dbc8, 0xeb6a'e3d1, 0xef6e'ebd8},
                                 {0xf372'd3e1, 0xf776'dbe8, 0xfb7a'e3f0, 0xff7e'ebf8}},
                                {{0x8706'8504'1907'1300, 0x8f0e'8d0c'2917'2311},
                                 {0x9716'9514'1927'1321, 0x9f1e'9d1c'2937'2330},
                                 {0xa726'a524'5947'5340, 0xaf2e'ad2c'6957'6351},
                                 {0xb736'b534'5967'5361, 0xbf3e'bd3c'6977'6370},
                                 {0xc746'c544'9987'9380, 0xcf4e'cd4c'a997'a391},
                                 {0xd756'd554'99a7'93a1, 0xdf5e'dd5c'a9b7'a3b0},
                                 {0xe766'e564'd9c7'd3c0, 0xef6e'ed6c'e9d7'e3d1},
                                 {0xf776'f574'd9e7'd3e1, 0xff7e'fd7c'e9f7'e3f0}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwadduwv) {
  TestWideningVectorInstruction(0xd10c2457,  // vwaddu.wv v8, v16, v24, v0.t
                                {{0x8100, 0x8394, 0x8508, 0x879c, 0x8911, 0x8ba4, 0x8d18, 0x8fac},
                                 {0x9121, 0x9394, 0x9528, 0x979c, 0x9930, 0x9ba4, 0x9d38, 0x9fac},
                                 {0xa140, 0xa3d4, 0xa548, 0xa7dc, 0xa951, 0xabe4, 0xad58, 0xafec},
                                 {0xb161, 0xb3d4, 0xb568, 0xb7dc, 0xb970, 0xbbe4, 0xbd78, 0xbfec},
                                 {0xc180, 0xc414, 0xc588, 0xc81c, 0xc991, 0xcc24, 0xcd98, 0xd02c},
                                 {0xd1a1, 0xd414, 0xd5a8, 0xd81c, 0xd9b0, 0xdc24, 0xddb8, 0xe02c},
                                 {0xe1c0, 0xe454, 0xe5c8, 0xe85c, 0xe9d1, 0xec64, 0xedd8, 0xf06c},
                                 {0xf1e1, 0xf454, 0xf5e8, 0xf85c, 0xf9f0, 0xfc64, 0xfdf8, 0x006c}},
                                {{0x8303'1300, 0x8707'1b08, 0x8b0b'2311, 0x8f0f'2b18},
                                 {0x9313'1321, 0x9717'1b28, 0x9b1b'2330, 0x9f1f'2b38},
                                 {0xa323'5340, 0xa727'5b48, 0xab2b'6351, 0xaf2f'6b58},
                                 {0xb333'5361, 0xb737'5b68, 0xbb3b'6370, 0xbf3f'6b78},
                                 {0xc343'9380, 0xc747'9b88, 0xcb4b'a391, 0xcf4f'ab98},
                                 {0xd353'93a1, 0xd757'9ba8, 0xdb5b'a3b0, 0xdf5f'abb8},
                                 {0xe363'd3c0, 0xe767'dbc8, 0xeb6b'e3d1, 0xef6f'ebd8},
                                 {0xf373'd3e1, 0xf777'dbe8, 0xfb7b'e3f0, 0xff7f'ebf8}},
                                {{0x8706'8505'1907'1300, 0x8f0e'8d0d'2917'2311},
                                 {0x9716'9515'1927'1321, 0x9f1e'9d1d'2937'2330},
                                 {0xa726'a525'5947'5340, 0xaf2e'ad2d'6957'6351},
                                 {0xb736'b535'5967'5361, 0xbf3e'bd3d'6977'6370},
                                 {0xc746'c545'9987'9380, 0xcf4e'cd4d'a997'a391},
                                 {0xd756'd555'99a7'93a1, 0xdf5e'dd5d'a9b7'a3b0},
                                 {0xe766'e565'd9c7'd3c0, 0xef6e'ed6d'e9d7'e3d1},
                                 {0xf776'f575'd9e7'd3e1, 0xff7e'fd7d'e9f7'e3f0}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwsubuwv) {
  TestWideningVectorInstruction(0xd90c2457,  // vwsubu.wv v8, v16, v24, v0.t
                                {{0x8100, 0x8270, 0x8500, 0x8670, 0x88ff, 0x8a70, 0x8d00, 0x8e70},
                                 {0x90ff, 0x9290, 0x9500, 0x9690, 0x9900, 0x9a90, 0x9d00, 0x9e90},
                                 {0xa100, 0xa270, 0xa500, 0xa670, 0xa8ff, 0xaa70, 0xad00, 0xae70},
                                 {0xb0ff, 0xb290, 0xb500, 0xb690, 0xb900, 0xba90, 0xbd00, 0xbe90},
                                 {0xc100, 0xc270, 0xc500, 0xc670, 0xc8ff, 0xca70, 0xcd00, 0xce70},
                                 {0xd0ff, 0xd290, 0xd500, 0xd690, 0xd900, 0xda90, 0xdd00, 0xde90},
                                 {0xe100, 0xe270, 0xe500, 0xe670, 0xe8ff, 0xea70, 0xed00, 0xee70},
                                 {0xf0ff, 0xf290, 0xf500, 0xf690, 0xf900, 0xfa90, 0xfd00, 0xfe90}},
                                {{0x8301'ef00, 0x8705'ef00, 0x8b09'eeff, 0x8f0d'ef00},
                                 {0x9312'0eff, 0x9716'0f00, 0x9b1a'0f00, 0x9f1e'0f00},
                                 {0xa321'ef00, 0xa725'ef00, 0xab29'eeff, 0xaf2d'ef00},
                                 {0xb332'0eff, 0xb736'0f00, 0xbb3a'0f00, 0xbf3e'0f00},
                                 {0xc341'ef00, 0xc745'ef00, 0xcb49'eeff, 0xcf4d'ef00},
                                 {0xd352'0eff, 0xd756'0f00, 0xdb5a'0f00, 0xdf5e'0f00},
                                 {0xe361'ef00, 0xe765'ef00, 0xeb69'eeff, 0xef6d'ef00},
                                 {0xf372'0eff, 0xf776'0f00, 0xfb7a'0f00, 0xff7e'0f00}},
                                {{0x8706'8503'ecfd'ef00, 0x8f0e'8d0b'ecfd'eeff},
                                 {0x9716'9514'0cfe'0eff, 0x9f1e'9d1c'0cfe'0f00},
                                 {0xa726'a523'ecfd'ef00, 0xaf2e'ad2b'ecfd'eeff},
                                 {0xb736'b534'0cfe'0eff, 0xbf3e'bd3c'0cfe'0f00},
                                 {0xc746'c543'ecfd'ef00, 0xcf4e'cd4b'ecfd'eeff},
                                 {0xd756'd554'0cfe'0eff, 0xdf5e'dd5c'0cfe'0f00},
                                 {0xe766'e563'ecfd'ef00, 0xef6e'ed6b'ecfd'eeff},
                                 {0xf776'f574'0cfe'0eff, 0xff7e'fd7c'0cfe'0f00}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVwsubwv) {
  TestWideningVectorInstruction(0xdd0c2457,  // vwsub.wv v8, v16, v24, v0.t
                                {{0x8100, 0x8370, 0x8500, 0x8770, 0x88ff, 0x8b70, 0x8d00, 0x8f70},
                                 {0x90ff, 0x9390, 0x9500, 0x9790, 0x9900, 0x9b90, 0x9d00, 0x9f90},
                                 {0xa100, 0xa370, 0xa500, 0xa770, 0xa8ff, 0xab70, 0xad00, 0xaf70},
                                 {0xb0ff, 0xb390, 0xb500, 0xb790, 0xb900, 0xbb90, 0xbd00, 0xbf90},
                                 {0xc100, 0xc370, 0xc500, 0xc770, 0xc8ff, 0xcb70, 0xcd00, 0xcf70},
                                 {0xd0ff, 0xd390, 0xd500, 0xd790, 0xd900, 0xdb90, 0xdd00, 0xdf90},
                                 {0xe100, 0xe370, 0xe500, 0xe770, 0xe8ff, 0xeb70, 0xed00, 0xef70},
                                 {0xf0ff, 0xf390, 0xf500, 0xf790, 0xf900, 0xfb90, 0xfd00, 0xff90}},
                                {{0x8302'ef00, 0x8706'ef00, 0x8b0a'eeff, 0x8f0e'ef00},
                                 {0x9313'0eff, 0x9717'0f00, 0x9b1b'0f00, 0x9f1f'0f00},
                                 {0xa322'ef00, 0xa726'ef00, 0xab2a'eeff, 0xaf2e'ef00},
                                 {0xb333'0eff, 0xb737'0f00, 0xbb3b'0f00, 0xbf3f'0f00},
                                 {0xc342'ef00, 0xc746'ef00, 0xcb4a'eeff, 0xcf4e'ef00},
                                 {0xd353'0eff, 0xd757'0f00, 0xdb5b'0f00, 0xdf5f'0f00},
                                 {0xe362'ef00, 0xe766'ef00, 0xeb6a'eeff, 0xef6e'ef00},
                                 {0xf373'0eff, 0xf777'0f00, 0xfb7b'0f00, 0xff7f'0f00}},
                                {{0x8706'8504'ecfd'ef00, 0x8f0e'8d0c'ecfd'eeff},
                                 {0x9716'9515'0cfe'0eff, 0x9f1e'9d1d'0cfe'0f00},
                                 {0xa726'a524'ecfd'ef00, 0xaf2e'ad2c'ecfd'eeff},
                                 {0xb736'b535'0cfe'0eff, 0xbf3e'bd3d'0cfe'0f00},
                                 {0xc746'c544'ecfd'ef00, 0xcf4e'cd4c'ecfd'eeff},
                                 {0xd756'd555'0cfe'0eff, 0xdf5e'dd5d'0cfe'0f00},
                                 {0xe766'e564'ecfd'ef00, 0xef6e'ed6c'ecfd'eeff},
                                 {0xf776'f575'0cfe'0eff, 0xff7e'fd7d'0cfe'0f00}},
                                kVectorCalculationsSource);
}

TEST_F(Riscv64InterpreterTest, TestVseXX) {
  TestVseXX(0x8427,  // vse8.v v8, (x1), v0.t
            {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
             {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
             {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 171, 44, 173, 46, 175},
             {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191},
             {64, 193, 66, 195, 68, 197, 70, 199, 72, 201, 74, 203, 76, 205, 78, 207},
             {80, 209, 82, 211, 84, 213, 86, 215, 88, 217, 90, 219, 92, 221, 94, 223},
             {96, 225, 98, 227, 100, 229, 102, 231, 104, 233, 106, 235, 108, 237, 110, 239},
             {112, 241, 114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255}},
            {},
            {},
            {},
            0,
            kVectorCalculationsSourceLegacy);
  TestVseXX(0xd427,  // vse16.v v8, (x1), v0.t
            {},
            {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
             {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
             {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
             {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
             {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
             {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
             {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
             {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
            {},
            {},
            1,
            kVectorCalculationsSourceLegacy);
  TestVseXX(0xe427,  // vse32.v v8, (x1), v0.t
            {},
            {},
            {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
             {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
             {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
             {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
             {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
             {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
             {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
             {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
            {},
            2,
            kVectorCalculationsSourceLegacy);
  TestVseXX(0xf427,  // vse64.v v8, (x1), v0.t
            {},
            {},
            {},
            {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
             {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
             {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
             {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
             {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
             {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
             {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
             {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
            3,
            kVectorCalculationsSourceLegacy);
}

TEST_F(Riscv64InterpreterTest, TestVleXX) {
  TestVleXX(0x8407,  // vle8.v v8, (x1), v0.t
            {{0, 129, 2, 131, 4, 133, 6, 135, 8, 137, 10, 139, 12, 141, 14, 143},
             {16, 145, 18, 147, 20, 149, 22, 151, 24, 153, 26, 155, 28, 157, 30, 159},
             {32, 161, 34, 163, 36, 165, 38, 167, 40, 169, 42, 171, 44, 173, 46, 175},
             {48, 177, 50, 179, 52, 181, 54, 183, 56, 185, 58, 187, 60, 189, 62, 191},
             {64, 193, 66, 195, 68, 197, 70, 199, 72, 201, 74, 203, 76, 205, 78, 207},
             {80, 209, 82, 211, 84, 213, 86, 215, 88, 217, 90, 219, 92, 221, 94, 223},
             {96, 225, 98, 227, 100, 229, 102, 231, 104, 233, 106, 235, 108, 237, 110, 239},
             {112, 241, 114, 243, 116, 245, 118, 247, 120, 249, 122, 251, 124, 253, 126, 255}},
            {},
            {},
            {},
            0,
            kVectorCalculationsSourceLegacy);
  TestVleXX(0xd407,  // vle16.v v8, (x1), v0.t
            {},
            {{0x8100, 0x8302, 0x8504, 0x8706, 0x8908, 0x8b0a, 0x8d0c, 0x8f0e},
             {0x9110, 0x9312, 0x9514, 0x9716, 0x9918, 0x9b1a, 0x9d1c, 0x9f1e},
             {0xa120, 0xa322, 0xa524, 0xa726, 0xa928, 0xab2a, 0xad2c, 0xaf2e},
             {0xb130, 0xb332, 0xb534, 0xb736, 0xb938, 0xbb3a, 0xbd3c, 0xbf3e},
             {0xc140, 0xc342, 0xc544, 0xc746, 0xc948, 0xcb4a, 0xcd4c, 0xcf4e},
             {0xd150, 0xd352, 0xd554, 0xd756, 0xd958, 0xdb5a, 0xdd5c, 0xdf5e},
             {0xe160, 0xe362, 0xe564, 0xe766, 0xe968, 0xeb6a, 0xed6c, 0xef6e},
             {0xf170, 0xf372, 0xf574, 0xf776, 0xf978, 0xfb7a, 0xfd7c, 0xff7e}},
            {},
            {},
            1,
            kVectorCalculationsSourceLegacy);
  TestVleXX(0xe407,  // vle32.v v8, (x1), v0.t
            {},
            {},
            {{0x8302'8100, 0x8706'8504, 0x8b0a'8908, 0x8f0e'8d0c},
             {0x9312'9110, 0x9716'9514, 0x9b1a'9918, 0x9f1e'9d1c},
             {0xa322'a120, 0xa726'a524, 0xab2a'a928, 0xaf2e'ad2c},
             {0xb332'b130, 0xb736'b534, 0xbb3a'b938, 0xbf3e'bd3c},
             {0xc342'c140, 0xc746'c544, 0xcb4a'c948, 0xcf4e'cd4c},
             {0xd352'd150, 0xd756'd554, 0xdb5a'd958, 0xdf5e'dd5c},
             {0xe362'e160, 0xe766'e564, 0xeb6a'e968, 0xef6e'ed6c},
             {0xf372'f170, 0xf776'f574, 0xfb7a'f978, 0xff7e'fd7c}},
            {},
            2,
            kVectorCalculationsSourceLegacy);
  TestVleXX(0xf407,  // vle64.v v8, (x1), v0.t
            {},
            {},
            {},
            {{0x8706'8504'8302'8100, 0x8f0e'8d0c'8b0a'8908},
             {0x9716'9514'9312'9110, 0x9f1e'9d1c'9b1a'9918},
             {0xa726'a524'a322'a120, 0xaf2e'ad2c'ab2a'a928},
             {0xb736'b534'b332'b130, 0xbf3e'bd3c'bb3a'b938},
             {0xc746'c544'c342'c140, 0xcf4e'cd4c'cb4a'c948},
             {0xd756'd554'd352'd150, 0xdf5e'dd5c'db5a'd958},
             {0xe766'e564'e362'e160, 0xef6e'ed6c'eb6a'e968},
             {0xf776'f574'f372'f170, 0xff7e'fd7c'fb7a'f978}},
            3,
            kVectorCalculationsSourceLegacy);
}

TEST_F(Riscv64InterpreterTest, TestVleXXff) {
  TestVleXXff(0x1008407, 6, 0, 6);  // vle8ff.v v8, (x1), v0.t
  TestVleXXff(0x1008407, 8, 0, 8);
  TestVleXXff(0x1008407, 16, 0, 16);
  TestVleXXff(0x1008407, 32, 0, 32);
  TestVleXXff(0x1008407, 255, 0, 128);  // All 128 bytes accessible.

  TestVleXXff(0x100d407, 6, 1, 3);  // vle16ff.v v8, (x1), v0.t
  TestVleXXff(0x100d407, 8, 1, 4);
  TestVleXXff(0x100d407, 16, 1, 8);
  TestVleXXff(0x100d407, 32, 1, 16);

  TestVleXXff(0x100e407, 6, 2, 1);  // vle32ff.v v8, (x1), v0.t
  TestVleXXff(0x100e407, 8, 2, 2);
  TestVleXXff(0x100e407, 16, 2, 4);
  TestVleXXff(0x100e407, 32, 2, 8);
  TestVleXXff(0x100e407, 64, 2, 16);

  TestVleXXff(0x100f407, 8, 3, 1);  // vle64ff.v v8, (x1), v0.t
  TestVleXXff(0x100f407, 16, 3, 2);
  TestVleXXff(0x100f407, 32, 3, 4);
  TestVleXXff(0x100f407, 64, 3, 8);

  TestVleXXff(0x100f407, 6, 3, 16, true);  // Should raise exception and not change VL
}

TEST_F(Riscv64InterpreterTest, TestVnsrl) {
  TestNarrowingVectorInstruction(
      0xb101b457,  // vnsrl.wi v8,v16,3,v0.t
      {{32, 96, 160, 224, 33, 97, 161, 225, 34, 98, 162, 226, 35, 99, 163, 227},
       {36, 100, 164, 228, 37, 101, 165, 229, 38, 102, 166, 230, 39, 103, 167, 231},
       {40, 104, 168, 232, 41, 105, 169, 233, 42, 106, 170, 234, 43, 107, 171, 235},
       {44, 108, 172, 236, 45, 109, 173, 237, 46, 110, 174, 238, 47, 111, 175, 239}},
      {{0x5020, 0xd0a0, 0x5121, 0xd1a1, 0x5222, 0xd2a2, 0x5323, 0xd3a3},
       {0x5424, 0xd4a4, 0x5525, 0xd5a5, 0x5626, 0xd6a6, 0x5727, 0xd7a7},
       {0x5828, 0xd8a8, 0x5929, 0xd9a9, 0x5a2a, 0xdaaa, 0x5b2b, 0xdbab},
       {0x5c2c, 0xdcac, 0x5d2d, 0xddad, 0x5e2e, 0xdeae, 0x5f2f, 0xdfaf}},
      {{0x9060'5020, 0x9161'5121, 0x9262'5222, 0x9363'5323},
       {0x9464'5424, 0x9565'5525, 0x9666'5626, 0x9767'5727},
       {0x9868'5828, 0x9969'5929, 0x9a6a'5a2a, 0x9b6b'5b2b},
       {0x9c6c'5c2c, 0x9d6d'5d2d, 0x9e6e'5e2e, 0x9f6f'5f2f}},
      kVectorCalculationsSourceLegacy);
  TestNarrowingVectorInstruction(0xb100c457,  // vnsrl.wx v8,v16,x1,v0.t
                                 {{32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39},
                                  {40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47},
                                  {48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55},
                                  {56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63}},
                                 {{0xc0a0, 0xc1a1, 0xc2a2, 0xc3a3, 0xc4a4, 0xc5a5, 0xc6a6, 0xc7a7},
                                  {0xc8a8, 0xc9a9, 0xcaaa, 0xcbab, 0xccac, 0xcdad, 0xceae, 0xcfaf},
                                  {0xd0b0, 0xd1b1, 0xd2b2, 0xd3b3, 0xd4b4, 0xd5b5, 0xd6b6, 0xd7b7},
                                  {0xd8b8, 0xd9b9, 0xdaba, 0xdbbb, 0xdcbc, 0xddbd, 0xdebe, 0xdfbf}},
                                 {{0x0021'c1a1, 0x0023'c3a3, 0x0025'c5a5, 0x0027'c7a7},
                                  {0x0029'c9a9, 0x002b'cbab, 0x002d'cdad, 0x002f'cfaf},
                                  {0x0031'd1b1, 0x0033'd3b3, 0x0035'd5b5, 0x0037'd7b7},
                                  {0x0039'd9b9, 0x003b'dbbb, 0x003d'ddbd, 0x003f'dfbf}},
                                 kVectorCalculationsSourceLegacy);
  TestNarrowingVectorInstruction(
      0xb10c0457,  // vnsrl.wv v8,v16,v24,v0.t
      {{0, 192, 80, 28, 68, 34, 8, 2, 136, 196, 81, 92, 153, 38, 9, 2},
       {32, 200, 82, 156, 84, 42, 10, 2, 152, 204, 83, 220, 185, 46, 11, 2},
       {64, 208, 84, 29, 100, 50, 12, 3, 168, 212, 85, 93, 217, 54, 13, 3},
       {96, 216, 86, 157, 116, 58, 14, 3, 184, 220, 87, 221, 249, 62, 15, 3}},
      {{0x8100, 0x6850, 0x8544, 0xf0e8, 0x4989, 0x0971, 0x009b, 0x0009},
       {0xa120, 0x6a52, 0x9554, 0xf2ea, 0x5999, 0x0b73, 0x00bb, 0x000b},
       {0xc140, 0x6c54, 0xa564, 0xf4ec, 0x69a9, 0x0d75, 0x00db, 0x000d},
       {0xe160, 0x6e56, 0xb574, 0xf6ee, 0x79b9, 0x0f77, 0x00fb, 0x000f}},
      {{0x8302'8100, 0x8645'8544, 0x4a8a'4989, 0x1e9d'1c9b},
       {0xa726'a524, 0x0057'9756, 0x0000'5b9b, 0x0000'00bf},
       {0xc342'c140, 0xa665'a564, 0x6aaa'69a9, 0x5edd'5cdb},
       {0xe766'e564, 0x0077'b776, 0x0000'7bbb, 0x0000'00ff}},
      kVectorCalculationsSourceLegacy);
}

TEST_F(Riscv64InterpreterTest, TestVnsra) {
  TestNarrowingVectorInstruction(
      0xb501b457,  // vnsra.wi v8,v16,3,v0.t
      {{32, 96, 160, 224, 33, 97, 161, 225, 34, 98, 162, 226, 35, 99, 163, 227},
       {36, 100, 164, 228, 37, 101, 165, 229, 38, 102, 166, 230, 39, 103, 167, 231},
       {40, 104, 168, 232, 41, 105, 169, 233, 42, 106, 170, 234, 43, 107, 171, 235},
       {44, 108, 172, 236, 45, 109, 173, 237, 46, 110, 174, 238, 47, 111, 175, 239}},
      {{0x5020, 0xd0a0, 0x5121, 0xd1a1, 0x5222, 0xd2a2, 0x5323, 0xd3a3},
       {0x5424, 0xd4a4, 0x5525, 0xd5a5, 0x5626, 0xd6a6, 0x5727, 0xd7a7},
       {0x5828, 0xd8a8, 0x5929, 0xd9a9, 0x5a2a, 0xdaaa, 0x5b2b, 0xdbab},
       {0x5c2c, 0xdcac, 0x5d2d, 0xddad, 0x5e2e, 0xdeae, 0x5f2f, 0xdfaf}},
      {{0x9060'5020, 0x9161'5121, 0x9262'5222, 0x9363'5323},
       {0x9464'5424, 0x9565'5525, 0x9666'5626, 0x9767'5727},
       {0x9868'5828, 0x9969'5929, 0x9a6a'5a2a, 0x9b6b'5b2b},
       {0x9c6c'5c2c, 0x9d6d'5d2d, 0x9e6e'5e2e, 0x9f6f'5f2f}},
      kVectorCalculationsSourceLegacy);
  TestNarrowingVectorInstruction(
      0xb500c457,  // vnsra.wx v8,v16,x1,v0.t
      {{224, 224, 225, 225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231},
       {232, 232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 239},
       {240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247},
       {248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255}},
      {{0xc0a0, 0xc1a1, 0xc2a2, 0xc3a3, 0xc4a4, 0xc5a5, 0xc6a6, 0xc7a7},
       {0xc8a8, 0xc9a9, 0xcaaa, 0xcbab, 0xccac, 0xcdad, 0xceae, 0xcfaf},
       {0xd0b0, 0xd1b1, 0xd2b2, 0xd3b3, 0xd4b4, 0xd5b5, 0xd6b6, 0xd7b7},
       {0xd8b8, 0xd9b9, 0xdaba, 0xdbbb, 0xdcbc, 0xddbd, 0xdebe, 0xdfbf}},
      {{0xffe1'c1a1, 0xffe3'c3a3, 0xffe5'c5a5, 0xffe7'c7a7},
       {0xffe9'c9a9, 0xffeb'cbab, 0xffed'cdad, 0xffef'cfaf},
       {0xfff1'd1b1, 0xfff3'd3b3, 0xfff5'd5b5, 0xfff7'd7b7},
       {0xfff9'd9b9, 0xfffb'dbbb, 0xfffd'ddbd, 0xffff'dfbf}},
      kVectorCalculationsSourceLegacy);
  TestNarrowingVectorInstruction(
      0xb50c0457,  // vnsra.wv v8,v16,v24,v0.t
      {{0, 192, 80, 28, 196, 226, 248, 254, 136, 196, 81, 92, 153, 230, 249, 254},
       {32, 200, 82, 156, 212, 234, 250, 254, 152, 204, 83, 220, 185, 238, 251, 254},
       {64, 208, 84, 29, 228, 242, 252, 255, 168, 212, 85, 93, 217, 246, 253, 255},
       {96, 216, 86, 157, 244, 250, 254, 255, 184, 220, 87, 221, 249, 254, 255, 255}},
      {{0x8100, 0x6850, 0x8544, 0xf0e8, 0xc989, 0xf971, 0xff9b, 0xfff9},
       {0xa120, 0x6a52, 0x9554, 0xf2ea, 0xd999, 0xfb73, 0xffbb, 0xfffb},
       {0xc140, 0x6c54, 0xa564, 0xf4ec, 0xe9a9, 0xfd75, 0xffdb, 0xfffd},
       {0xe160, 0x6e56, 0xb574, 0xf6ee, 0xf9b9, 0xff77, 0xfffb, 0xffff}},
      {{0x8302'8100, 0x8645'8544, 0x4a8a'4989, 0x1e9d'1c9b},
       {0xa726'a524, 0xffd7'9756, 0xffff'db9b, 0xffff'ffbf},
       {0xc342'c140, 0xa665'a564, 0x6aaa'69a9, 0x5edd'5cdb},
       {0xe766'e564, 0xfff7'b776, 0xffff'fbbb, 0xffff'ffff}},
      kVectorCalculationsSourceLegacy);
}

TEST_F(Riscv64InterpreterTest, TestVcpopm) {
  TestVXmXXsInstruction(
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
      kVectorCalculationsSourceLegacy[0]);
}

TEST_F(Riscv64InterpreterTest, TestVfirstm) {
  TestVXmXXsInstruction(
      0x4108a0d7,  // vfirst.m x1, v16, v0.t
      { [0 ... 8] = ~0ULL,
        [9 ... 128] = 8 },
      { [0 ... 8] = ~0ULL,
        [9 ... 128] = 8 },
      kVectorCalculationsSourceLegacy[0]);
}

TEST_F(Riscv64InterpreterTest, TestVXext) {
  TestExtendingVectorInstruction(
      0x49012457,  // vzext.vf8 v8,v16,v0.t
      {}, {},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0081},
       {0x0000'0000'0000'0002, 0x0000'0000'0000'0083},
       {0x0000'0000'0000'0004, 0x0000'0000'0000'0085},
       {0x0000'0000'0000'0006, 0x0000'0000'0000'0087},
       {0x0000'0000'0000'0008, 0x0000'0000'0000'0089},
       {0x0000'0000'0000'000a, 0x0000'0000'0000'008b},
       {0x0000'0000'0000'000c, 0x0000'0000'0000'008d},
       {0x0000'0000'0000'000e, 0x0000'0000'0000'008f}},
      kVectorCalculationsSource,
      8);

  TestExtendingVectorInstruction(
      0x4901a457,  // vsext.vf8 v8,v16,v0.t
      {}, {},
      {{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ff81},
       {0x0000'0000'0000'0002, 0xffff'ffff'ffff'ff83},
       {0x0000'0000'0000'0004, 0xffff'ffff'ffff'ff85},
       {0x0000'0000'0000'0006, 0xffff'ffff'ffff'ff87},
       {0x0000'0000'0000'0008, 0xffff'ffff'ffff'ff89},
       {0x0000'0000'0000'000a, 0xffff'ffff'ffff'ff8b},
       {0x0000'0000'0000'000c, 0xffff'ffff'ffff'ff8d},
       {0x0000'0000'0000'000e, 0xffff'ffff'ffff'ff8f}},
      kVectorCalculationsSource,
      8);

  TestExtendingVectorInstruction(
      0x49022457,  // vzext.vf4 v8, v16, v0.t
      {},
      {{0x0000'0000, 0x0000'0081, 0x0000'0002, 0x0000'0083},
       {0x0000'0004, 0x0000'0085, 0x0000'0006, 0x0000'0087},
       {0x0000'0008, 0x0000'0089, 0x0000'000a, 0x0000'008b},
       {0x0000'000c, 0x0000'008d, 0x0000'000e, 0x0000'008f},
       {0x0000'0010, 0x0000'0091, 0x0000'0012, 0x0000'0093},
       {0x0000'0014, 0x0000'0095, 0x0000'0016, 0x0000'0097},
       {0x0000'0018, 0x0000'0099, 0x0000'001a, 0x0000'009b},
       {0x0000'001c, 0x0000'009d, 0x0000'001e, 0x0000'009f}},
      {{0x0000'0000'0000'8100, 0x0000'0000'0000'8302},
       {0x0000'0000'0000'8504, 0x0000'0000'0000'8706},
       {0x0000'0000'0000'8908, 0x0000'0000'0000'8b0a},
       {0x0000'0000'0000'8d0c, 0x0000'0000'0000'8f0e},
       {0x0000'0000'0000'9110, 0x0000'0000'0000'9312},
       {0x0000'0000'0000'9514, 0x0000'0000'0000'9716},
       {0x0000'0000'0000'9918, 0x0000'0000'0000'9b1a},
       {0x0000'0000'0000'9d1c, 0x0000'0000'0000'9f1e}},
      kVectorCalculationsSource,
      4);

  TestExtendingVectorInstruction(
      0x4902a457,  // vsext.vf4 v8,v16,v0.t
      {},
      {{0x0000'0000, 0xffff'ff81, 0x0000'0002, 0xffff'ff83},
       {0x0000'0004, 0xffff'ff85, 0x0000'0006, 0xffff'ff87},
       {0x0000'0008, 0xffff'ff89, 0x0000'000a, 0xffff'ff8b},
       {0x0000'000c, 0xffff'ff8d, 0x0000'000e, 0xffff'ff8f},
       {0x0000'0010, 0xffff'ff91, 0x0000'0012, 0xffff'ff93},
       {0x0000'0014, 0xffff'ff95, 0x0000'0016, 0xffff'ff97},
       {0x0000'0018, 0xffff'ff99, 0x0000'001a, 0xffff'ff9b},
       {0x0000'001c, 0xffff'ff9d, 0x0000'001e, 0xffff'ff9f}},
      {{0xffff'ffff'ffff'8100, 0xffff'ffff'ffff'8302},
       {0xffff'ffff'ffff'8504, 0xffff'ffff'ffff'8706},
       {0xffff'ffff'ffff'8908, 0xffff'ffff'ffff'8b0a},
       {0xffff'ffff'ffff'8d0c, 0xffff'ffff'ffff'8f0e},
       {0xffff'ffff'ffff'9110, 0xffff'ffff'ffff'9312},
       {0xffff'ffff'ffff'9514, 0xffff'ffff'ffff'9716},
       {0xffff'ffff'ffff'9918, 0xffff'ffff'ffff'9b1a},
       {0xffff'ffff'ffff'9d1c, 0xffff'ffff'ffff'9f1e}},
      kVectorCalculationsSource,
      4);

  TestExtendingVectorInstruction(
      0x49032457,  // vzext.vf2 v8,v16,v0.t
      {{0x0000, 0x0081, 0x0002, 0x0083, 0x0004, 0x0085, 0x0006, 0x0087},
       {0x0008, 0x0089, 0x000a, 0x008b, 0x000c, 0x008d, 0x000e, 0x008f},
       {0x0010, 0x0091, 0x0012, 0x0093, 0x0014, 0x0095, 0x0016, 0x0097},
       {0x0018, 0x0099, 0x001a, 0x009b, 0x001c, 0x009d, 0x001e, 0x009f},
       {0x0020, 0x00a1, 0x0022, 0x00a3, 0x0024, 0x00a5, 0x0026, 0x00a7},
       {0x0028, 0x00a9, 0x002a, 0x00ab, 0x002c, 0x00ad, 0x002e, 0x00af},
       {0x0030, 0x00b1, 0x0032, 0x00b3, 0x0034, 0x00b5, 0x0036, 0x00b7},
       {0x0038, 0x00b9, 0x003a, 0x00bb, 0x003c, 0x00bd, 0x003e, 0x00bf}},
      {{0x0000'8100, 0x0000'8302, 0x0000'8504, 0x0000'8706},
       {0x0000'8908, 0x0000'8b0a, 0x0000'8d0c, 0x0000'8f0e},
       {0x0000'9110, 0x0000'9312, 0x0000'9514, 0x0000'9716},
       {0x0000'9918, 0x0000'9b1a, 0x0000'9d1c, 0x0000'9f1e},
       {0x0000'a120, 0x0000'a322, 0x0000'a524, 0x0000'a726},
       {0x0000'a928, 0x0000'ab2a, 0x0000'ad2c, 0x0000'af2e},
       {0x0000'b130, 0x0000'b332, 0x0000'b534, 0x0000'b736},
       {0x0000'b938, 0x0000'bb3a, 0x0000'bd3c, 0x0000'bf3e}},
      {{0x0000'0000'8302'8100, 0x0000'0000'8706'8504},
       {0x0000'0000'8b0a'8908, 0x0000'0000'8f0e'8d0c},
       {0x0000'0000'9312'9110, 0x0000'0000'9716'9514},
       {0x0000'0000'9b1a'9918, 0x0000'0000'9f1e'9d1c},
       {0x0000'0000'a322'a120, 0x0000'0000'a726'a524},
       {0x0000'0000'ab2a'a928, 0x0000'0000'af2e'ad2c},
       {0x0000'0000'b332'b130, 0x0000'0000'b736'b534},
       {0x0000'0000'bb3a'b938, 0x0000'0000'bf3e'bd3c}},
      kVectorCalculationsSource,
      2);

  TestExtendingVectorInstruction(
      0x4903a457,  // vsext.vf2 v8,v16,v0.t
      {{0x0000, 0xff81, 0x0002, 0xff83, 0x0004, 0xff85, 0x0006, 0xff87},
       {0x0008, 0xff89, 0x000a, 0xff8b, 0x000c, 0xff8d, 0x000e, 0xff8f},
       {0x0010, 0xff91, 0x0012, 0xff93, 0x0014, 0xff95, 0x0016, 0xff97},
       {0x0018, 0xff99, 0x001a, 0xff9b, 0x001c, 0xff9d, 0x001e, 0xff9f},
       {0x0020, 0xffa1, 0x0022, 0xffa3, 0x0024, 0xffa5, 0x0026, 0xffa7},
       {0x0028, 0xffa9, 0x002a, 0xffab, 0x002c, 0xffad, 0x002e, 0xffaf},
       {0x0030, 0xffb1, 0x0032, 0xffb3, 0x0034, 0xffb5, 0x0036, 0xffb7},
       {0x0038, 0xffb9, 0x003a, 0xffbb, 0x003c, 0xffbd, 0x003e, 0xffbf}},
      {{0xffff'8100, 0xffff'8302, 0xffff'8504, 0xffff'8706},
       {0xffff'8908, 0xffff'8b0a, 0xffff'8d0c, 0xffff'8f0e},
       {0xffff'9110, 0xffff'9312, 0xffff'9514, 0xffff'9716},
       {0xffff'9918, 0xffff'9b1a, 0xffff'9d1c, 0xffff'9f1e},
       {0xffff'a120, 0xffff'a322, 0xffff'a524, 0xffff'a726},
       {0xffff'a928, 0xffff'ab2a, 0xffff'ad2c, 0xffff'af2e},
       {0xffff'b130, 0xffff'b332, 0xffff'b534, 0xffff'b736},
       {0xffff'b938, 0xffff'bb3a, 0xffff'bd3c, 0xffff'bf3e}},
      {{0xffff'ffff'8302'8100, 0xffff'ffff'8706'8504},
       {0xffff'ffff'8b0a'8908, 0xffff'ffff'8f0e'8d0c},
       {0xffff'ffff'9312'9110, 0xffff'ffff'9716'9514},
       {0xffff'ffff'9b1a'9918, 0xffff'ffff'9f1e'9d1c},
       {0xffff'ffff'a322'a120, 0xffff'ffff'a726'a524},
       {0xffff'ffff'ab2a'a928, 0xffff'ffff'af2e'ad2c},
       {0xffff'ffff'b332'b130, 0xffff'ffff'b736'b534},
       {0xffff'ffff'bb3a'b938, 0xffff'ffff'bf3e'bd3c}},
      kVectorCalculationsSource,
      2);
}

// For Vrgather the expectations for different VLMULs are very different, i.e. different vlmax
// values produce different results for the same src. So we have to call each of them explicitly.
// To produce none-all-zero expected_results, all registers index vectors(v24) have first half
// masked to have valid indexes (< vlmax); x1 is masked to be < vlmax; uimm is accordingly to be 0,
// 1 or 3.
TEST_F(Riscv64InterpreterTest, TestVrgather) {
  // VLMUL = 0
  TestVectorRegisterGather(0x310c0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 2, 0, 6, 0, 10, 0, 14},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 137, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x8302, 0x8908, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000}},
                           {{0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000}},
                           /*vlmul=*/0,
                           kVectorCalculationsSource);

  TestVectorRegisterGather(0x3100c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}},
                           {{0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504}},
                           {{0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908}},
                           {{0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100}},
                           /*vlmul=*/0,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302}},
      {{0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504}},
      {{0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908}},
      /*vlmul=*/0,
      kVectorCalculationsSource);

  // VLMUL = 1
  TestVectorRegisterGather(0x310c0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 18, 4, 22, 137, 26, 12, 30, 145, 0, 20, 0, 24, 0, 28, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 2, 0, 6, 0, 10, 0, 14},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000}},
                           {{0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000}},
                           /*vlmul=*/1,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
                            {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}},
                           {{0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514}},
                           {{0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908},
                            {0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908, 0x8b0a'8908}},
                           {{0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110}},
                           /*vlmul=*/1,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302}},
      {{0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504}},
      {{0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908}},
      /*vlmul=*/1,
      kVectorCalculationsSource);

  // VLMUL = 2
  TestVectorRegisterGather(0x310c0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 18, 4, 22, 137, 26, 12, 30, 145, 0, 20, 0, 24, 0, 28, 0},
                            {32, 50, 36, 54, 169, 58, 44, 62, 177, 0, 52, 0, 56, 0, 60, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {32, 50, 36, 54, 169, 58, 44, 62, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 2, 0, 6, 0, 10, 0, 14},
                            {32, 50, 36, 54, 169, 58, 44, 62, 0, 34, 0, 38, 0, 42, 0, 46},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 0, 0, 0, 0, 0, 0, 0},
                            {32, 50, 36, 54, 169, 58, 44, 62, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000}},
                           {{0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000}},
                           /*vlmul=*/2,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42}},
                           {{0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514},
                            {0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514, 0x9514}},
                           {{0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928}},
                           {{0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110},
                            {0x9716'9514'9312'9110, 0x9716'9514'9312'9110}},
                           /*vlmul=*/2,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302}},
      {{0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504}},
      {{0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908},
       {0x8f0e'8d0c'8b0a'8908, 0x8f0e'8d0c'8b0a'8908}},
      /*vlmul=*/2,
      kVectorCalculationsSource);

  // VLMUL = 3
  TestVectorRegisterGather(0x310c0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 18, 4, 22, 137, 26, 12, 30, 145, 0, 20, 0, 24, 0, 28, 0},
                            {32, 50, 36, 54, 169, 58, 44, 62, 177, 0, 52, 0, 56, 0, 60, 0},
                            {64, 82, 68, 86, 201, 90, 76, 94, 209, 0, 84, 0, 88, 0, 92, 0},
                            {96, 114, 100, 118, 233, 122, 108, 126, 241, 0, 116, 0, 120, 0, 124, 0},
                            {0, 18, 4, 22, 137, 26, 12, 30, 0, 2, 0, 6, 0, 10, 0, 14},
                            {32, 50, 36, 54, 169, 58, 44, 62, 0, 34, 0, 38, 0, 42, 0, 46},
                            {64, 82, 68, 86, 201, 90, 76, 94, 0, 66, 0, 70, 0, 74, 0, 78},
                            {96, 114, 100, 118, 233, 122, 108, 126, 0, 98, 0, 102, 0, 106, 0, 110}},
                           {{0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0xc140, 0xc948, 0xd352, 0xd958, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0xc140, 0xc948, 0xd352, 0xd958, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0xc140, 0xc948, 0xd352, 0xd958, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8908, 0x9312, 0x9918, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0xc140, 0xc948, 0xd352, 0xd958, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0xa726'a524, 0x0000'0000, 0x0000'0000}},
                           {{0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000}},
                           /*vlmul=*/3,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
                            {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42}},
                           {{0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554},
                            {0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554, 0xd554}},
                           {{0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928},
                            {0xab2a'a928, 0xab2a'a928, 0xab2a'a928, 0xab2a'a928}},
                           {{0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150},
                            {0xd756'd554'd352'd150, 0xd756'd554'd352'd150}},
                           /*vlmul=*/3,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3101b457,  // vrgather.vi v8,v16,3,v0.t
      {{131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131},
       {131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131}},
      {{0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706},
       {0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706, 0x8706}},
      {{0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c},
       {0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c, 0x8f0e'8d0c}},
      {{0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918},
       {0x9f1e'9d1c'9b1a'9918, 0x9f1e'9d1c'9b1a'9918}},
      /*vlmul=*/3,
      kVectorCalculationsSource);

  // VLMUL = 5
  TestVectorRegisterGather(0x310c0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {},
                           {},
                           /*vlmul=*/5,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100}},
                           {},
                           {},
                           /*vlmul=*/5,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000},
       {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}},
      {},
      {},
      /*vlmul=*/5,
      kVectorCalculationsSource);

  // VLMUL = 6
  TestVectorRegisterGather(0x310c0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 0, 2, 129, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8302'8100, 0x0000'0000, 0x0000'0000}},
                           {},
                           /*vlmul=*/6,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}},
                           {{0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100},
                            {0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100, 0x8100}},
                           {{0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100}},
                           {},
                           /*vlmul=*/6,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302}},
      {{0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
       {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000}},
      {},
      /*vlmul=*/6,
      kVectorCalculationsSource);

  // VLMUL = 7
  TestVectorRegisterGather(0x310c0457,  // vrgather.vv v8,v16,v24,v0.t
                           {{0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 2, 0, 6, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 4, 6, 129, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0}},
                           {{0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000},
                            {0x8100, 0x8100, 0x8302, 0x8100, 0x0000, 0x0000, 0x0000, 0x0000}},
                           {{0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000},
                            {0x8302'8100, 0x8706'8504, 0x0000'0000, 0x0000'0000}},
                           {{0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000},
                            {0x8706'8504'8302'8100, 0x0000'0000'0000'0000}},
                           /*vlmul=*/7,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(0x3100c457,  // vrgather.vx v8,v16,x1,v0.t
                           {{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                            {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}},
                           {{0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504},
                            {0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504, 0x8504}},
                           {{0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100},
                            {0x8302'8100, 0x8302'8100, 0x8302'8100, 0x8302'8100}},
                           {{0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100},
                            {0x8706'8504'8302'8100, 0x8706'8504'8302'8100}},
                           /*vlmul=*/7,
                           kVectorCalculationsSource);
  TestVectorRegisterGather(
      0x3100b457,  // vrgather.vi v8,v16,1,v0.t
      {{129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129},
       {129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129}},
      {{0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302},
       {0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302, 0x8302}},
      {{0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504},
       {0x8706'8504, 0x8706'8504, 0x8706'8504, 0x8706'8504}},
      {{0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000},
       {0x0000'0000'0000'0000, 0x0000'0000'0000'0000}},
      /*vlmul=*/7,
      kVectorCalculationsSource);
}

}  // namespace

}  // namespace berberis
