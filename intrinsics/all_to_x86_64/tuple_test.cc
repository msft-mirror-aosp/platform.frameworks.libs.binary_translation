/*
 * Copyright (C) 2019 The Android Open Source Project
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

#include <cinttypes>
#include <optional>
#include <tuple>

#include "berberis/base/bit_util.h"
#include "berberis/intrinsics/simd_register.h"

#pragma clang diagnostic push
// Clang does not allow use of C++ types in “extern "C"” functions - but we need to declare one to
// test it.
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

extern "C" std::tuple<uint64_t> AsmTupleTestI64(std::tuple<uint64_t>*);

// This function takes first parameter %rdi and uses it as the address of a tuple.
// If tuple is returned on registers it would contain the address of a tuple passed via pointer.
// If tuple is returned on stack this would be address of the returned tuple (hidden parameter).
asm(R"(.p2align 4, 0x90
       .type AsmTupleTestI64,@function
       AsmTupleTestI64:
       .cfi_startproc
       movl $42, (%rdi)
       movq %rdi, %rax
       ret
       .size AsmTupleTestI64, .-AsmTupleTestI64
       .cfi_endproc)");

extern "C" std::tuple<berberis::SIMD128Register, berberis::SIMD128Register>
AsmTupleTestSIMDRegisterSIMDRegister(
    std::tuple<berberis::SIMD128Register, berberis::SIMD128Register>*);

// This function takes first parameter %rdi and uses it as the address of a tuple.
// If tuple is returned on registers it would contain the address of a tuple passed via pointer.
// The function returns {this pointer, 0, garbage, garbage} in that case.
// If tuple is returned on stack this would be address of the returned tuple (hidden parameter).
// The function returns {1, 2, 3, 4} in that case.
asm(R"(.p2align 4, 0x90
       .type AsmTupleTestSIMDRegisterSIMDRegister,@function
       AsmTupleTestSIMDRegisterSIMDRegister:
       .cfi_startproc
       movq $1, (%rdi)
       movq $2, 8(%rdi)
       movq $3, 16(%rdi)
       movq $4, 24(%rdi)
       movq %rdi, %rax
       movq %rdi, %xmm0
       ret
       .size AsmTupleTestSIMDRegisterSIMDRegister, .-AsmTupleTestSIMDRegisterSIMDRegister
       .cfi_endproc)");

extern "C" std::tuple<berberis::SIMD128Register,
                      berberis::SIMD128Register,
                      berberis::SIMD128Register>
AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegister(
    std::tuple<berberis::SIMD128Register, berberis::SIMD128Register, berberis::SIMD128Register>*);

// This function takes first parameter %rdi and uses it as the address of a tuple.
// If tuple is returned on registers it would contain the address of a tuple passed via pointer.
// The function returns {this pointer, 0, garbage, garbage} in that case.
// If tuple is returned on stack this would be address of the returned tuple (hidden parameter).
// The function returns {1, 2, 3, 4, 5, 6} in that case.
asm(R"(.p2align 4, 0x90
       .type AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegister,@function
       AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegister:
       .cfi_startproc
       movq $1, (%rdi)
       movq $2, 8(%rdi)
       movq $3, 16(%rdi)
       movq $4, 24(%rdi)
       movq $5, 32(%rdi)
       movq $6, 40(%rdi)
       movq %rdi, %rax
       movq %rdi, %xmm0
       ret
       .size AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegister, .-AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegister
       .cfi_endproc)");

extern "C" std::tuple<berberis::SIMD128Register,
                      berberis::SIMD128Register,
                      berberis::SIMD128Register,
                      berberis::SIMD128Register>
AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegisterSIMDRegister(
    std::tuple<berberis::SIMD128Register,
               berberis::SIMD128Register,
               berberis::SIMD128Register,
               berberis::SIMD128Register>*);

// This function takes first parameter %rdi and uses it as the address of a tuple.
// If tuple is returned on registers it would contain the address of a tuple passed via pointer.
// The function returns {this pointer, 0, garbage, garbage} in that case.
// If tuple is returned on stack this would be address of the returned tuple (hidden parameter).
// The function returns {1, 2, 3, 4, 5, 6, 7, 8} in that case.
asm(R"(.p2align 4, 0x90
       .type AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegisterSIMDRegister,@function
       AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegisterSIMDRegister:
       .cfi_startproc
       movq $1, (%rdi)
       movq $2, 8(%rdi)
       movq $3, 16(%rdi)
       movq $4, 24(%rdi)
       movq $5, 32(%rdi)
       movq $6, 40(%rdi)
       movq $7, 48(%rdi)
       movq $8, 56(%rdi)
       movq %rdi, %rax
       movq %rdi, %xmm0
       ret
       .size AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegisterSIMDRegister, .-AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegisterSIMDRegister
       .cfi_endproc)");

#pragma clang diagnostic pop

namespace berberis {

namespace {

template <typename T, T AsmTupleTest(T*), typename ExpectedValue, typename ExpectedZeroValue>
std::optional<bool> TupleIsReturnedOnRegisters(ExpectedValue kExpectedValue,
                                               ExpectedZeroValue kExpectedZeroValue) {
  T result_if_on_regs{};
  T result_if_on_stack{};
  result_if_on_stack = AsmTupleTest(&result_if_on_regs);
  if (result_if_on_regs == kExpectedValue) {
    // When result is on regs function returns a pointer to result_if_on_regs.
    void* result_if_on_regs_ptr = &result_if_on_regs;
    static_assert(sizeof(result_if_on_regs_ptr) <= sizeof(result_if_on_stack));
    if (memcmp(&result_if_on_stack, &result_if_on_regs_ptr, sizeof(result_if_on_regs_ptr)) == 0) {
      return true;
    }
    // Shouldn't happen with proper x86-64 compiler.
    return {};
  } else if (result_if_on_regs == kExpectedZeroValue && result_if_on_stack == kExpectedValue) {
    return false;
  } else {
    // Shouldn't happen with proper x86-64 compiler.
    return {};
  }
}

// Note: tuple is returned on registers when libc++ is used and on stack if libstdc++ is used.
TEST(LibCxxAbi, Tuple_UInt64) {
  auto tuple_is_returned_on_registers =
      TupleIsReturnedOnRegisters<std::tuple<uint64_t>, AsmTupleTestI64>(std::tuple{uint64_t{42}},
                                                                        std::tuple{uint64_t{0}});
  ASSERT_TRUE(tuple_is_returned_on_registers.has_value());
#ifdef _LIBCPP_VERSION
  EXPECT_TRUE(*tuple_is_returned_on_registers);
#else
  EXPECT_FALSE(*tuple_is_returned_on_registers);
#endif
}

// Note: tuple is returned on registers when libc++ is used and on stack if libstdc++ is used.
TEST(LibCxxAbi, Tuple_SIMDRegisterSIMDRegister) {
  auto tuple_is_returned_on_registers =
      TupleIsReturnedOnRegisters<std::tuple<SIMD128Register, SIMD128Register>,
                                 AsmTupleTestSIMDRegisterSIMDRegister>(
          std::tuple{Int64x2{1, 2}, Int64x2{3, 4}}, std::tuple{Int64x2{0, 0}, Int64x2{0, 0}});
  ASSERT_TRUE(tuple_is_returned_on_registers.has_value());
#if defined(_LIBCPP_VERSION) && defined(__AVX__)
  EXPECT_TRUE(*tuple_is_returned_on_registers);
#else
  EXPECT_FALSE(*tuple_is_returned_on_registers);
#endif
}

// Note: tuple is returned on registers when libc++ is used and on stack if libstdc++ is used.
TEST(LibCxxAbi, Tuple_SIMDRegisterSIMDRegisterSIMDRegister) {
  auto tuple_is_returned_on_registers =
      TupleIsReturnedOnRegisters<std::tuple<SIMD128Register, SIMD128Register, SIMD128Register>,
                                 AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegister>(
          std::tuple{Int64x2{1, 2}, Int64x2{3, 4}, Int64x2{5, 6}},
          std::tuple{Int64x2{0, 0}, Int64x2{0, 0}, Int64x2{0, 0}});
  ASSERT_TRUE(tuple_is_returned_on_registers.has_value());
  EXPECT_FALSE(*tuple_is_returned_on_registers);
}

// Note: tuple is returned on registers when libc++ is used and on stack if libstdc++ is used.
TEST(LibCxxAbi, Tuple_SIMDRegisterSIMDRegisterSIMDRegisterSIMDRegister) {
  auto tuple_is_returned_on_registers = TupleIsReturnedOnRegisters<
      std::tuple<SIMD128Register, SIMD128Register, SIMD128Register, SIMD128Register>,
      AsmTupleTestSIMDRegisterSIMDRegisterSIMDRegisterSIMDRegister>(
      std::tuple{Int64x2{1, 2}, Int64x2{3, 4}, Int64x2{5, 6}, Int64x2{7, 8}},
      std::tuple{Int64x2{0, 0}, Int64x2{0, 0}, Int64x2{0, 0}, Int64x2{0, 0}});
  ASSERT_TRUE(tuple_is_returned_on_registers.has_value());
  EXPECT_FALSE(*tuple_is_returned_on_registers);
}

}  // namespace

}  // namespace berberis
