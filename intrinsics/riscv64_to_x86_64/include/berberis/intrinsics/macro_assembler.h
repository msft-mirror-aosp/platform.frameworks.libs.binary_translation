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

#ifndef RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_H_
#define RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_H_

#include <limits.h>
#include <type_traits>  // is_same_v

#include <functional>
#include <tuple>
#include <utility>

// Don't include arch-dependent parts because macro-assembler doesn't depend on implementation of
// Float32/Float64 types but can be compiled for different architecture (soong's host architecture,
// not device architecture AKA berberis' host architecture).
#include "berberis/intrinsics/common/intrinsics_float.h"
#include "berberis/intrinsics/constants_pool.h"

namespace berberis {

template <typename Assembler>
class MacroAssembler : public Assembler {
 public:
  using MacroAssemblers = std::tuple<MacroAssembler<Assembler>,
                                     typename Assembler::BaseAssembler,
                                     typename Assembler::FinalAssembler>;

  template <typename... Args>
  explicit MacroAssembler(Args&&... args) : Assembler(std::forward<Args>(args)...) {
  }

#define IMPORT_ASSEMBLER_FUNCTIONS
#include "berberis/assembler/gen_assembler_x86_64-using-inl.h"
#undef IMPORT_ASSEMBLER_FUNCTIONS

#define DEFINE_MACRO_ASSEMBLER_GENERIC_FUNCTIONS
#include "berberis/intrinsics/all_to_x86_32_or_x86_64/macro_assembler-inl.h"
#undef DEFINE_MACRO_ASSEMBLER_GENERIC_FUNCTIONS

  void PNot(XMMRegister result) {
    Pandn(result, {.disp = constants_pool::kVectorConst<uint8_t{0b1111'1111}>});
  }

  void Vpnot(XMMRegister result, XMMRegister src) {
    Vpandn(result, src, {.disp = constants_pool::kVectorConst<uint8_t{0b1111'1111}>});
  }

#include "berberis/intrinsics/macro_assembler_interface-inl.h"  // NOLINT generated file

 private:

  // Useful constants for PshufXXX instructions.
  enum {
    kShuffleDDBB = 0b11110101
  };
};

}  // namespace berberis

// Macro specializations.
#include "berberis/intrinsics/macro_assembler_arith_impl.h"
#include "berberis/intrinsics/macro_assembler_bitmanip_impl.h"
#include "berberis/intrinsics/macro_assembler_floating_point_impl.h"

#endif  // RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_H_
