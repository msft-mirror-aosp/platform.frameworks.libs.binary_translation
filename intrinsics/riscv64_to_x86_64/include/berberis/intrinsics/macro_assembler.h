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

#include "berberis/intrinsics/intrinsics_float.h"

namespace berberis {

template <typename Assembler>
class MacroAssembler : public Assembler {
 public:
  template <typename... Args>
  explicit MacroAssembler(Args&&... args) : Assembler(std::forward<Args>(args)...) {
  }

  using Condition = typename Assembler::Condition;
  using Label = typename Assembler::Label;
  using Operand = typename Assembler::Operand;
  using Register = typename Assembler::Register;
  using XMMRegister = typename Assembler::XMMRegister;

  using Float32 = intrinsics::Float32;
  using Float64 = intrinsics::Float64;

#include "berberis/intrinsics/macro_assembler_interface-inl.h"  // NOLINT generated file

  using Assembler::Pand;
  using Assembler::Pandn;
  using Assembler::Pmov;
  using Assembler::Por;
  using Assembler::Pshufd;
  using Assembler::Vpand;
  using Assembler::Vpandn;
  using Assembler::Vpor;
  using Assembler::Vpshufd;

  using Assembler::gpr_a;
  using Assembler::gpr_c;
  using Assembler::gpr_d;

  template <typename format, typename... allowed_formats>
  static constexpr bool FormatIs = (std::is_same_v<format, allowed_formats> || ...);

#define DEFINE_MOV_INSTRUCTION(insn_name, parameters, arguments)                                   \
  template <typename format_out, typename format_in = format_out>                                  \
  void insn_name parameters {                                                                      \
    if constexpr (FormatIs<format_out, int8_t, uint8_t> && FormatIs<format_in, int8_t, uint8_t>) { \
      Assembler::Movb arguments;                                                                   \
    } else if constexpr (FormatIs<format_out, int16_t, uint16_t> && FormatIs<format_in, int8_t>) { \
      Assembler::Movsxbw arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int16_t, uint16_t> &&                                \
                         FormatIs<format_in, uint8_t>) {                                           \
      Assembler::Movzxbw arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int16_t, uint16_t> &&                                \
                         FormatIs<format_in, int16_t, uint16_t>) {                                 \
      Assembler::Movw arguments;                                                                   \
    } else if constexpr (FormatIs<format_out, int32_t, uint32_t> && FormatIs<format_in, int8_t>) { \
      Assembler::Movsxbl arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int32_t, uint32_t> &&                                \
                         FormatIs<format_in, uint8_t>) {                                           \
      Assembler::Movzxbl arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int32_t, uint32_t> &&                                \
                         FormatIs<format_in, int16_t>) {                                           \
      Assembler::Movsxwl arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int32_t, uint32_t> &&                                \
                         FormatIs<format_in, uint16_t>) {                                          \
      Assembler::Movzxwl arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int32_t, uint32_t> &&                                \
                         FormatIs<format_in, int32_t, uint32_t>) {                                 \
      Assembler::Movl arguments;                                                                   \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> && FormatIs<format_in, int8_t>) { \
      Assembler::Movsxbq arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> &&                                \
                         FormatIs<format_in, uint8_t>) {                                           \
      Assembler::Movzxbq arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> &&                                \
                         FormatIs<format_in, int16_t>) {                                           \
      Assembler::Movsxwq arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> &&                                \
                         FormatIs<format_in, uint16_t>) {                                          \
      Assembler::Movzxwq arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> &&                                \
                         FormatIs<format_in, int32_t>) {                                           \
      Assembler::Movsxwq arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> &&                                \
                         FormatIs<format_in, uint32_t>) {                                          \
      Assembler::Movzxwq arguments;                                                                \
    } else {                                                                                       \
      static_assert(                                                                               \
          FormatIs<format_out, int64_t, uint64_t> && FormatIs<format_in, int64_t, uint64_t>,       \
          "Only int{8,16,32,64}_t or uint{8,16,32,64}_t formats are supported");                   \
      Assembler::Movq arguments;                                                                   \
    }                                                                                              \
  }
  DEFINE_MOV_INSTRUCTION(Mov, (Register dest, Operand src), (dest, src))
  DEFINE_MOV_INSTRUCTION(Mov, (Register dest, Register src), (dest, src))
#undef DEFINE_MOV_INSTRUCTION

#define DEFINE_MOV_INSTRUCTION(insn_name, parameters, arguments)                                   \
  template <typename format_out, typename format_in = format_out>                                  \
  void insn_name parameters {                                                                      \
    if constexpr (FormatIs<format_out, int8_t, uint8_t> && FormatIs<format_in, int8_t, uint8_t>) { \
      Assembler::Movb arguments;                                                                   \
    } else if constexpr (FormatIs<format_out, int16_t, uint16_t> &&                                \
                         FormatIs<format_in, int16_t, uint16_t>) {                                 \
      Assembler::Movw arguments;                                                                   \
    } else if constexpr (FormatIs<format_out, int32_t, uint32_t> &&                                \
                         FormatIs<format_in, int32_t, uint32_t>) {                                 \
      Assembler::Movl arguments;                                                                   \
    } else {                                                                                       \
      static_assert(                                                                               \
          FormatIs<format_out, int64_t, uint64_t> && FormatIs<format_in, int64_t, uint64_t>,       \
          "Only int{8,16,32,64}_t or uint{8,16,32,64}_t formats are supported");                   \
      Assembler::Movq arguments;                                                                   \
    }                                                                                              \
  }
  DEFINE_MOV_INSTRUCTION(Mov, (Operand dest, Register src), (dest, src))
#undef DEFINE_MOV_INSTRUCTION

#define DEFINE_XMM_INT_INSTRUCTION(insn_name, parameters, arguments)                       \
  template <typename format>                                                               \
  void insn_name parameters {                                                              \
    if constexpr (FormatIs<format, int8_t, uint8_t>) {                                     \
      Assembler::insn_name##b arguments;                                                   \
    } else if constexpr (FormatIs<format, int16_t, uint16_t>) {                            \
      Assembler::insn_name##w arguments;                                                   \
    } else if constexpr (FormatIs<format, int32_t, uint32_t>) {                            \
      Assembler::insn_name##d arguments;                                                   \
    } else {                                                                               \
      static_assert(FormatIs<format, int64_t, uint64_t>,                                   \
                    "Only int{8,16,32,64}_t or uint{8,16,32,64}_t formats are supported"); \
      Assembler::insn_name##q arguments;                                                   \
    }                                                                                      \
  }
#define DEFINE_PCMP_INSTRUCTION(insn_name)                                                      \
  DEFINE_XMM_INT_INSTRUCTION(Pcmp##insn_name, (XMMRegister dest, Operand src), (dest, src))     \
  DEFINE_XMM_INT_INSTRUCTION(Pcmp##insn_name, (XMMRegister dest, XMMRegister src), (dest, src)) \
  DEFINE_XMM_INT_INSTRUCTION(                                                                   \
      Vpcmp##insn_name, (XMMRegister dest, XMMRegister src1, Operand src2), (dest, src1, src2)) \
  DEFINE_XMM_INT_INSTRUCTION(Vpcmp##insn_name,                                                  \
                             (XMMRegister dest, XMMRegister src1, XMMRegister src2),            \
                             (dest, src1, src2))
  DEFINE_PCMP_INSTRUCTION(eq)
  DEFINE_PCMP_INSTRUCTION(gt)
#undef DEFINE_PCMP_INSTRUCTION
#undef DEFINE_XMM_INT_INSTRUCTION

#define DEFINE_XMM_FLOAT_INSTRUCTION(insn_name, parameters, arguments)             \
  template <typename format>                                                       \
  void insn_name parameters {                                                      \
    if constexpr (FormatIs<format, float, Float32>) {                              \
      Assembler::insn_name##s arguments;                                           \
    } else {                                                                       \
      static_assert(FormatIs<format, double, Float64>,                             \
                    "Only float/Float32 or double/Float64 formats are supported"); \
      Assembler::insn_name##d arguments;                                           \
    }                                                                              \
  }
  DEFINE_XMM_FLOAT_INSTRUCTION(Movs, (XMMRegister dest, Operand src), (dest, src))
  DEFINE_XMM_FLOAT_INSTRUCTION(Movs, (XMMRegister dest, XMMRegister src), (dest, src))
  DEFINE_XMM_FLOAT_INSTRUCTION(Vmovs, (XMMRegister dest, Operand src), (dest, src))
  DEFINE_XMM_FLOAT_INSTRUCTION(Vmovs,
                               (XMMRegister dest, XMMRegister src1, XMMRegister src2),
                               (dest, src1, src2))
#define CMP_INSTRUCTION(insn_name)                                                                \
  DEFINE_XMM_FLOAT_INSTRUCTION(Cmp##insn_name##p, (XMMRegister dest, Operand src), (dest, src))   \
  DEFINE_XMM_FLOAT_INSTRUCTION(Cmp##insn_name##s, (XMMRegister dest, Operand src), (dest, src))   \
  DEFINE_XMM_FLOAT_INSTRUCTION(                                                                   \
      Cmp##insn_name##p, (XMMRegister dest, XMMRegister src), (dest, src))                        \
  DEFINE_XMM_FLOAT_INSTRUCTION(                                                                   \
      Cmp##insn_name##s, (XMMRegister dest, XMMRegister src), (dest, src))                        \
  DEFINE_XMM_FLOAT_INSTRUCTION(                                                                   \
      Vcmp##insn_name##p, (XMMRegister dest, XMMRegister src1, Operand src2), (dest, src1, src2)) \
  DEFINE_XMM_FLOAT_INSTRUCTION(                                                                   \
      Vcmp##insn_name##s, (XMMRegister dest, XMMRegister src1, Operand src2), (dest, src1, src2)) \
  DEFINE_XMM_FLOAT_INSTRUCTION(Vcmp##insn_name##p,                                                \
                               (XMMRegister dest, XMMRegister src1, XMMRegister src2),            \
                               (dest, src1, src2))                                                \
  DEFINE_XMM_FLOAT_INSTRUCTION(Vcmp##insn_name##s,                                                \
                               (XMMRegister dest, XMMRegister src1, XMMRegister src2),            \
                               (dest, src1, src2))
  CMP_INSTRUCTION(eq)
  CMP_INSTRUCTION(le)
  CMP_INSTRUCTION(lt)
  CMP_INSTRUCTION(ord)
  CMP_INSTRUCTION(neq)
  CMP_INSTRUCTION(nle)
  CMP_INSTRUCTION(nlt)
  CMP_INSTRUCTION(unord)
#undef CMP_INSTRUCTION
#undef DEFINE_XMM_FLOAT_INSTRUCTION

 private:

  // Useful constants for PshufXXX instructions.
  enum {
    kShuffleDDBB = 0b11110101
  };

};

}  // namespace berberis

// Macro specializations.
#include "berberis/intrinsics/macro_assembler-impl.h"

#endif  // RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_H_
