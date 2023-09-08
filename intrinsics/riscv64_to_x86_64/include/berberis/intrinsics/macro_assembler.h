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
  template <typename IntType>
  using ImmFormat =
      std::conditional_t<sizeof(IntType) <= sizeof(int32_t), std::make_signed_t<IntType>, int32_t>;

 public:
  template <typename... Args>
  explicit MacroAssembler(Args&&... args) : Assembler(std::forward<Args>(args)...) {
  }

  using Condition = typename Assembler::Condition;
  using Label = typename Assembler::Label;
  using Operand = typename Assembler::Operand;
  using Register = typename Assembler::Register;
  using ScaleFactor = typename Assembler::ScaleFactor;
  using XMMRegister = typename Assembler::XMMRegister;

  using Float32 = intrinsics::Float32;
  using Float64 = intrinsics::Float64;

#include "berberis/intrinsics/macro_assembler_interface-inl.h"  // NOLINT generated file

  using Assembler::Bind;
  using Assembler::Btq;
  using Assembler::Fldcw;
  using Assembler::Fnstcw;
  using Assembler::Jcc;
  using Assembler::Ldmxcsr;
  using Assembler::Leal;
  using Assembler::Leaq;
  using Assembler::MakeLabel;
  using Assembler::Movl;
  using Assembler::Pand;
  using Assembler::Pandn;
  using Assembler::Pcmpeqb;
  using Assembler::Pmov;
  using Assembler::Por;
  using Assembler::Pshufd;
  using Assembler::Setcc;
  using Assembler::Stmxcsr;
  using Assembler::Vpand;
  using Assembler::Vpandn;
  using Assembler::Vpcmpeqb;
  using Assembler::Vpor;
  using Assembler::Vpshufd;

  using Assembler::gpr_a;
  using Assembler::gpr_c;
  using Assembler::gpr_d;

  template <typename format, typename... allowed_formats>
  static constexpr bool FormatIs = (std::is_same_v<format, allowed_formats> || ...);

#define DEFINE_EXPAND_INSTRUCTION(opt_check, parameters, arguments)                                \
  template <typename format_out, typename format_in>                                               \
  void Expand parameters {                                                                         \
    if constexpr (FormatIs<format_out, int8_t, uint8_t> && FormatIs<format_in, int8_t, uint8_t>) { \
      opt_check;                                                                                   \
      Assembler::Movb arguments;                                                                   \
    } else if constexpr (FormatIs<format_out, int16_t, uint16_t> && FormatIs<format_in, int8_t>) { \
      Assembler::Movsxbw arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int16_t, uint16_t> &&                                \
                         FormatIs<format_in, uint8_t>) {                                           \
      Assembler::Movzxbw arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int16_t, uint16_t> &&                                \
                         FormatIs<format_in, int16_t, uint16_t>) {                                 \
      opt_check;                                                                                   \
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
      opt_check;                                                                                   \
      Assembler::Movl arguments;                                                                   \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> && FormatIs<format_in, int8_t>) { \
      Assembler::Movsxbq arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> &&                                \
                         FormatIs<format_in, uint8_t>) {                                           \
      Assembler::Movzxbl arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> &&                                \
                         FormatIs<format_in, int16_t>) {                                           \
      Assembler::Movsxwq arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> &&                                \
                         FormatIs<format_in, uint16_t>) {                                          \
      Assembler::Movzxwl arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> &&                                \
                         FormatIs<format_in, int32_t>) {                                           \
      Assembler::Movsxlq arguments;                                                                \
    } else if constexpr (FormatIs<format_out, int64_t, uint64_t> &&                                \
                         FormatIs<format_in, uint32_t>) {                                          \
      Assembler::Movl arguments;                                                                   \
    } else {                                                                                       \
      static_assert(                                                                               \
          FormatIs<format_out, int64_t, uint64_t> && FormatIs<format_in, int64_t, uint64_t>,       \
          "Only int{8,16,32,64}_t or uint{8,16,32,64}_t formats are supported");                   \
      opt_check;                                                                                   \
      Assembler::Movq arguments;                                                                   \
    }                                                                                              \
  }
  DEFINE_EXPAND_INSTRUCTION(, (Register dest, Operand src), (dest, src))
  DEFINE_EXPAND_INSTRUCTION(if (dest == src) return, (Register dest, Register src), (dest, src))
#undef DEFINE_EXPAND_INSTRUCTION

#define DEFINE_INT_INSTRUCTION(insn_name, insn_siffix, parameters, arguments)              \
  template <typename format>                                                               \
  std::enable_if_t<std::is_integral_v<format>, void> insn_name##insn_siffix parameters {   \
    if constexpr (FormatIs<format, int8_t, uint8_t>) {                                     \
      Assembler::insn_name##b##insn_siffix arguments;                                      \
    } else if constexpr (FormatIs<format, int16_t, uint16_t>) {                            \
      Assembler::insn_name##w##insn_siffix arguments;                                      \
    } else if constexpr (FormatIs<format, int32_t, uint32_t>) {                            \
      Assembler::insn_name##l##insn_siffix arguments;                                      \
    } else {                                                                               \
      static_assert(FormatIs<format, int64_t, uint64_t>,                                   \
                    "Only int{8,16,32,64}_t or uint{8,16,32,64}_t formats are supported"); \
      Assembler::insn_name##q##insn_siffix arguments;                                      \
    }                                                                                      \
  }
  DEFINE_INT_INSTRUCTION(CmpXchg, , (Operand dest, Register src), (dest, src))
  DEFINE_INT_INSTRUCTION(CmpXchg, , (Register dest, Register src), (dest, src))
  DEFINE_INT_INSTRUCTION(LockCmpXchg, , (Operand dest, Register src), (dest, src))
  DEFINE_INT_INSTRUCTION(Mov, , (Operand dest, ImmFormat<format> imm), (dest, imm))
  DEFINE_INT_INSTRUCTION(Mov, , (Operand dest, Register src), (dest, src))
  DEFINE_INT_INSTRUCTION(Mov, , (Register dest, std::make_signed_t<format> imm), (dest, imm))
  DEFINE_INT_INSTRUCTION(Mov, , (Register dest, Operand src), (dest, src))
#define DEFINE_ARITH_INSTRUCTION(insn_name)                                                \
  DEFINE_INT_INSTRUCTION(insn_name, , (Operand dest, ImmFormat<format> imm), (dest, imm))  \
  DEFINE_INT_INSTRUCTION(insn_name, , (Operand dest, Register src), (dest, src))           \
  DEFINE_INT_INSTRUCTION(insn_name, , (Register dest, ImmFormat<format> imm), (dest, imm)) \
  DEFINE_INT_INSTRUCTION(insn_name, , (Register dest, Operand src), (dest, src))           \
  DEFINE_INT_INSTRUCTION(insn_name, , (Register dest, Register src), (dest, src))
  DEFINE_ARITH_INSTRUCTION(Adc)
  DEFINE_ARITH_INSTRUCTION(Add)
  DEFINE_ARITH_INSTRUCTION(And)
  DEFINE_ARITH_INSTRUCTION(Cmp)
  DEFINE_ARITH_INSTRUCTION(Or)
  DEFINE_ARITH_INSTRUCTION(Sbb)
  DEFINE_ARITH_INSTRUCTION(Sub)
  DEFINE_ARITH_INSTRUCTION(Xor)
#define DEFINE_SHIFT_INSTRUCTION(insn_name)                                     \
  DEFINE_INT_INSTRUCTION(insn_name, , (Operand dest, int8_t imm), (dest, imm))  \
  DEFINE_INT_INSTRUCTION(insn_name, ByCl, (Operand dest), (dest))               \
  DEFINE_INT_INSTRUCTION(insn_name, , (Register dest, int8_t imm), (dest, imm)) \
  DEFINE_INT_INSTRUCTION(insn_name, ByCl, (Register dest), (dest))
  DEFINE_SHIFT_INSTRUCTION(Rcl)
  DEFINE_SHIFT_INSTRUCTION(Rcr)
  DEFINE_SHIFT_INSTRUCTION(Rol)
  DEFINE_SHIFT_INSTRUCTION(Ror)
  DEFINE_SHIFT_INSTRUCTION(Sar)
  DEFINE_SHIFT_INSTRUCTION(Shl)
  DEFINE_SHIFT_INSTRUCTION(Shr)
#undef DEFINE_INT_INSTRUCTION
#undef DEFINE_SHIFT_INSTRUCTION

#define DEFINE_INT_INSTRUCTION(insn_name, parameters, arguments)                       \
  template <typename format>                                                           \
  std::enable_if_t<std::is_integral_v<format>, void> insn_name parameters {            \
    if constexpr (FormatIs<format, int16_t, uint16_t>) {                               \
      Assembler::insn_name##w arguments;                                               \
    } else if constexpr (FormatIs<format, int32_t, uint32_t>) {                        \
      Assembler::insn_name##l arguments;                                               \
    } else {                                                                           \
      static_assert(FormatIs<format, int64_t, uint64_t>,                               \
                    "Only int{16,32,64}_t or uint{16,32,64}_t formats are supported"); \
      Assembler::insn_name##q arguments;                                               \
    }                                                                                  \
  }
  DEFINE_INT_INSTRUCTION(Cmov, (Condition cond, Register dest, Operand src), (cond, dest, src))
  DEFINE_INT_INSTRUCTION(Cmov, (Condition cond, Register dest, Register src), (cond, dest, src))
#define DEFINE_BIT_INSTRUCTION(insn_name)                                                \
  DEFINE_INT_INSTRUCTION(insn_name, (Operand dest, ImmFormat<format> imm), (dest, imm))  \
  DEFINE_INT_INSTRUCTION(insn_name, (Operand dest, Register src), (dest, src))           \
  DEFINE_INT_INSTRUCTION(insn_name, (Register dest, ImmFormat<format> imm), (dest, imm)) \
  DEFINE_INT_INSTRUCTION(insn_name, (Register dest, Register src), (dest, src))
  DEFINE_BIT_INSTRUCTION(Bt)
  DEFINE_BIT_INSTRUCTION(Btc)
  DEFINE_BIT_INSTRUCTION(Btr)
  DEFINE_BIT_INSTRUCTION(Bts)
#undef DEFINE_BIT_INSTRUCTION
#define DEFINE_BIT_INSTRUCTION(insn_name)                                                \
  DEFINE_INT_INSTRUCTION(insn_name, (Register dest, Operand src), (dest, src))           \
  DEFINE_INT_INSTRUCTION(insn_name, (Register dest, Register src), (dest, src))
  DEFINE_BIT_INSTRUCTION(Bsf)
  DEFINE_BIT_INSTRUCTION(Bsr)
  DEFINE_BIT_INSTRUCTION(Lzcnt)
  DEFINE_BIT_INSTRUCTION(Tzcnt)
#undef DEFINE_INT_INSTRUCTION

  // Note: Mov<int32_t> from one register to that same register doesn't zero-out top 32bits,
  // like real Movq would! If you want that effect then use Expand<tnt32_t, int32_t> instead!
  template <typename format>
  std::enable_if_t<std::is_integral_v<format>, void> Mov(Register dest, Register src) {
    if (dest == src) {
      return;
    }
    if constexpr (FormatIs<format, int8_t, uint8_t>) {
      Assembler::Movb(dest, src);
    } else if constexpr (FormatIs<format, int16_t, uint16_t>) {
      Assembler::Movw(dest, src);
    } else if constexpr (FormatIs<format, int32_t, uint32_t>) {
      Assembler::Movl(dest, src);
    } else {
      static_assert(FormatIs<format, int64_t, uint64_t>,
                    "Only int{8,16,32,64}_t or uint{8,16,32,64}_t formats are supported");
      Assembler::Movq(dest, src);
    }
  }

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

#define DEFINE_MOVS_INSTRUCTION(insn_name, opt_check, parameters, arguments)       \
  template <typename format>                                                       \
  void insn_name parameters {                                                      \
    if constexpr (FormatIs<format, float, Float32>) {                              \
      opt_check;                                                                   \
      Assembler::insn_name##s arguments;                                           \
    } else {                                                                       \
      static_assert(FormatIs<format, double, Float64>,                             \
                    "Only float/Float32 or double/Float64 formats are supported"); \
      opt_check;                                                                   \
      Assembler::insn_name##d arguments;                                           \
    }                                                                              \
  }
  DEFINE_MOVS_INSTRUCTION(Movs, , (XMMRegister dest, Operand src), (dest, src))
  DEFINE_MOVS_INSTRUCTION(Movs, , (Operand dest, XMMRegister src), (dest, src))
  DEFINE_MOVS_INSTRUCTION(Movs,
                          if (dest == src) return,
                          (XMMRegister dest, XMMRegister src),
                          (dest, src))
  DEFINE_MOVS_INSTRUCTION(Vmovs, , (XMMRegister dest, Operand src), (dest, src))
  DEFINE_MOVS_INSTRUCTION(Vmovs, , (Operand dest, XMMRegister src), (dest, src))
  DEFINE_MOVS_INSTRUCTION(Vmovs,
                          if ((dest == src1) && (dest == src2)) return,
                          (XMMRegister dest, XMMRegister src1, XMMRegister src2),
                          (dest, src1, src2))
#undef DEFINE_MOVS_INSTRUCTION

#define DEFINE_XMM_MOV_INSTRUCTION(insn_name, parameters, arguments)               \
  template <typename format>                                                       \
  void insn_name parameters {                                                      \
    if constexpr (FormatIs<format, float, Float32>) {                              \
      Assembler::insn_name##d arguments;                                           \
    } else {                                                                       \
      static_assert(FormatIs<format, double, Float64>,                             \
                    "Only float/Float32 or double/Float64 formats are supported"); \
      Assembler::insn_name##q arguments;                                           \
    }                                                                              \
  }
  DEFINE_XMM_MOV_INSTRUCTION(Mov, (XMMRegister dest, Operand src), (dest, src))
  DEFINE_XMM_MOV_INSTRUCTION(Mov, (Operand dest, XMMRegister src), (dest, src))
  DEFINE_XMM_MOV_INSTRUCTION(Mov, (XMMRegister dest, Register src), (dest, src))
  DEFINE_XMM_MOV_INSTRUCTION(Mov, (Register dest, XMMRegister src), (dest, src))
  DEFINE_XMM_MOV_INSTRUCTION(Vmov, (XMMRegister dest, Operand src), (dest, src))
  DEFINE_XMM_MOV_INSTRUCTION(Vmov, (Operand dest, XMMRegister src), (dest, src))
  DEFINE_XMM_MOV_INSTRUCTION(Vmov, (XMMRegister dest, Register src), (dest, src))
  DEFINE_XMM_MOV_INSTRUCTION(Vmov, (Register dest, XMMRegister src), (dest, src))
#undef DEFINE_XMM_MOV_INSTRUCTION

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
  DEFINE_XMM_FLOAT_INSTRUCTION(Comis, (XMMRegister dest, Operand src), (dest, src))
  DEFINE_XMM_FLOAT_INSTRUCTION(Comis, (XMMRegister dest, XMMRegister src), (dest, src))
  DEFINE_XMM_FLOAT_INSTRUCTION(Ucomis, (XMMRegister dest, Operand src), (dest, src))
  DEFINE_XMM_FLOAT_INSTRUCTION(Ucomis, (XMMRegister dest, XMMRegister src), (dest, src))
  DEFINE_XMM_FLOAT_INSTRUCTION(Vcomis, (XMMRegister dest, Operand src), (dest, src))
  DEFINE_XMM_FLOAT_INSTRUCTION(Vcomis, (XMMRegister dest, XMMRegister src), (dest, src))
  DEFINE_XMM_FLOAT_INSTRUCTION(Vucomis, (XMMRegister dest, Operand src), (dest, src))
  DEFINE_XMM_FLOAT_INSTRUCTION(Vucomis, (XMMRegister dest, XMMRegister src), (dest, src))
#define DEFINE_CMP_INSTRUCTION(insn_name)                                                         \
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
  DEFINE_CMP_INSTRUCTION(eq)
  DEFINE_CMP_INSTRUCTION(le)
  DEFINE_CMP_INSTRUCTION(lt)
  DEFINE_CMP_INSTRUCTION(ord)
  DEFINE_CMP_INSTRUCTION(neq)
  DEFINE_CMP_INSTRUCTION(nle)
  DEFINE_CMP_INSTRUCTION(nlt)
  DEFINE_CMP_INSTRUCTION(unord)
#undef DEFINE_CMP_INSTRUCTION
#undef DEFINE_XMM_FLOAT_INSTRUCTION

 private:

  // Useful constants for PshufXXX instructions.
  enum {
    kShuffleDDBB = 0b11110101
  };

};

}  // namespace berberis

// Macro specializations.
#include "berberis/intrinsics/macro_assembler_bitmanip_impl.h"
#include "berberis/intrinsics/macro_assembler_floating_point_impl.h"

#endif  // RISCV64_TO_X86_64_BERBERIS_INTRINSICS_MACRO_ASSEMBLER_H_
