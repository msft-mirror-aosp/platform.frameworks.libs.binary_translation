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

#ifndef DEFINE_MACRO_ASSEMBLER_GENERIC_FUNCTIONS
#error This file is supposed to be included from berberis/intrinsics/macro_assembler-inl.h
#else
#undef DEFINE_MACRO_ASSEMBLER_GENERIC_FUNCTIONS
#endif

using Condition = typename Assembler::Condition;
using Label = typename Assembler::Label;
using Operand = typename Assembler::Operand;
using Register = typename Assembler::Register;
using ScaleFactor = typename Assembler::ScaleFactor;
using XMMRegister = typename Assembler::XMMRegister;

using Float32 = intrinsics::Float32;
using Float64 = intrinsics::Float64;

template <typename IntType>
using ImmFormat =
    std::conditional_t<sizeof(IntType) <= sizeof(int32_t), std::make_signed_t<IntType>, int32_t>;

template <typename format, typename... allowed_formats>
static constexpr bool kFormatIs = (std::is_same_v<format, allowed_formats> || ...);

template <typename IntType>
static constexpr bool kIntType =
    kFormatIs<IntType, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;

template <typename IntType>
static constexpr bool kIntTypeWLQ =
    kFormatIs<IntType, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;

template <typename FloatType>
static constexpr bool kFloatType = kFormatIs<FloatType, Float32, Float64>;

#define DEFINE_EXPAND_INSTRUCTION(Declare_dest, Declare_src)         \
  template <typename format_out, typename format_in>                 \
  std::enable_if_t<kIntType<format_out> && kIntType<format_in> &&    \
                   sizeof(format_in) <= sizeof(format_out)>          \
  Expand(Declare_dest, Declare_src) {                                \
    if constexpr (std::is_same_v<decltype(dest), decltype(src)> &&   \
                  sizeof(format_out) == sizeof(format_in)) {         \
      if (dest == src) {                                             \
        return;                                                      \
      }                                                              \
    }                                                                \
    if constexpr (kFormatIs<format_out, int8_t, uint8_t> &&          \
                  kFormatIs<format_in, int8_t, uint8_t>) {           \
      Assembler::Movb(dest, src);                                    \
    } else if constexpr (kFormatIs<format_out, int16_t, uint16_t> && \
                         kFormatIs<format_in, int8_t>) {             \
      Assembler::Movsxbw(dest, src);                                 \
    } else if constexpr (kFormatIs<format_out, int16_t, uint16_t> && \
                         kFormatIs<format_in, uint8_t>) {            \
      Assembler::Movzxbw(dest, src);                                 \
    } else if constexpr (kFormatIs<format_out, int16_t, uint16_t> && \
                         kFormatIs<format_in, int16_t, uint16_t>) {  \
      Assembler::Movw(dest, src);                                    \
    } else if constexpr (kFormatIs<format_out, int32_t, uint32_t> && \
                         kFormatIs<format_in, int8_t>) {             \
      Assembler::Movsxbl(dest, src);                                 \
    } else if constexpr (kFormatIs<format_out, int32_t, uint32_t> && \
                         kFormatIs<format_in, uint8_t>) {            \
      Assembler::Movzxbl(dest, src);                                 \
    } else if constexpr (kFormatIs<format_out, int32_t, uint32_t> && \
                         kFormatIs<format_in, int16_t>) {            \
      Assembler::Movsxwl(dest, src);                                 \
    } else if constexpr (kFormatIs<format_out, int32_t, uint32_t> && \
                         kFormatIs<format_in, uint16_t>) {           \
      Assembler::Movzxwl(dest, src);                                 \
    } else if constexpr (kFormatIs<format_out, int32_t, uint32_t> && \
                         kFormatIs<format_in, int32_t, uint32_t>) {  \
      Assembler::Movl(dest, src);                                    \
    } else if constexpr (kFormatIs<format_out, int64_t, uint64_t> && \
                         kFormatIs<format_in, int8_t>) {             \
      Assembler::Movsxbq(dest, src);                                 \
    } else if constexpr (kFormatIs<format_out, int64_t, uint64_t> && \
                         kFormatIs<format_in, uint8_t>) {            \
      Assembler::Movzxbl(dest, src);                                 \
    } else if constexpr (kFormatIs<format_out, int64_t, uint64_t> && \
                         kFormatIs<format_in, int16_t>) {            \
      Assembler::Movsxwq(dest, src);                                 \
    } else if constexpr (kFormatIs<format_out, int64_t, uint64_t> && \
                         kFormatIs<format_in, uint16_t>) {           \
      Assembler::Movzxwl(dest, src);                                 \
    } else if constexpr (kFormatIs<format_out, int64_t, uint64_t> && \
                         kFormatIs<format_in, int32_t>) {            \
      Assembler::Movsxlq(dest, src);                                 \
    } else if constexpr (kFormatIs<format_out, int64_t, uint64_t> && \
                         kFormatIs<format_in, uint32_t>) {           \
      Assembler::Movl(dest, src);                                    \
    } else {                                                         \
      Assembler::Movq(dest, src);                                    \
    }                                                                \
  }
DEFINE_EXPAND_INSTRUCTION(Register dest, Operand src)
DEFINE_EXPAND_INSTRUCTION(Register dest, Register src)
#undef DEFINE_EXPAND_INSTRUCTION

#define DEFINE_INT_INSTRUCTION(insn_name, insn_siffix, parameters, arguments) \
  template <typename format>                                                  \
  std::enable_if_t<kIntType<format>> insn_name##insn_siffix parameters {      \
    if constexpr (kFormatIs<format, int8_t, uint8_t>) {                       \
      Assembler::insn_name##b##insn_siffix arguments;                         \
    } else if constexpr (kFormatIs<format, int16_t, uint16_t>) {              \
      Assembler::insn_name##w##insn_siffix arguments;                         \
    } else if constexpr (kFormatIs<format, int32_t, uint32_t>) {              \
      Assembler::insn_name##l##insn_siffix arguments;                         \
    } else {                                                                  \
      Assembler::insn_name##q##insn_siffix arguments;                         \
    }                                                                         \
  }
DEFINE_INT_INSTRUCTION(CmpXchg, , (Operand dest, Register src), (dest, src))
DEFINE_INT_INSTRUCTION(CmpXchg, , (Register dest, Register src), (dest, src))
DEFINE_INT_INSTRUCTION(LockCmpXchg, , (Operand dest, Register src), (dest, src))
DEFINE_INT_INSTRUCTION(Mov, , (Operand dest, ImmFormat<format> imm), (dest, imm))
DEFINE_INT_INSTRUCTION(Mov, , (Operand dest, Register src), (dest, src))
DEFINE_INT_INSTRUCTION(Mov, , (Register dest, std::make_signed_t<format> imm), (dest, imm))
DEFINE_INT_INSTRUCTION(Mov, , (Register dest, Operand src), (dest, src))
DEFINE_INT_INSTRUCTION(Test, , (Operand dest, ImmFormat<format> imm), (dest, imm))
DEFINE_INT_INSTRUCTION(Test, , (Operand dest, Register src), (dest, src))
DEFINE_INT_INSTRUCTION(Test, , (Register dest, ImmFormat<format> imm), (dest, imm))
DEFINE_INT_INSTRUCTION(Test, , (Register dest, Register src), (dest, src))
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
  std::enable_if_t<kIntTypeWLQ<format>> insn_name parameters {                         \
    if constexpr (kFormatIs<format, int16_t, uint16_t>) {                              \
      Assembler::insn_name##w arguments;                                               \
    } else if constexpr (kFormatIs<format, int32_t, uint32_t>) {                       \
      Assembler::insn_name##l arguments;                                               \
    } else {                                                                           \
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
#define DEFINE_BIT_INSTRUCTION(insn_name)                                      \
  DEFINE_INT_INSTRUCTION(insn_name, (Register dest, Operand src), (dest, src)) \
  DEFINE_INT_INSTRUCTION(insn_name, (Register dest, Register src), (dest, src))
DEFINE_BIT_INSTRUCTION(Bsf)
DEFINE_BIT_INSTRUCTION(Bsr)
DEFINE_BIT_INSTRUCTION(Lzcnt)
DEFINE_BIT_INSTRUCTION(Tzcnt)
#undef DEFINE_INT_INSTRUCTION

// Note: Mov<int32_t> from one register to that same register doesn't zero-out top 32bits,
// like real Movq would! If you want that effect then use Expand<tnt32_t, int32_t> instead!
template <typename format>
std::enable_if_t<kIntType<format>> Mov(Register dest, Register src) {
  if (dest == src) {
    return;
  }
  if constexpr (kFormatIs<format, int8_t, uint8_t>) {
    Assembler::Movb(dest, src);
  } else if constexpr (kFormatIs<format, int16_t, uint16_t>) {
    Assembler::Movw(dest, src);
  } else if constexpr (kFormatIs<format, int32_t, uint32_t>) {
    Assembler::Movl(dest, src);
  } else {
    Assembler::Movq(dest, src);
  }
}

#define DEFINE_XMM_INT_INSTRUCTION(insn_name, parameters, arguments) \
  template <typename format>                                         \
  std::enable_if_t<kIntType<format>> insn_name parameters {          \
    if constexpr (kFormatIs<format, int8_t, uint8_t>) {              \
      Assembler::insn_name##b arguments;                             \
    } else if constexpr (kFormatIs<format, int16_t, uint16_t>) {     \
      Assembler::insn_name##w arguments;                             \
    } else if constexpr (kFormatIs<format, int32_t, uint32_t>) {     \
      Assembler::insn_name##d arguments;                             \
    } else {                                                         \
      Assembler::insn_name##q arguments;                             \
    }                                                                \
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

#define DEFINE_MOVS_INSTRUCTION(insn_name, opt_check, parameters, arguments) \
  template <typename format>                                                 \
  std::enable_if_t<kFloatType<format>> insn_name parameters {                \
    if constexpr (kFormatIs<format, Float32>) {                              \
      opt_check;                                                             \
      Assembler::insn_name##s arguments;                                     \
    } else {                                                                 \
      opt_check;                                                             \
      Assembler::insn_name##d arguments;                                     \
    }                                                                        \
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

#define DEFINE_XMM_MOV_INSTRUCTION(insn_name, parameters, arguments) \
  template <typename format>                                         \
  std::enable_if_t<kFloatType<format>> insn_name parameters {        \
    if constexpr (kFormatIs<format, Float32>) {                      \
      Assembler::insn_name##d arguments;                             \
    } else {                                                         \
      Assembler::insn_name##q arguments;                             \
    }                                                                \
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

#define DEFINE_XMM_FLOAT_INSTRUCTION(insn_name, parameters, arguments) \
  template <typename format>                                           \
  std::enable_if_t<kFloatType<format>> insn_name parameters {          \
    if constexpr (kFormatIs<format, Float32>) {                        \
      Assembler::insn_name##s arguments;                               \
    } else {                                                           \
      Assembler::insn_name##d arguments;                               \
    }                                                                  \
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
