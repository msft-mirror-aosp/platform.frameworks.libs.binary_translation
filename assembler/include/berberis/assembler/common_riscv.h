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

#ifndef BERBERIS_ASSEMBLER_COMMON_RISCV_H_
#define BERBERIS_ASSEMBLER_COMMON_RISCV_H_

#include <cstddef>  // std::size_t
#include <cstdint>
#include <type_traits>  // std::enable_if, std::is_integral

#include "berberis/assembler/common.h"
#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"

namespace berberis {

// AssemblerRiscV includes implementation of most Risc V assembler instructions.
//
// RV32 and RV64 assemblers are nearly identical, but difference lies in handling
// of some instructions: RV32 uses certain encodings differently to handle compressed
// instructions, while RV64 adds some extra instructions to handle 32bit quantities
// (*not* 64bit quantities as the name implies, instead there are width-native instructions
// and extra 32bit ones for RV64).
//
// To handle that difference efficiently AssemblerRiscV is CRTP class: it's parameterized
// by its own descendant and pull certain functions from its implementation.

namespace rv32e {

class Assembler;

}  // namespace rv32e

namespace rv32i {

class Assembler;

}  // namespace rv32i

namespace rv64i {

class Assembler;

}  // namespace rv64i

template <typename Assembler>
class AssemblerRiscV : public AssemblerBase {
 public:
  explicit AssemblerRiscV(MachineCode* code) : AssemblerBase(code) {}

  class Register {
    constexpr bool operator==(const Register& reg) const { return num_ == reg.num_; }
    constexpr bool operator!=(const Register& reg) const { return num_ != reg.num_; }
    constexpr uint8_t GetPhysicalIndex() { return num_; }
    friend constexpr uint8_t ValueForFmtSpec(Register value) { return value.num_; }
    friend class AssemblerRiscV<Assembler>;
    friend class rv32e::Assembler;
    friend class rv32i::Assembler;
    friend class rv64i::Assembler;

   private:
    constexpr Register(uint8_t num) : num_(num) {}
    uint8_t num_;
  };

  // Note: register x0, technically, can be specified in assembler even if it doesn't exist
  // as separate hardware register. It even have alias “zero” even in clang assembler.
  static constexpr Register x0{0};
  static constexpr Register x1{1};
  static constexpr Register x2{2};
  static constexpr Register x3{3};
  static constexpr Register x4{4};
  static constexpr Register x5{5};
  static constexpr Register x6{6};
  static constexpr Register x7{7};
  static constexpr Register x8{8};
  static constexpr Register x9{9};
  static constexpr Register x10{10};
  static constexpr Register x11{11};
  static constexpr Register x12{12};
  static constexpr Register x13{13};
  static constexpr Register x14{14};
  static constexpr Register x15{15};
  static constexpr Register x16{16};
  static constexpr Register x17{17};
  static constexpr Register x18{18};
  static constexpr Register x19{19};
  static constexpr Register x20{20};
  static constexpr Register x21{21};
  static constexpr Register x22{22};
  static constexpr Register x23{23};
  static constexpr Register x24{24};
  static constexpr Register x25{25};
  static constexpr Register x26{26};
  static constexpr Register x27{27};
  static constexpr Register x28{28};
  static constexpr Register x29{29};
  static constexpr Register x30{30};
  static constexpr Register x31{31};

  // Aliases
  static constexpr Register no_register{0x80};
  static constexpr Register zero{0};

  template <typename RegisterType, typename ImmediateType>
  struct Operand {
    RegisterType base;
    ImmediateType disp;
  };

  // Immediates are kept in a form ready to be used with emitter.
  class BImmediate;
  class CsrImmediate;
  class IImmediate;
  using Immediate = IImmediate;
  class JImmediate;
  // In RISC V manual shifts are described as using I-format with complex restrictions for which
  // immediates are accepted and allowed (with parts of what manual classifies as “immediate” used
  // to determine the actual instruction used and rules which differ between RV32 and RV64!).
  //
  // Instead of doing special handling for the instructions in python scripts we just reclassify
  // these parts of immediate as “opcode” and reclassify these instructions as “Shift32-type” and
  // “Shift64-type”.
  //
  // This also means that the same instructions for RV32 and RV64 would have different types, but
  // since we don't have a goal to make RV32 a strict subset of RV64 that's acceptable.
  //
  // In addition we provide aliases in RV32 and RV64 assemblers to make sure users of assembler may
  // still use ShiftImmediate and MakeShiftImmediate for native width without thinking about
  // details of implementation.
  class Shift32Immediate;
  class Shift64Immediate;
  class PImmediate;
  class SImmediate;
  class UImmediate;

  // Don't use templates here to enable implicit conversions.
#define BERBERIS_DEFINE_MAKE_IMMEDIATE(Immediate, MakeImmediate)           \
  static constexpr std::optional<Immediate> MakeImmediate(int8_t value);   \
  static constexpr std::optional<Immediate> MakeImmediate(uint8_t value);  \
  static constexpr std::optional<Immediate> MakeImmediate(int16_t value);  \
  static constexpr std::optional<Immediate> MakeImmediate(uint16_t value); \
  static constexpr std::optional<Immediate> MakeImmediate(int32_t value);  \
  static constexpr std::optional<Immediate> MakeImmediate(uint32_t value); \
  static constexpr std::optional<Immediate> MakeImmediate(int64_t value);  \
  static constexpr std::optional<Immediate> MakeImmediate(uint64_t value)
  BERBERIS_DEFINE_MAKE_IMMEDIATE(BImmediate, MakeBImmediate);
  BERBERIS_DEFINE_MAKE_IMMEDIATE(CsrImmediate, MakeCsrImmediate);
  BERBERIS_DEFINE_MAKE_IMMEDIATE(IImmediate, MakeImmediate);
  BERBERIS_DEFINE_MAKE_IMMEDIATE(IImmediate, MakeIImmediate);
  BERBERIS_DEFINE_MAKE_IMMEDIATE(JImmediate, MakeJImmediate);
  BERBERIS_DEFINE_MAKE_IMMEDIATE(PImmediate, MakePImmediate);
  BERBERIS_DEFINE_MAKE_IMMEDIATE(Shift32Immediate, MakeShift32Immediate);
  BERBERIS_DEFINE_MAKE_IMMEDIATE(Shift64Immediate, MakeShift64Immediate);
  BERBERIS_DEFINE_MAKE_IMMEDIATE(SImmediate, MakeSImmediate);
  BERBERIS_DEFINE_MAKE_IMMEDIATE(UImmediate, MakeUImmediate);
#undef BERBERIS_DEFINE_MAKE_IMMEDIATE

 private:
  // RawImmediate is used to bypass checks in constructor. It's not supposed to be used directly.
  class RawImmediate {
   private:
    friend class BImmediate;
    friend class CsrImmediate;
    friend class IImmediate;
    friend class JImmediate;
    friend class Shift32Immediate;
    friend class Shift64Immediate;
    friend class PImmediate;
    friend class SImmediate;
    friend class UImmediate;
    friend class AssemblerRiscV;

    constexpr RawImmediate(int32_t value) : value_(value) {}
    int32_t value_;
  };

 public:
#define BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(Immediate, IntType) \
  constexpr Immediate(IntType value) : Immediate(MakeRaw(value)) { CHECK(AccetableValue(value)); }
#define BERBERIS_DEFINE_IMMEDIATE(Immediate, kMaskValue, ...)                                    \
  class Immediate {                                                                              \
   public:                                                                                       \
    static constexpr int32_t kMask = static_cast<int32_t>(kMaskValue);                           \
                                                                                                 \
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(Immediate, int8_t)                                     \
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(Immediate, uint8_t)                                    \
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(Immediate, int16_t)                                    \
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(Immediate, uint16_t)                                   \
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(Immediate, int32_t)                                    \
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(Immediate, uint32_t)                                   \
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(Immediate, int64_t)                                    \
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(Immediate, uint64_t)                                   \
                                                                                                 \
    constexpr Immediate() : value_(0) {}                                                         \
                                                                                                 \
    constexpr int32_t EncodedValue() {                                                           \
      return value_;                                                                             \
    }                                                                                            \
                                                                                                 \
    friend bool operator==(Immediate const&, Immediate const&) = default;                        \
                                                                                                 \
    friend class AssemblerRiscV;                                                                 \
    __VA_ARGS__                                                                                  \
                                                                                                 \
   private:                                                                                      \
    constexpr Immediate(RawImmediate raw) : value_(raw.value_) {}                                \
    /* Return true if value would fit into immediate. */                                         \
    template <typename IntType>                                                                  \
    static constexpr bool AccetableValue(IntType value);                                         \
    /* Make RawImmediate from immediate value. */                                                \
    /* Note: value is not checked for correctness! Public interface is MakeImmediate factory. */ \
    template <typename IntType>                                                                  \
    static constexpr RawImmediate MakeRaw(IntType value);                                        \
                                                                                                 \
    int32_t value_;                                                                              \
  }
  BERBERIS_DEFINE_IMMEDIATE(BImmediate, 0xfe00'0f80);
  BERBERIS_DEFINE_IMMEDIATE(CsrImmediate, 0x000f'8000);
  BERBERIS_DEFINE_IMMEDIATE(
      IImmediate, 0xfff0'0000, constexpr IImmediate(SImmediate s_imm)
      : value_((s_imm.value_ & 0xfe00'0000) | ((s_imm.value_ & 0x0000'0f80) << 13)) {}

      friend SImmediate;);
  BERBERIS_DEFINE_IMMEDIATE(JImmediate, 0xffff'f000);
  BERBERIS_DEFINE_IMMEDIATE(PImmediate, 0xfe00'0000);
  BERBERIS_DEFINE_IMMEDIATE(Shift32Immediate, 0x01f00000);
  BERBERIS_DEFINE_IMMEDIATE(Shift64Immediate, 0x03f00000);
  BERBERIS_DEFINE_IMMEDIATE(
      SImmediate, 0xfe00'0f80, constexpr SImmediate(Immediate imm)
      : value_((imm.value_ & 0xfe00'0000) | ((imm.value_ & 0x01f0'0000) >> 13)) {}

      friend class IImmediate;);
  BERBERIS_DEFINE_IMMEDIATE(UImmediate, 0xffff'f000);
#undef BERBERIS_DEFINE_IMMEDIATE
#undef BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR

  // Macro operations.
  void Finalize() { ResolveJumps(); }

  void ResolveJumps();

// Instructions.
#include "berberis/assembler/gen_assembler_common_riscv-inl.h"  // NOLINT generated file!

 protected:
  // Information about operands.
  template <typename OperandType, typename = void>
  class OperandInfo;

  // Wrapped operand with information of where in the encoded instruction should it be placed.
  template <typename OperandMarker, typename RegisterType>
  struct RegisterOperand {
    constexpr int32_t EncodeImmediate() {
      return value.GetPhysicalIndex()
             << OperandInfo<RegisterOperand<OperandMarker, RegisterType>>::kOffset;
    }

    RegisterType value;
  };

  // Operand class  markers. Note, these classes shouldn't ever be instantiated, they are just used
  // to carry information about operands.
  class RdMarker;
  class RmMarker;
  class Rs1Marker;
  class Rs2Marker;
  class Rs3Marker;

  template <typename RegisterType>
  class OperandInfo<RegisterOperand<RdMarker, RegisterType>> {
   public:
    static constexpr bool IsImmediate = false;
    static constexpr uint8_t kOffset = 7;
    static constexpr uint32_t kMask = 0x0000'0f80;
  };

  template <typename RegisterType>
  class OperandInfo<RegisterOperand<RmMarker, RegisterType>> {
   public:
    static constexpr bool IsImmediate = false;
    static constexpr uint8_t kOffset = 12;
    static constexpr uint32_t kMask = 0x0000'7000;
  };

  template <typename RegisterType>
  class OperandInfo<RegisterOperand<Rs1Marker, RegisterType>> {
   public:
    static constexpr bool IsImmediate = false;
    static constexpr uint8_t kOffset = 15;
    static constexpr uint32_t kMask = 0x000f'8000;
  };

  template <typename RegisterType>
  class OperandInfo<RegisterOperand<Rs2Marker, RegisterType>> {
   public:
    static constexpr bool IsImmediate = false;
    static constexpr uint8_t kOffset = 20;
    static constexpr uint32_t kMask = 0x01f0'0000;
  };

  template <typename RegisterType>
  class OperandInfo<RegisterOperand<Rs3Marker, RegisterType>> {
   public:
    static constexpr bool IsImmediate = false;
    static constexpr uint8_t kOffset = 27;
    static constexpr uint32_t kMask = 0xf800'0000;
  };

  template <typename Immediate>
  class OperandInfo<Immediate, std::enable_if_t<sizeof(Immediate::kMask) != 0>> {
   public:
    static constexpr bool IsImmediate = true;
    static constexpr uint8_t kOffset = 0;
    static constexpr uint32_t kMask = Immediate::kMask;
  };

  template <typename RegisterType>
  RegisterOperand<RdMarker, RegisterType> Rd(RegisterType value) {
    return {value};
  }

  template <typename RegisterType>
  RegisterOperand<RmMarker, RegisterType> Rm(RegisterType value) {
    return {value};
  }

  template <typename RegisterType>
  RegisterOperand<Rs1Marker, RegisterType> Rs1(RegisterType value) {
    return {value};
  }

  template <typename RegisterType>
  RegisterOperand<Rs2Marker, RegisterType> Rs2(RegisterType value) {
    return {value};
  }

  template <typename RegisterType>
  RegisterOperand<Rs3Marker, RegisterType> Rs3(RegisterType value) {
    return {value};
  }

  template <uint32_t kOpcode, uint32_t kOpcodeMask, typename... ArgumentsTypes>
  void EmitInstruction(ArgumentsTypes... arguments) {
    // All uncompressed instructions in RISC-V have two lowest bit set and we don't handle
    // compressed instructions here.
    static_assert((kOpcode & 0b11) == 0b11);
    // Instruction shouldn't have any bits set outside of its opcode mask.
    static_assert((kOpcode & ~kOpcodeMask) == 0);
    // Places for all operands in the opcode should not intersect with opcode.
    static_assert((((kOpcodeMask & OperandInfo<ArgumentsTypes>::kMask) == 0) && ...));
    Emit32((kOpcode | ... | [](auto argument) {
      if constexpr (OperandInfo<decltype(argument)>::IsImmediate) {
        return argument.EncodedValue();
      } else {
        return argument.EncodeImmediate();
      }
    }(arguments)));
  }

  template <uint32_t kOpcode, typename ArgumentsType0, typename OperandType>
  void EmitITypeInstruction(ArgumentsType0&& argument0, OperandType&& operand) {
    return EmitInstruction<kOpcode, 0x0000'707f>(Rd(argument0), Rs1(operand.base), operand.disp);
  }

  template <uint32_t kOpcode,
            typename ArgumentsType0,
            typename ArgumentsType1,
            typename ImmediateType>
  void EmitITypeInstruction(ArgumentsType0&& argument0,
                            ArgumentsType1&& argument1,
                            ImmediateType&& immediate) {
    return EmitInstruction<kOpcode, 0x0000'707f>(Rd(argument0), Rs1(argument1), immediate);
  }

  template <uint32_t kOpcode,
            typename ArgumentsType0,
            typename ArgumentsType1,
            typename ArgumentsType2>
  void EmitRTypeInstruction(ArgumentsType0&& argument0,
                            ArgumentsType1&& argument1,
                            ArgumentsType2&& argument2) {
    return EmitInstruction<kOpcode, 0xfe00'707f>(Rd(argument0), Rs1(argument1), Rs2(argument2));
  }

  template <uint32_t kOpcode, typename ArgumentsType0, typename OperandType>
  void EmitSTypeInstruction(ArgumentsType0&& argument0, OperandType&& operand) {
    return EmitInstruction<kOpcode, 0x0000'707f>(Rs2(argument0), Rs1(operand.base), operand.disp);
  }

 private:
  AssemblerRiscV() = delete;
  AssemblerRiscV(const AssemblerRiscV&) = delete;
  AssemblerRiscV(AssemblerRiscV&&) = delete;
  void operator=(const AssemblerRiscV&) = delete;
  void operator=(AssemblerRiscV&&) = delete;
};

#define BERBERIS_DEFINE_MAKE_IMMEDIATE(Immediate, MakeImmediate, IntType)   \
  template <typename Assembler>                                             \
  constexpr std::optional<typename AssemblerRiscV<Assembler>::Immediate>    \
  AssemblerRiscV<Assembler>::MakeImmediate(IntType value) {                 \
    if (!AssemblerRiscV<Assembler>::Immediate::AccetableValue(value)) {     \
      return {};                                                            \
    }                                                                       \
    return Immediate{AssemblerRiscV<Assembler>::Immediate::MakeRaw(value)}; \
  }
#define BERBERIS_DEFINE_MAKE_IMMEDIATE_SET(Immediate, MakeImmediate) \
  BERBERIS_DEFINE_MAKE_IMMEDIATE(Immediate, MakeImmediate, int8_t)   \
  BERBERIS_DEFINE_MAKE_IMMEDIATE(Immediate, MakeImmediate, uint8_t)  \
  BERBERIS_DEFINE_MAKE_IMMEDIATE(Immediate, MakeImmediate, int16_t)  \
  BERBERIS_DEFINE_MAKE_IMMEDIATE(Immediate, MakeImmediate, uint16_t) \
  BERBERIS_DEFINE_MAKE_IMMEDIATE(Immediate, MakeImmediate, int32_t)  \
  BERBERIS_DEFINE_MAKE_IMMEDIATE(Immediate, MakeImmediate, uint32_t) \
  BERBERIS_DEFINE_MAKE_IMMEDIATE(Immediate, MakeImmediate, int64_t)  \
  BERBERIS_DEFINE_MAKE_IMMEDIATE(Immediate, MakeImmediate, uint64_t)
BERBERIS_DEFINE_MAKE_IMMEDIATE_SET(BImmediate, MakeBImmediate)
BERBERIS_DEFINE_MAKE_IMMEDIATE_SET(CsrImmediate, MakeCsrImmediate)
BERBERIS_DEFINE_MAKE_IMMEDIATE_SET(Immediate, MakeImmediate)
BERBERIS_DEFINE_MAKE_IMMEDIATE_SET(JImmediate, MakeJImmediate)
BERBERIS_DEFINE_MAKE_IMMEDIATE_SET(PImmediate, MakePImmediate)
BERBERIS_DEFINE_MAKE_IMMEDIATE_SET(Shift32Immediate, MakeShift32Immediate)
BERBERIS_DEFINE_MAKE_IMMEDIATE_SET(Shift64Immediate, MakeShift64Immediate)
BERBERIS_DEFINE_MAKE_IMMEDIATE_SET(SImmediate, MakeSImmediate)
BERBERIS_DEFINE_MAKE_IMMEDIATE_SET(UImmediate, MakeUImmediate)
#undef BERBERIS_DEFINE_MAKE_IMMEDIATE_SET
#undef BERBERIS_DEFINE_MAKE_IMMEDIATE

// Return true if value would fit into B-immediate.
template <typename Assembler>
template <typename IntType>
constexpr bool AssemblerRiscV<Assembler>::BImmediate::AccetableValue(IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // B-immediate accepts 12 bits, but encodes signed even values, that's why we only may accept
  // low 12 bits of any unsigned value.
  // Encode mask as the largest accepted value plus one and cut it to IntType size.
  constexpr uint64_t kUnsigned64bitInputMask = 0xffff'ffff'ffff'f001;
  if constexpr (!std::is_signed_v<IntType>) {
    constexpr IntType kUnsignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kUnsignedInputMask) == IntType{0};
  } else {
    // For signed values we also accept the same values as for unsigned case, but also accept
    // value that have all bits in am kUnsignedInputMask set.
    // B-immediate compresses these into one single sign bit, but lowest bit have to be zero.
    constexpr IntType kSignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kSignedInputMask) == IntType{0} ||
           static_cast<IntType>(value & kSignedInputMask) == (kSignedInputMask & ~int64_t{1});
  }
}

// Return true if value would fit into Csr-immediate.
template <typename Assembler>
template <typename IntType>
constexpr bool AssemblerRiscV<Assembler>::CsrImmediate::AccetableValue(IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // Csr immediate is unsigned immediate with possible values between 0 and 31.
  // If we make value unsigned negative numbers would become numbers >127 and would be rejected.
  return std::make_unsigned_t<IntType>(value) < 32;
}

// Return true if value would fit into immediate.
template <typename Assembler>
template <typename IntType>
constexpr bool AssemblerRiscV<Assembler>::Immediate::AccetableValue(IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // I-immediate accepts 12 bits, but encodes signed values, that's why we only may accept low
  // 11 bits of any unsigned value.
  // Encode mask as the largest accepted value plus one and cut it to IntType size.
  constexpr uint64_t kUnsigned64bitInputMask = 0xffff'ffff'ffff'f800;
  if constexpr (!std::is_signed_v<IntType>) {
    constexpr IntType kUnsignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kUnsignedInputMask) == IntType{0};
  } else {
    // For signed values we accept the same values as for unsigned case, but also accept
    // values that have all bits in kUnsignedInputMask set.
    // I-immediate compresses these into one single sign bit.
    constexpr IntType kSignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kSignedInputMask) == IntType{0} ||
           static_cast<IntType>(value & kSignedInputMask) == kSignedInputMask;
  }
}

// Return true if value would fit into J-immediate.
template <typename Assembler>
template <typename IntType>
constexpr bool AssemblerRiscV<Assembler>::JImmediate::AccetableValue(IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // J-immediate accepts 20 bits, but encodes signed even values, that's why we only may accept
  // bits from 1 to 19 of any unsigned value. Encode mask as the largest accepted value plus 1 and
  // cut it to IntType size.
  constexpr uint64_t kUnsigned64bitInputMask = 0xffff'ffff'fff0'0001;
  if constexpr (!std::is_signed_v<IntType>) {
    constexpr IntType kUnsignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kUnsignedInputMask) == IntType{0};
  } else {
    // For signed values we accept the same values as for unsigned case, but also accept
    // value that have all bits in kUnsignedInputMask set except zero bit (which is zero).
    // J-immediate compresses these into one single sign bit, but lowest bit have to be zero.
    constexpr IntType kSignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kSignedInputMask) == IntType{0} ||
           static_cast<IntType>(value & kSignedInputMask) == (kSignedInputMask & ~int64_t{1});
  }
}

// Return true if value would fit into P-immediate.
template <typename Assembler>
template <typename IntType>
constexpr bool AssemblerRiscV<Assembler>::PImmediate::AccetableValue(IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // P-immediate accepts 7 bits, but encodes only values divisible by 32, that's why we only may
  // accept bits from 5 to 10 of any unsigned value. Encode mask as the largest accepted value
  // plus 31 and cut it to IntType size.
  constexpr uint64_t kUnsigned64bitInputMask = 0xffff'ffff'ffff'f81f;
  if constexpr (!std::is_signed_v<IntType>) {
    constexpr IntType kUnsignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kUnsignedInputMask) == IntType{0};
  } else {
    // For signed values we accept the same values as for unsigned case, but also accept
    // value that have all bits in kUnsignedInputMask set except the lowest 5 bits (which are zero).
    // P-immediate compresses these into one single sign bit, but lowest bits have to be zero.
    constexpr IntType kSignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kSignedInputMask) == IntType{0} ||
           static_cast<IntType>(value & kSignedInputMask) == (kSignedInputMask & ~int64_t{0x1f});
  }
}

// Return true if value would fit into Shift32-immediate.
template <typename Assembler>
template <typename IntType>
constexpr bool AssemblerRiscV<Assembler>::Shift32Immediate::AccetableValue(IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // Shift32 immediate is unsigned immediate with possible values between 0 and 31.
  // If we make value unsigned negative numbers would become numbers >127 and would be rejected.
  return std::make_unsigned_t<IntType>(value) < 32;
}

// Return true if value would fit into Shift64-immediate.
template <typename Assembler>
template <typename IntType>
constexpr bool AssemblerRiscV<Assembler>::Shift64Immediate::AccetableValue(IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // Shift64 immediate is unsigned immediate with possible values between 0 and 63.
  // If we make value unsigned negative numbers would become numbers >127 and would be rejected.
  return std::make_unsigned_t<IntType>(value) < 64;
}

// Immediate (I-immediate in RISC V documentation) and S-Immediate are siblings: they encode
// the same values but in a different way.
// AccetableValue are the same for that reason, but MakeRaw are different.
template <typename Assembler>
template <typename IntType>
constexpr bool AssemblerRiscV<Assembler>::SImmediate::AccetableValue(IntType value) {
  return AssemblerRiscV<Assembler>::Immediate::AccetableValue(value);
}

// Return true if value would fit into U-immediate.
template <typename Assembler>
template <typename IntType>
constexpr bool AssemblerRiscV<Assembler>::UImmediate::AccetableValue(IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // U-immediate accepts 20 bits, but encodes only values divisible by 4096, that's why we only
  // may accept bits from 12 to 30 of any unsigned value. Encode mask as the largest accepted value
  // plus 4095 and cut it to IntType size.
  constexpr uint64_t kUnsigned64bitInputMask = 0xffff'ffff'8000'0fff;
  if constexpr (!std::is_signed_v<IntType>) {
    constexpr IntType kUnsignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kUnsignedInputMask) == IntType{0};
  } else {
    // For signed values we accept the same values as for unsigned case, but also accept
    // value that have all bits in kUnsignedInputMask set except lower 12 bits (which are zero).
    // U-immediate compresses these into one single sign bit, but lowest bits have to be zero.
    constexpr IntType kSignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kSignedInputMask) == IntType{0} ||
           static_cast<IntType>(value & kSignedInputMask) == (kSignedInputMask & ~int64_t{0xfff});
  }
}

// Make RawImmediate from immediate value.
// Note: value is not checked for correctness here! Public interface is MakeBImmediate factory.
template <typename Assembler>
template <typename IntType>
constexpr AssemblerRiscV<Assembler>::RawImmediate AssemblerRiscV<Assembler>::BImmediate::MakeRaw(
    IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // Note: we have to convert type to int32_t before processing it! Otherwise we would produce
  // incorrect value for negative inputs since one single input sign in the small immediate would
  // turn into many bits in the insruction.
  return (static_cast<int32_t>(value) & static_cast<int32_t>(0x8000'0000)) |
         ((static_cast<int32_t>(value) & static_cast<int32_t>(0x0000'0800)) >> 4) |
         ((static_cast<int32_t>(value) & static_cast<int32_t>(0x0000'001f)) << 7) |
         ((static_cast<int32_t>(value) & static_cast<int32_t>(0x0000'07e0)) << 20);
}

// Make RawImmediate from immediate value.
// Note: value is not checked for correctness here! Public interface is MakeImmediate factory.
template <typename Assembler>
template <typename IntType>
constexpr AssemblerRiscV<Assembler>::RawImmediate AssemblerRiscV<Assembler>::CsrImmediate::MakeRaw(
    IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // Note: this is correct if input value is between 0 and 31, but that would be checked in
  // MakeCsrImmediate.
  return static_cast<int32_t>(value) << 15;
}

// Make RawImmediate from immediate value.
// Note: value is not checked for correctness here! Public interface is MakeImmediate factory.
template <typename Assembler>
template <typename IntType>
constexpr AssemblerRiscV<Assembler>::RawImmediate AssemblerRiscV<Assembler>::Immediate::MakeRaw(
    IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  return static_cast<int32_t>(value) << 20;
}

// Make RawImmediate from immediate value.
// Note: value is not checked for correctness here! Public interface is MakeJImmediate factory.
template <typename Assembler>
template <typename IntType>
constexpr AssemblerRiscV<Assembler>::RawImmediate AssemblerRiscV<Assembler>::JImmediate::MakeRaw(
    IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // Note: we have to convert type to int32_t before processing it! Otherwise we would produce
  // incorrect value for negative inputs since one single input sign in the small immediate would
  // turn into many bits in the insruction.
  return (static_cast<int32_t>(value) & static_cast<int32_t>(0x800f'f000)) |
         ((static_cast<int32_t>(value) & static_cast<int32_t>(0x0000'0800)) << 9) |
         ((static_cast<int32_t>(value) & static_cast<int32_t>(0x0000'07fe)) << 20);
}

// Make RawImmediate from immediate value.
// Note: value is not checked for correctness here! Public interface is MakeImmediate factory.
template <typename Assembler>
template <typename IntType>
constexpr AssemblerRiscV<Assembler>::RawImmediate AssemblerRiscV<Assembler>::PImmediate::MakeRaw(
    IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // Note: this is correct if input value is divisible by 32, but that would be checked in
  // MakePImmediate.
  return static_cast<int32_t>(value) << 20;
}

// Make RawImmediate from immediate value.
// Note: value is not checked for correctness here! Public interface is MakeImmediate factory.
template <typename Assembler>
template <typename IntType>
constexpr AssemblerRiscV<Assembler>::RawImmediate
AssemblerRiscV<Assembler>::Shift32Immediate::MakeRaw(IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // Note: this is correct if input value is between 0 and 31, but that would be checked in
  // MakeShift32Immediate.
  return static_cast<int32_t>(value) << 20;
}

// Make RawImmediate from immediate value.
// Note: value is not checked for correctness here! Public interface is MakeImmediate factory.
template <typename Assembler>
template <typename IntType>
constexpr AssemblerRiscV<Assembler>::RawImmediate
AssemblerRiscV<Assembler>::Shift64Immediate::MakeRaw(IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // Note: this is only correct if input value is between 0 and 63, but that would be checked in
  // MakeShift64Immediate.
  return static_cast<int32_t>(value) << 20;
}

// Make RawImmediate from immediate value.
// Note: value is not checked for correctness here! Public interface is MakeSImmediate factory.
template <typename Assembler>
template <typename IntType>
constexpr AssemblerRiscV<Assembler>::RawImmediate AssemblerRiscV<Assembler>::SImmediate::MakeRaw(
    IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // Here, because we are only using platforms with 32bit ints conversion to 32bit signed int may
  // happen both before masking and after but we are doing it before for consistency.
  return ((static_cast<int32_t>(value) & static_cast<int32_t>(0xffff'ffe0)) << 20) |
         ((static_cast<int32_t>(value) & static_cast<int32_t>(0x0000'001f)) << 7);
}

// Make RawImmediate from immediate value.
// Note: value is not checked for correctness here! Public interface is MakeImmediate factory.
template <typename Assembler>
template <typename IntType>
constexpr AssemblerRiscV<Assembler>::RawImmediate AssemblerRiscV<Assembler>::UImmediate::MakeRaw(
    IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // Note: this is only correct if input value is between divisible by 4096 , but that would be
  // checked in MakeUImmediate.
  return static_cast<int32_t>(value);
}

template <typename Assembler>
inline void AssemblerRiscV<Assembler>::ResolveJumps() {}

}  // namespace berberis

#endif  // BERBERIS_ASSEMBLER_COMMON_X86_H_
