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

#ifndef BERBERIS_DECODER_RISCV64_DECODER_H_
#define BERBERIS_DECODER_RISCV64_DECODER_H_

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <type_traits>

#include "berberis/base/bit_util.h"

namespace berberis {

// Decode() method takes a sequence of bytes and decodes it into the instruction opcode and fields.
// The InsnConsumer's method corresponding to the decoded opcode is called with the decoded fields
// as an argument. Returned is the instruction size.
template <class InsnConsumer>
class Decoder {
 public:
  explicit Decoder(InsnConsumer* insn_consumer) : insn_consumer_(insn_consumer) {}

  // https://eel.is/c++draft/enum#dcl.enum-8
  // For an enumeration whose underlying type is fixed, the values of the enumeration are the values
  // of the underlying type. Otherwise, the values of the enumeration are the values representable
  // by a hypothetical integer type with minimal width M such that all enumerators can be
  // represented. The width of the smallest bit-field large enough to hold all the values of the
  // enumeration type is M. It is possible to define an enumeration that has values not defined by
  // any of its enumerators. If the enumerator-list is empty, the values of the enumeration are as
  // if the enumeration had a single enumerator with value 0.

  // To ensure that we wouldn't trigger UB by accident each opcode includes kMaxValue variant in
  // kOpOcode, kSystemOpcode and so on, which have all possible bit values set.
  //
  // Note: that value is not used anywhere in the code, it exists solely to make conversion of not
  // yet known to decoder RISC-V instructions robust.

  enum class AmoOpcode {
    kLr = 0b00010,
    kSc = 0b00011,
    kAmoswap = 0b00001,
    kAmoadd = 0b00000,
    kAmoxor = 0b00100,
    kAmoand = 0b01100,
    kAmoor = 0b01000,
    kAmomin = 0b10000,
    kAmomax = 0b10100,
    kAmominu = 0b11000,
    kAmomaxu = 0b11100,
    kMaxValue = 0b11111,
  };

  enum class BranchOpcode {
    kBeq = 0b000,
    kBne = 0b001,
    kBlt = 0b100,
    kBge = 0b101,
    kBltu = 0b110,
    kBgeu = 0b111,
    kMaxValue = 0b111,
  };

  enum class CsrOpcode {
    kCsrrw = 0b01,
    kCsrrs = 0b10,
    kCsrrc = 0b11,
    kMaxValue = 0b11,
  };

  enum class CsrImmOpcode {
    kCsrrwi = 0b01,
    kCsrrsi = 0b10,
    kCsrrci = 0b11,
    kMaxValue = 0b11,
  };

  enum class FmaOpcode {
    kFmadd = 0b00,
    kFmsub = 0b01,
    kFnmsub = 0b10,
    kFnmadd = 0b11,
    kMaxValue = 0b11,
  };

  enum class FenceOpcode {
    kFence = 0b0000,
    kFenceTso = 0b1000,
    kMaxValue = 0b1111,
  };

  enum class OpOpcode {
    kAdd = 0b0000'000'000,
    kSub = 0b0100'000'000,
    kSll = 0b0000'000'001,
    kSlt = 0b0000'000'010,
    kSltu = 0b0000'000'011,
    kXor = 0b0000'000'100,
    kSrl = 0b0000'000'101,
    kSra = 0b0100'000'101,
    kOr = 0b0000'000'110,
    kAnd = 0b0000'000'111,
    kMul = 0b0000'001'000,
    kMulh = 0b0000'001'001,
    kMulhsu = 0b0000'001'010,
    kMulhu = 0b0000'001'011,
    kDiv = 0b0000'001'100,
    kDivu = 0b0000'001'101,
    kRem = 0b0000'001'110,
    kRemu = 0b0000'001'111,
    kAndn = 0b0100'000'111,
    kOrn = 0b0100'000'110,
    kXnor = 0b0100'000'100,
    kMax = 0b0000'101'110,
    kMaxu = 0b0000'101'111,
    kMin = 0b0000'101'100,
    kMinu = 0b0000'101'101,
    kRol = 0b0110'000'001,
    kRor = 0b0110'000'101,
    kSh1add = 0b0010'000'010,
    kSh2add = 0b0010'000'100,
    kSh3add = 0b0010'000'110,
    kMaxValue = 0b1111'111'111,
  };

  enum class Op32Opcode {
    kAddw = 0b0000'000'000,
    kAdduw = 0b0000'100'000,
    kSubw = 0b0100'000'000,
    kSllw = 0b0000'000'001,
    kSrlw = 0b0000'000'101,
    kSraw = 0b0100'000'101,
    kMulw = 0b0000'001'000,
    kDivw = 0b0000'001'100,
    kDivuw = 0b0000'001'101,
    kRemw = 0b0000'001'110,
    kRemuw = 0b0000'001'111,
    kRolw = 0b0110'000'001,
    kRorw = 0b0110'000'101,
    kSh1adduw = 0b0010'000'010,
    kSh2adduw = 0b0010'000'100,
    kSh3adduw = 0b0010'000'110,
    kMaxValue = 0b1111'111'111,
  };

  enum class OpSingleInputOpcode {
    kZexth = 0b0000'100'100,
    kMaxValue = 0b1111'111'111,
  };

  enum class OpFpGpRegisterTargetNoRoundingOpcode {
    kFle = 0b00'000,
    kFlt = 0b00'001,
    kFeq = 0b00'010,
    kMaxValue = 0b11'111,
  };

  enum class OpFpGpRegisterTargetSingleInputNoRoundingOpcode {
    kFclass = 0b00'00000'001,
    kMaxValue = 0b11'11111'111,
  };

  enum class OpFpNoRoundingOpcode {
    kFSgnj = 0b00'000,
    kFSgnjn = 0b00'001,
    kFSgnjx = 0b00'010,
    kFMin = 0b01'000,
    kFMax = 0b01'001,
    kMaxValue = 0b11'111,
  };

  enum class OpFpOpcode {
    kFAdd = 0b00,
    kFSub = 0b01,
    kFMul = 0b10,
    kFDiv = 0b11,
    kMaxValue = 0b11,
  };

  enum class OpFpSingleInputOpcode {
    kFSqrt = 0b11'00000,
    kMaxValue = 0b11'11111,
  };

  enum class OpFpSingleInputNoRoundingOpcode {
    kFmv,
  };

  enum class OpImmOpcode {
    kAddi = 0b000,
    kSlti = 0b010,
    kSltiu = 0b011,
    kXori = 0b100,
    kOri = 0b110,
    kAndi = 0b111,
    kMaxValue = 0b111,
  };

  enum class OpImm32Opcode {
    kAddiw = 0b000,
    kMaxValue = 0b111,
  };

  enum class ShiftImmOpcode {
    kSlli = 0b000000'001,
    kSrli = 0b000000'101,
    kSrai = 0b010000'101,
    kMaxValue = 0b11111'111,
  };

  enum class ShiftImm32Opcode {
    kSlliw = 0b0000000'001,
    kSrliw = 0b0000000'101,
    kSraiw = 0b0100000'101,
    kMaxValue = 0b111111'111,
  };

  enum class BitmanipImmOpcode {
    kClz = 0b0110000'00000'001,
    kCpop = 0b0110000'00010'001,
    kCtz = 0b0110000'00001'001,
    kSextb = 0b0110000'00100'001,
    kSexth = 0b0110000'00101'001,
    kOrcb = 0b0010100'00111'101,
    kRev8 = 0b0110101'11000'101,
    kRori = 0b011000'101,
    kMaxValue = 0b111111'111111'111,
  };

  enum class BitmanipImm32Opcode {
    kClzw = 0b0110000'00000'001,
    kCpopw = 0b0110000'00010'001,
    kCtzw = 0b0110000'00001'001,
    kRoriw = 0b0110000'101,
    kSlliuw = 0b0000100'001,
    kMaxValue = 0b1111111'111111'111,
  };

  enum class SystemOpcode {
    kEcall = 0b000000000000'00000'000'00000,
    kEbreak = 0b000000000001'00000'000'00000,
    kMaxValue = 0b111111111111'11111'111'11111,
  };

  // Technically CsrRegister is instruction argument, but it's handling is closer to the handling
  // of opcode: instructions which deal with different registers have radically different semantic
  // while most combinations trigger “illegal instruction opcode”.

  enum class CsrRegister {
    kFFlags = 0b00'00'0000'0001,
    kFrm = 0b00'00'0000'0010,
    kFCsr = 0b00'00'0000'0011,
    kMaxValue = 0b11'11'1111'1111,
  };

  // Load/Store instruction include 3bit “width” field while all other floating-point instructions
  // include 2bit “fmt” field.
  //
  // Decoder unifies these differences and uses FloatOperandType for types of all floating-point
  // operands.
  //
  // Load/Store for regular instruction coulnd't be simiarly unified: Load instructions include
  // seven types, while Store instructions have only four.
  //
  // Fcvt instructions have their own operand type encoding because they are only supporting 32bit
  // and 64bit operands, there is no support for 8bit and 16bit operands.
  //
  // This is because Load can perform either sign-extension or zero-extension for all sizes except
  // 64bit (which doesn't need neither sign-extension nor zero-extension since operand size is the
  // same as register size in that case).

  enum class FcvtOperandType {
    k32bitSigned = 0b00000,
    k32bitUnsigned = 0b00001,
    k64bitSigned = 0b00010,
    k64bitUnsigned = 0b00011,
    kMaxValue = 0b11111,
  };

  enum class FloatOperandType {
    kFloat = 0b00,
    kDouble = 0b01,
    kHalf = 0b10,
    kQuad = 0b11,
    kMaxValue = 0b11,
  };

  enum class LoadOperandType {
    k8bitSigned = 0b000,
    k16bitSigned = 0b001,
    k32bitSigned = 0b010,
    k64bit = 0b011,
    k8bitUnsigned = 0b100,
    k16bitUnsigned = 0b101,
    k32bitUnsigned = 0b110,
    kMaxValue = 0b1111,
  };

  enum class StoreOperandType {
    k8bit = 0b000,
    k16bit = 0b001,
    k32bit = 0b010,
    k64bit = 0b011,
    kMaxValue = 0b111,
  };

  struct AmoArgs {
    AmoOpcode opcode;
    StoreOperandType operand_type;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
    bool rl : 1;
    bool aq : 1;
  };

  struct BranchArgs {
    BranchOpcode opcode;
    uint8_t src1;
    uint8_t src2;
    int16_t offset;
  };

  struct CsrArgs {
    CsrOpcode opcode;
    uint8_t dst;
    uint8_t src;
    CsrRegister csr;
  };

  struct CsrImmArgs {
    CsrImmOpcode opcode;
    uint8_t dst;
    uint8_t imm;
    CsrRegister csr;
  };

  struct FcvtFloatToFloatArgs {
    FloatOperandType dst_type;
    FloatOperandType src_type;
    uint8_t dst;
    uint8_t src;
    uint8_t rm;
  };

  struct FcvtFloatToIntegerArgs {
    FcvtOperandType dst_type;
    FloatOperandType src_type;
    uint8_t dst;
    uint8_t src;
    uint8_t rm;
  };

  struct FcvtIntegerToFloatArgs {
    FloatOperandType dst_type;
    FcvtOperandType src_type;
    uint8_t dst;
    uint8_t src;
    uint8_t rm;
  };

  struct FenceArgs {
    FenceOpcode opcode;
    uint8_t dst;
    uint8_t src;
    bool sw : 1;
    bool sr : 1;
    bool so : 1;
    bool si : 1;
    bool pw : 1;
    bool pr : 1;
    bool po : 1;
    bool pi : 1;
  };

  struct FenceIArgs {
    uint8_t dst;
    uint8_t src;
    int16_t imm;
  };

  struct FmaArgs {
    FmaOpcode opcode;
    FloatOperandType operand_type;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
    uint8_t src3;
    uint8_t rm;
  };

  struct JumpAndLinkArgs {
    uint8_t dst;
    int32_t offset;
    uint8_t insn_len;
  };

  struct JumpAndLinkRegisterArgs {
    uint8_t dst;
    uint8_t base;
    int16_t offset;
    uint8_t insn_len;
  };

  template <typename OpcodeType>
  struct OpArgsTemplate {
    OpcodeType opcode;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
  };

  struct OpSingleInputArgs {
    OpSingleInputOpcode opcode;
    uint8_t dst;
    uint8_t src;
  };

  using OpArgs = OpArgsTemplate<OpOpcode>;
  using Op32Args = OpArgsTemplate<Op32Opcode>;

  struct OpFpArgs {
    OpFpOpcode opcode;
    FloatOperandType operand_type;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
    uint8_t rm;
  };

  struct OpFpGpRegisterTargetNoRoundingArgs {
    OpFpGpRegisterTargetNoRoundingOpcode opcode;
    FloatOperandType operand_type;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
  };

  struct OpFpGpRegisterTargetSingleInputNoRoundingArgs {
    OpFpGpRegisterTargetSingleInputNoRoundingOpcode opcode;
    FloatOperandType operand_type;
    uint8_t dst;
    uint8_t src;
  };

  struct FmvFloatToIntegerArgs {
    FloatOperandType operand_type;
    uint8_t dst;
    uint8_t src;
  };

  struct FmvIntegerToFloatArgs {
    FloatOperandType operand_type;
    uint8_t dst;
    uint8_t src;
  };

  struct OpFpNoRoundingArgs {
    OpFpNoRoundingOpcode opcode;
    FloatOperandType operand_type;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
  };

  struct OpFpSingleInputArgs {
    OpFpSingleInputOpcode opcode;
    FloatOperandType operand_type;
    uint8_t dst;
    uint8_t src;
    uint8_t rm;
  };

  struct OpFpSingleInputNoRoundingArgs {
    OpFpSingleInputNoRoundingOpcode opcode;
    FloatOperandType operand_type;
    uint8_t dst;
    uint8_t src;
  };

  template <typename OpcodeType>
  struct OpImmArgsTemplate {
    OpcodeType opcode;
    uint8_t dst;
    uint8_t src;
    int16_t imm;
  };

  using OpImmArgs = OpImmArgsTemplate<OpImmOpcode>;
  using OpImm32Args = OpImmArgsTemplate<OpImm32Opcode>;

  template <typename OperandTypeEnum>
  struct LoadArgsTemplate {
    OperandTypeEnum operand_type;
    uint8_t dst;
    uint8_t src;
    int16_t offset;
  };

  using LoadArgs = LoadArgsTemplate<LoadOperandType>;
  using LoadFpArgs = LoadArgsTemplate<FloatOperandType>;

  template <typename OpcodeType>
  struct ShiftImmArgsTemplate {
    OpcodeType opcode;
    uint8_t dst;
    uint8_t src;
    uint8_t imm;
  };

  using ShiftImmArgs = ShiftImmArgsTemplate<ShiftImmOpcode>;
  using ShiftImm32Args = ShiftImmArgsTemplate<ShiftImm32Opcode>;

  template <typename OpcodeType>
  struct BitmanipImmArgsTemplate {
    OpcodeType opcode;
    uint8_t dst;
    uint8_t src;
    uint8_t shamt;
  };

  using BitmanipImmArgs = BitmanipImmArgsTemplate<BitmanipImmOpcode>;
  using BitmanipImm32Args = BitmanipImmArgsTemplate<BitmanipImm32Opcode>;

  template <typename OperandTypeEnum>
  struct StoreArgsTemplate {
    OperandTypeEnum operand_type;
    uint8_t src;
    int16_t offset;
    uint8_t data;
  };

  using StoreArgs = StoreArgsTemplate<StoreOperandType>;
  using StoreFpArgs = StoreArgsTemplate<FloatOperandType>;

  struct SystemArgs {
    SystemOpcode opcode;
  };

  struct UpperImmArgs {
    uint8_t dst;
    int32_t imm;
  };

  static uint8_t GetInsnSize(const uint16_t* code) {
    constexpr uint16_t kInsnLenMask = uint16_t{0b11};
    return ((*code & kInsnLenMask) != kInsnLenMask) ? 2 : 4;
  }

  uint8_t Decode(const uint16_t* code) {
    uint8_t insn_size = GetInsnSize(code);
    if (insn_size == 2) {
      code_ = *code;
      return DecodeCompressedInstruction();
    }
    CHECK_EQ(insn_size, 4);
    // Warning: do not cast and dereference the pointer
    // since the address may not be 4-bytes aligned.
    memcpy(&code_, code, sizeof(code_));
    return DecodeBaseInstruction();
  }

  uint8_t DecodeCompressedInstruction() {
    CompressedOpcode opcode_bits{(GetBits<uint8_t, 13, 3>() << 2) | GetBits<uint8_t, 0, 2>()};

    switch (opcode_bits) {
      case CompressedOpcode::kAddi4spn:
        DecodeCompressedAddi4spn();
        break;
      case CompressedOpcode::kFld:
        DecodeCompressedLoadStore<LoadStore::kLoad, FloatOperandType::kDouble>();
        break;
      case CompressedOpcode::kLw:
        DecodeCompressedLoadStore<LoadStore::kLoad, LoadOperandType::k32bitSigned>();
        break;
      case CompressedOpcode::kLd:
        DecodeCompressedLoadStore<LoadStore::kLoad, LoadOperandType::k64bit>();
        break;
      case CompressedOpcode::kFsd:
        DecodeCompressedLoadStore<LoadStore::kStore, FloatOperandType::kDouble>();
        break;
      case CompressedOpcode::kSw:
        DecodeCompressedLoadStore<LoadStore::kStore, StoreOperandType::k32bit>();
        break;
      case CompressedOpcode::kSd:
        DecodeCompressedLoadStore<LoadStore::kStore, StoreOperandType::k64bit>();
        break;
      case CompressedOpcode::kAddi:
        DecodeCompressedAddi();
        break;
      case CompressedOpcode::kAddiw:
        DecodeCompressedAddiw();
        break;
      case CompressedOpcode::kLi:
        DecodeCompressedLi();
        break;
      case CompressedOpcode::kLui_Addi16sp:
        DecodeCompressedLuiAddi16sp();
        break;
      case CompressedOpcode::kMisc_Alu:
        DecodeCompressedMiscAlu();
        break;
      case CompressedOpcode::kJ:
        DecodeCompressedJ();
        break;
      case CompressedOpcode::kBeqz:
      case CompressedOpcode::kBnez:
        DecodeCompressedBeqzBnez();
        break;
      case CompressedOpcode::kSlli:
        DecodeCompressedSlli();
        break;
      case CompressedOpcode::kFldsp:
        DecodeCompressedLoadsp<FloatOperandType::kDouble>();
        break;
      case CompressedOpcode::kLwsp:
        DecodeCompressedLoadsp<LoadOperandType::k32bitSigned>();
        break;
      case CompressedOpcode::kLdsp:
        DecodeCompressedLoadsp<LoadOperandType::k64bit>();
        break;
      case CompressedOpcode::kJr_Jalr_Mv_Add:
        DecodeCompressedJr_Jalr_Mv_Add();
        break;
      case CompressedOpcode::kFsdsp:
        DecodeCompressedStoresp<FloatOperandType::kDouble>();
        break;
      case CompressedOpcode::kSwsp:
        DecodeCompressedStoresp<StoreOperandType::k32bit>();
        break;
      case CompressedOpcode::kSdsp:
        DecodeCompressedStoresp<StoreOperandType::k64bit>();
        break;
      default:
        insn_consumer_->Unimplemented();
    }
    return 2;
  }

  void DecodeCompressedLi() {
    uint8_t low_imm = GetBits<uint8_t, 2, 5>();
    uint8_t high_imm = GetBits<uint8_t, 12, 1>();
    uint8_t rd = GetBits<uint8_t, 7, 5>();
    int8_t imm = SignExtend<6>((high_imm << 5) + low_imm);
    const OpImmArgs args = {
        .opcode = OpImmOpcode::kAddi,
        .dst = rd,
        .src = 0,
        .imm = imm,
    };
    insn_consumer_->OpImm(args);
  }

  void DecodeCompressedMiscAlu() {
    uint8_t r = GetBits<uint8_t, 7, 3>() + 8;
    uint8_t low_imm = GetBits<uint8_t, 2, 5>();
    uint8_t high_imm = GetBits<uint8_t, 12, 1>();
    uint8_t imm = (high_imm << 5) + low_imm;
    switch (GetBits<uint8_t, 10, 2>()) {
      case 0b00: {
        const ShiftImmArgs args = {
            .opcode = ShiftImmOpcode::kSrli,
            .dst = r,
            .src = r,
            .imm = imm,
        };
        return insn_consumer_->OpImm(args);
      }
      case 0b01: {
        const ShiftImmArgs args = {
            .opcode = ShiftImmOpcode::kSrai,
            .dst = r,
            .src = r,
            .imm = imm,
        };
        return insn_consumer_->OpImm(args);
      }
      case 0b10: {
        const OpImmArgs args = {
            .opcode = OpImmOpcode::kAndi,
            .dst = r,
            .src = r,
            .imm = SignExtend<6>(imm),
        };
        return insn_consumer_->OpImm(args);
      }
    }
    uint8_t rs2 = GetBits<uint8_t, 2, 3>() + 8;
    if (GetBits<uint8_t, 12, 1>() == 0) {
      OpOpcode opcode;
      switch (GetBits<uint8_t, 5, 2>()) {
        case 0b00:
          opcode = OpOpcode::kSub;
          break;
        case 0b01:
          opcode = OpOpcode::kXor;
          break;
        case 0b10:
          opcode = OpOpcode::kOr;
          break;
        case 0b11:
          opcode = OpOpcode::kAnd;
          break;
      }
      const OpArgs args = {
          .opcode = opcode,
          .dst = r,
          .src1 = r,
          .src2 = rs2,
      };
      return insn_consumer_->Op(args);
    } else {
      Op32Opcode opcode;
      switch (GetBits<uint8_t, 5, 2>()) {
        case 0b00:
          opcode = Op32Opcode::kSubw;
          break;
        case 0b01:
          opcode = Op32Opcode::kAddw;
          break;
        default:
          return Undefined();
      }
      const Op32Args args = {
          .opcode = opcode,
          .dst = r,
          .src1 = r,
          .src2 = rs2,
      };
      return insn_consumer_->Op(args);
    }
  }

  template <auto kOperandType>
  void DecodeCompressedStoresp() {
    uint8_t raw_imm = GetBits<uint8_t, 7, 6>();
    uint8_t rs2 = GetBits<uint8_t, 2, 5>();
    constexpr uint8_t k32bit[64] = {
        0x00, 0x10, 0x20, 0x30, 0x01, 0x11, 0x21, 0x31, 0x02, 0x12, 0x22, 0x32, 0x03,
        0x13, 0x23, 0x33, 0x04, 0x14, 0x24, 0x34, 0x05, 0x15, 0x25, 0x35, 0x06, 0x16,
        0x26, 0x36, 0x07, 0x17, 0x27, 0x37, 0x08, 0x18, 0x28, 0x38, 0x09, 0x19, 0x29,
        0x39, 0x0a, 0x1a, 0x2a, 0x3a, 0x0b, 0x1b, 0x2b, 0x3b, 0x0c, 0x1c, 0x2c, 0x3c,
        0x0d, 0x1d, 0x2d, 0x3d, 0x0e, 0x1e, 0x2e, 0x3e, 0x0f, 0x1f, 0x2f, 0x3f};
    constexpr uint8_t k64bit[64] = {
        0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x02, 0x12, 0x22, 0x32, 0x42,
        0x52, 0x62, 0x72, 0x04, 0x14, 0x24, 0x34, 0x44, 0x54, 0x64, 0x74, 0x06, 0x16,
        0x26, 0x36, 0x46, 0x56, 0x66, 0x76, 0x08, 0x18, 0x28, 0x38, 0x48, 0x58, 0x68,
        0x78, 0x0a, 0x1a, 0x2a, 0x3a, 0x4a, 0x5a, 0x6a, 0x7a, 0x0c, 0x1c, 0x2c, 0x3c,
        0x4c, 0x5c, 0x6c, 0x7c, 0x0e, 0x1e, 0x2e, 0x3e, 0x4e, 0x5e, 0x6e, 0x7e};
    int16_t imm = (((uint8_t(kOperandType) & 1) == 0) ? k32bit : k64bit)[raw_imm] << 2;
    const StoreArgsTemplate<decltype(kOperandType)> args = {
        .operand_type = kOperandType,
        .src = 2,
        .offset = imm,
        .data = rs2,
    };
    insn_consumer_->Store(args);
  }

  void DecodeCompressedLuiAddi16sp() {
    uint8_t low_imm = GetBits<uint8_t, 2, 5>();
    uint8_t high_imm = GetBits<uint8_t, 12, 1>();
    uint8_t rd = GetBits<uint8_t, 7, 5>();
    if (rd != 2) {
      int32_t imm = SignExtend<18>((high_imm << 17) + (low_imm << 12));
      const UpperImmArgs args = {
          .dst = rd,
          .imm = imm,
      };
      return insn_consumer_->Lui(args);
    }
    constexpr uint8_t kAddi16spLow[32] = {0x00, 0x08, 0x20, 0x28, 0x40, 0x48, 0x60, 0x68,
                                          0x10, 0x18, 0x30, 0x38, 0x50, 0x58, 0x70, 0x78,
                                          0x04, 0x0c, 0x24, 0x2c, 0x44, 0x4c, 0x64, 0x6c,
                                          0x14, 0x1c, 0x34, 0x3c, 0x54, 0x5c, 0x74, 0x7c};
    int16_t imm = SignExtend<10>((high_imm << 9) + (kAddi16spLow[low_imm] << 2));
    const OpImmArgs args = {
        .opcode = OpImmOpcode::kAddi,
        .dst = 2,
        .src = 2,
        .imm = imm,
    };
    insn_consumer_->OpImm(args);
  }

  enum class LoadStore { kLoad, kStore };

  template <enum LoadStore kLoadStore, auto kOperandType>
  void DecodeCompressedLoadStore() {
    uint8_t low_imm = GetBits<uint8_t, 5, 2>();
    uint8_t high_imm = GetBits<uint8_t, 10, 3>();
    uint8_t imm;
    if constexpr ((uint8_t(kOperandType) & 1) == 0) {
      constexpr uint8_t kLwLow[4] = {0x0, 0x40, 0x04, 0x44};
      imm = (kLwLow[low_imm] | high_imm << 3);
    } else {
      imm = (low_imm << 6 | high_imm << 3);
    }
    uint8_t rd = GetBits<uint8_t, 2, 3>();
    uint8_t rs = GetBits<uint8_t, 7, 3>();
    if constexpr (kLoadStore == LoadStore::kStore) {
      const StoreArgsTemplate<decltype(kOperandType)> args = {
          .operand_type = kOperandType,
          .src = uint8_t(8 + rs),
          .offset = imm,
          .data = uint8_t(8 + rd),
      };
      insn_consumer_->Store(args);
    } else {
      const LoadArgsTemplate<decltype(kOperandType)> args = {
          .operand_type = kOperandType,
          .dst = uint8_t(8 + rd),
          .src = uint8_t(8 + rs),
          .offset = imm,
      };
      insn_consumer_->Load(args);
    }
  }

  template <auto kOperandType>
  void DecodeCompressedLoadsp() {
    uint8_t low_imm = GetBits<uint8_t, 2, 5>();
    uint8_t high_imm = GetBits<uint8_t, 12, 1>();
    uint8_t rd = GetBits<uint8_t, 7, 5>();
    constexpr uint8_t k32bitLow[32] = {0x00, 0x10, 0x20, 0x30, 0x01, 0x11, 0x21, 0x31,
                                       0x02, 0x12, 0x22, 0x32, 0x03, 0x13, 0x23, 0x33,
                                       0x04, 0x14, 0x24, 0x34, 0x05, 0x15, 0x25, 0x35,
                                       0x06, 0x16, 0x26, 0x36, 0x07, 0x17, 0x27, 0x37};
    constexpr uint8_t k64bitLow[32] = {0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70,
                                       0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72,
                                       0x04, 0x14, 0x24, 0x34, 0x44, 0x54, 0x64, 0x74,
                                       0x06, 0x16, 0x26, 0x36, 0x46, 0x56, 0x66, 0x76};
    int16_t imm = (high_imm << 5) +
                  ((((uint8_t(kOperandType) & 1) == 0) ? k32bitLow : k64bitLow)[low_imm] << 2);
    const LoadArgsTemplate<decltype(kOperandType)> args = {
        .operand_type = kOperandType,
        .dst = rd,
        .src = 2,
        .offset = imm,
    };
    insn_consumer_->Load(args);
  }

  void DecodeCompressedAddi() {
    uint8_t low_imm = GetBits<uint8_t, 2, 5>();
    uint8_t high_imm = GetBits<uint8_t, 12, 1>();
    int8_t imm = SignExtend<6>(high_imm << 5 | low_imm);
    uint8_t r = GetBits<uint8_t, 7, 5>();
    if (r == 0 || imm == 0) {
      insn_consumer_->Nop();
    }
    const OpImmArgs args = {
        .opcode = OpImmOpcode::kAddi,
        .dst = r,
        .src = r,
        .imm = imm,
    };
    insn_consumer_->OpImm(args);
  }

  void DecodeCompressedAddiw() {
    uint8_t low_imm = GetBits<uint8_t, 2, 5>();
    uint8_t high_imm = GetBits<uint8_t, 12, 1>();
    int8_t imm = SignExtend<6>(high_imm << 5 | low_imm);
    uint8_t r = GetBits<uint8_t, 7, 5>();
    const OpImm32Args args = {
        .opcode = OpImm32Opcode::kAddiw,
        .dst = r,
        .src = r,
        .imm = imm,
    };
    insn_consumer_->OpImm(args);
  }

  void DecodeCompressedBeqzBnez() {
    constexpr uint16_t kBHigh[8] = {0x0, 0x8, 0x10, 0x18, 0x100, 0x108, 0x110, 0x118};
    constexpr uint8_t kBLow[32] = {0x00, 0x20, 0x02, 0x22, 0x04, 0x24, 0x06, 0x26, 0x40, 0x60, 0x42,
                                   0x62, 0x44, 0x64, 0x46, 0x66, 0x80, 0xa0, 0x82, 0xa2, 0x84, 0xa4,
                                   0x86, 0xa6, 0xc0, 0xe0, 0xc2, 0xe2, 0xc4, 0xe4, 0xc6, 0xe6};
    uint8_t low_imm = GetBits<uint8_t, 2, 5>();
    uint8_t high_imm = GetBits<uint8_t, 10, 3>();
    uint8_t rs = GetBits<uint8_t, 7, 3>();

    const BranchArgs args = {
        .opcode = BranchOpcode(GetBits<uint8_t, 13, 1>()),
        .src1 = uint8_t(8 + rs),
        .src2 = 0,
        .offset = int16_t(SignExtend<9>(kBHigh[high_imm] + kBLow[low_imm])),
    };
    insn_consumer_->CompareAndBranch(args);
  }

  void DecodeCompressedJ() {
    constexpr uint16_t kJHigh[32] = {
        0x0,    0x400,  0x100,  0x500,  0x200,  0x600,  0x300,  0x700,  0x10,   0x410,  0x110,
        0x510,  0x210,  0x610,  0x310,  0x710,  0xf800, 0xfc00, 0xf900, 0xfd00, 0xfa00, 0xfe00,
        0xfb00, 0xff00, 0xf810, 0xfc10, 0xf910, 0xfd10, 0xfa10, 0xfe10, 0xfb10, 0xff10,
    };
    constexpr uint8_t kJLow[64] = {
        0x0,  0x20, 0x2,  0x22, 0x4,  0x24, 0x6,  0x26, 0x8,  0x28, 0xa,  0x2a, 0xc,
        0x2c, 0xe,  0x2e, 0x80, 0xa0, 0x82, 0xa2, 0x84, 0xa4, 0x86, 0xa6, 0x88, 0xa8,
        0x8a, 0xaa, 0x8c, 0xac, 0x8e, 0xae, 0x40, 0x60, 0x42, 0x62, 0x44, 0x64, 0x46,
        0x66, 0x48, 0x68, 0x4a, 0x6a, 0x4c, 0x6c, 0x4e, 0x6e, 0xc0, 0xe0, 0xc2, 0xe2,
        0xc4, 0xe4, 0xc6, 0xe6, 0xc8, 0xe8, 0xca, 0xea, 0xcc, 0xec, 0xce, 0xee,
    };
    const JumpAndLinkArgs args = {
        .dst = 0,
        .offset =
            bit_cast<int16_t>(kJHigh[GetBits<uint16_t, 8, 5>()]) | kJLow[GetBits<uint16_t, 2, 6>()],
        .insn_len = 2,
    };
    insn_consumer_->JumpAndLink(args);
  }

  void DecodeCompressedAddi4spn() {
    constexpr uint8_t kAddi4spnHigh[16] = {
        0x0, 0x40, 0x80, 0xc0, 0x4, 0x44, 0x84, 0xc4, 0x8, 0x48, 0x88, 0xc8, 0xc, 0x4c, 0x8c, 0xcc};
    constexpr uint8_t kAddi4spnLow[16] = {
        0x0, 0x2, 0x1, 0x3, 0x10, 0x12, 0x11, 0x13, 0x20, 0x22, 0x21, 0x23, 0x30, 0x32, 0x31, 0x33};
    int16_t imm = (kAddi4spnHigh[GetBits<uint8_t, 9, 4>()] | kAddi4spnLow[GetBits<uint8_t, 5, 4>()])
                  << 2;
    // If immediate is zero then this instruction is treated as unimplemented.
    // This includes RISC-V dedicated 16bit “unimplemented instruction” 0x0000.
    if (imm == 0) {
      return Undefined();
    }
    const OpImmArgs args = {
        .opcode = OpImmOpcode::kAddi,
        .dst = uint8_t(8 + GetBits<uint8_t, 2, 3>()),
        .src = 2,
        .imm = imm,
    };
    insn_consumer_->OpImm(args);
  }

  void DecodeCompressedJr_Jalr_Mv_Add() {
    uint8_t r = GetBits<uint8_t, 7, 5>();
    uint8_t rs2 = GetBits<uint8_t, 2, 5>();
    if (GetBits<uint8_t, 12, 1>()) {
      if (r == 0 && rs2 == 0) {
        const SystemArgs args = {
            .opcode = SystemOpcode::kEbreak,
        };
        return insn_consumer_->System(args);
      } else if (rs2 == 0) {
        const JumpAndLinkRegisterArgs args = {
            .dst = 1,
            .base = r,
            .offset = 0,
            .insn_len = 2,
        };
        insn_consumer_->JumpAndLinkRegister(args);
      } else {
        const OpArgs args = {
            .opcode = OpOpcode::kAdd,
            .dst = r,
            .src1 = r,
            .src2 = rs2,
        };
        insn_consumer_->Op(args);
      }
    } else {
      if (rs2 == 0) {
        const JumpAndLinkRegisterArgs args = {
            .dst = 0,
            .base = r,
            .offset = 0,
            .insn_len = 2,
        };
        insn_consumer_->JumpAndLinkRegister(args);
      } else {
        const OpArgs args = {
            .opcode = OpOpcode::kAdd,
            .dst = r,
            .src1 = 0,
            .src2 = rs2,
        };
        insn_consumer_->Op(args);
      }
    }
  }

  void DecodeCompressedSlli() {
    uint8_t r = GetBits<uint8_t, 7, 5>();
    uint8_t low_imm = GetBits<uint8_t, 2, 5>();
    uint8_t high_imm = GetBits<uint8_t, 12, 1>();
    uint8_t imm = (high_imm << 5) + low_imm;
    const ShiftImmArgs args = {
        .opcode = ShiftImmOpcode::kSlli,
        .dst = r,
        .src = r,
        .imm = imm,
    };
    return insn_consumer_->OpImm(args);
  }

  uint8_t DecodeBaseInstruction() {
    BaseOpcode opcode_bits{GetBits<uint8_t, 2, 5>()};

    switch (opcode_bits) {
      case BaseOpcode::kLoad:
        DecodeLoad<LoadOperandType>();
        break;
      case BaseOpcode::kLoadFp:
        DecodeLoad<FloatOperandType>();
        break;
      case BaseOpcode::kMiscMem:
        DecodeMiscMem();
        break;
      case BaseOpcode::kOpImm:
        DecodeOp<OpImmOpcode, ShiftImmOpcode, BitmanipImmOpcode, 6>();
        break;
      case BaseOpcode::kAuipc:
        DecodeAuipc();
        break;
      case BaseOpcode::kOpImm32:
        DecodeOp<OpImm32Opcode, ShiftImm32Opcode, BitmanipImm32Opcode, 5>();
        break;
      case BaseOpcode::kStore:
        DecodeStore<StoreOperandType>();
        break;
      case BaseOpcode::kStoreFp:
        DecodeStore<FloatOperandType>();
        break;
      case BaseOpcode::kAmo:
        DecodeAmo();
        break;
      case BaseOpcode::kOp:
        DecodeOp<OpOpcode>();
        break;
      case BaseOpcode::kLui:
        DecodeLui();
        break;
      case BaseOpcode::kOp32:
        DecodeOp<Op32Opcode>();
        break;
      case BaseOpcode::kMAdd:
      case BaseOpcode::kMSub:
      case BaseOpcode::kNmSub:
      case BaseOpcode::kNmAdd:
        DecodeFma();
        break;
      case BaseOpcode::kOpFp:
        DecodeOpFp();
        break;
      case BaseOpcode::kBranch:
        DecodeBranch();
        break;
      case BaseOpcode::kJalr:
        DecodeJumpAndLinkRegister();
        break;
      case BaseOpcode::kJal:
        DecodeJumpAndLink();
        break;
      case BaseOpcode::kSystem:
        DecodeSystem();
        break;
      default:
        insn_consumer_->Unimplemented();
    }
    return 4;
  }

 private:
  template <typename ResultType, uint32_t start, uint32_t size>
  ResultType GetBits() {
    static_assert(std::is_unsigned_v<ResultType>, "Only unsigned types are supported");
    static_assert(sizeof(ResultType) * CHAR_BIT >= size, "Too small ResultType for size");
    static_assert((start + size) <= 32 && size > 0, "Invalid start or size value");
    uint32_t shifted_val = code_ << (32 - start - size);
    return static_cast<ResultType>(shifted_val >> (32 - size));
  }

  // Signextend bits from size to the corresponding signed type of sizeof(Type) size.
  // If the result of this function is assigned to a wider signed type it'll automatically
  // sign-extend.
  template <unsigned size, typename Type>
  static auto SignExtend(const Type val) {
    static_assert(std::is_integral_v<Type>, "Only integral types are supported");
    static_assert(size > 0 && size < (sizeof(Type) * CHAR_BIT), "Invalid size value");
    typedef std::make_signed_t<Type> SignedType;
    struct {
      SignedType val : size;
    } holder = {.val = static_cast<SignedType>(val)};
    // Compiler takes care of sign-extension of the field with the specified bit-length.
    return static_cast<SignedType>(holder.val);
  }

  void Undefined() {
    // TODO(b/265372622): Handle undefined differently from unimplemented.
    insn_consumer_->Unimplemented();
  }

  void DecodeMiscMem() {
    uint8_t low_opcode = GetBits<uint8_t, 12, 3>();
    switch (low_opcode) {
      case 0b000: {
        uint8_t high_opcode = GetBits<uint8_t, 28, 4>();
        FenceOpcode opcode = FenceOpcode{high_opcode};
        const FenceArgs args = {
            .opcode = opcode,
            .dst = GetBits<uint8_t, 7, 5>(),
            .src = GetBits<uint8_t, 15, 5>(),
            .sw = bool(GetBits<uint8_t, 20, 1>()),
            .sr = bool(GetBits<uint8_t, 21, 1>()),
            .so = bool(GetBits<uint8_t, 22, 1>()),
            .si = bool(GetBits<uint8_t, 23, 1>()),
            .pw = bool(GetBits<uint8_t, 24, 1>()),
            .pr = bool(GetBits<uint8_t, 25, 1>()),
            .pi = bool(GetBits<uint8_t, 26, 1>()),
            .po = bool(GetBits<uint8_t, 27, 1>()),
        };
        insn_consumer_->Fence(args);
        break;
      }
      case 0b001: {
        uint16_t imm = GetBits<uint16_t, 20, 12>();
        const FenceIArgs args = {
            .dst = GetBits<uint8_t, 7, 5>(),
            .src = GetBits<uint8_t, 15, 5>(),
            .imm = SignExtend<12>(imm),
        };
        insn_consumer_->FenceI(args);
        break;
      }
      default:
        return Undefined();
    }
  }

  template <typename OpcodeType>
  void DecodeOp() {
    uint16_t low_opcode = GetBits<uint16_t, 12, 3>();
    uint16_t high_opcode = GetBits<uint16_t, 25, 7>();
    uint16_t opcode_bits = int16_t(low_opcode | (high_opcode << 3));
    OpcodeType opcode{opcode_bits};
    OpSingleInputOpcode single_input_opcode{opcode_bits};

    switch (single_input_opcode) {
      case OpSingleInputOpcode::kZexth: {
        DecodeSingleInputOp(single_input_opcode);
        return;
      }
      default:
        break;
    }
    const OpArgsTemplate<OpcodeType> args = {
        .opcode = opcode,
        .dst = GetBits<uint8_t, 7, 5>(),
        .src1 = GetBits<uint8_t, 15, 5>(),
        .src2 = GetBits<uint8_t, 20, 5>(),
    };
    insn_consumer_->Op(args);
  }

  void DecodeSingleInputOp(OpSingleInputOpcode opcode) {
    uint8_t src1 = GetBits<uint8_t, 15, 5>();
    uint8_t src2 = GetBits<uint8_t, 20, 5>();

    if (src2 != 0) {
      return Undefined();
    }
    const OpSingleInputArgs args = {.opcode = opcode, .dst = GetBits<uint8_t, 7, 5>(), .src = src1};
    insn_consumer_->OpSingleInput(args);
  }

  void DecodeAmo() {
    uint16_t low_opcode = GetBits<uint16_t, 12, 3>();
    uint16_t high_opcode = GetBits<uint16_t, 27, 5>();
    // lr instruction must have rs2 == 0
    if (high_opcode == 0b00010 && GetBits<uint8_t, 20, 5>() != 0) {
      return Undefined();
    }
    AmoOpcode opcode = AmoOpcode{high_opcode};
    StoreOperandType operand_type = StoreOperandType{low_opcode};
    const AmoArgs args = {
        .opcode = opcode,
        .operand_type = operand_type,
        .dst = GetBits<uint8_t, 7, 5>(),
        .src1 = GetBits<uint8_t, 15, 5>(),
        .src2 = GetBits<uint8_t, 20, 5>(),
        .rl = bool(GetBits<uint8_t, 25, 1>()),
        .aq = bool(GetBits<uint8_t, 26, 1>()),
    };
    insn_consumer_->Amo(args);
  }

  void DecodeFma() {
    uint8_t operand_type = GetBits<uint8_t, 25, 2>();
    uint8_t opcode_bits = GetBits<uint8_t, 2, 2>();
    const FmaArgs args = {
        .opcode = FmaOpcode(opcode_bits),
        .operand_type = FloatOperandType(operand_type),
        .dst = GetBits<uint8_t, 7, 5>(),
        .src1 = GetBits<uint8_t, 15, 5>(),
        .src2 = GetBits<uint8_t, 20, 5>(),
        .src3 = GetBits<uint8_t, 27, 5>(),
        .rm = GetBits<uint8_t, 12, 3>(),
    };
    insn_consumer_->Fma(args);
  }

  void DecodeLui() {
    int32_t imm = GetBits<uint32_t, 12, 20>();
    const UpperImmArgs args = {
        .dst = GetBits<uint8_t, 7, 5>(),
        .imm = imm << 12,
    };
    insn_consumer_->Lui(args);
  }

  void DecodeAuipc() {
    int32_t imm = GetBits<uint32_t, 12, 20>();
    const UpperImmArgs args = {
        .dst = GetBits<uint8_t, 7, 5>(),
        .imm = imm << 12,
    };
    insn_consumer_->Auipc(args);
  }

  template <typename OperandTypeEnum>
  void DecodeLoad() {
    OperandTypeEnum operand_type;
    if constexpr (std::is_same_v<OperandTypeEnum, FloatOperandType>) {
      auto decoded_operand_type = kLoadStoreWidthToFloatOperandType[GetBits<uint8_t, 12, 3>()];
      if (!decoded_operand_type.has_value()) {
        return Undefined();
      }
      operand_type = *decoded_operand_type;
    } else {
      operand_type = OperandTypeEnum{GetBits<uint8_t, 12, 3>()};
    }
    const LoadArgsTemplate<OperandTypeEnum> args = {
        .operand_type = operand_type,
        .dst = GetBits<uint8_t, 7, 5>(),
        .src = GetBits<uint8_t, 15, 5>(),
        .offset = SignExtend<12>(GetBits<uint16_t, 20, 12>()),
    };
    insn_consumer_->Load(args);
  }

  template <typename OperandTypeEnum>
  void DecodeStore() {
    OperandTypeEnum operand_type;
    if constexpr (std::is_same_v<OperandTypeEnum, FloatOperandType>) {
      auto decoded_operand_type = kLoadStoreWidthToFloatOperandType[GetBits<uint8_t, 12, 3>()];
      if (!decoded_operand_type.has_value()) {
        return Undefined();
      }
      operand_type = *decoded_operand_type;
    } else {
      operand_type = OperandTypeEnum{GetBits<uint8_t, 12, 3>()};
    }

    uint16_t low_imm = GetBits<uint16_t, 7, 5>();
    uint16_t high_imm = GetBits<uint16_t, 25, 7>();

    const StoreArgsTemplate<OperandTypeEnum> args = {
        .operand_type = operand_type,
        .src = GetBits<uint8_t, 15, 5>(),
        .offset = SignExtend<12>(int16_t(low_imm | (high_imm << 5))),
        .data = GetBits<uint8_t, 20, 5>(),
    };
    insn_consumer_->Store(args);
  }

  template <typename OpOpcodeType,
            typename ShiftOcodeType,
            typename BitmanipOpCodeType,
            uint32_t kShiftFieldSize>
  void DecodeOp() {
    uint8_t low_opcode = GetBits<uint8_t, 12, 3>();
    if (low_opcode != 0b001 && low_opcode != 0b101) {
      OpOpcodeType opcode{low_opcode};

      uint16_t imm = GetBits<uint16_t, 20, 12>();

      const OpImmArgsTemplate<OpOpcodeType> args = {
          .opcode = opcode,
          .dst = GetBits<uint8_t, 7, 5>(),
          .src = GetBits<uint8_t, 15, 5>(),
          .imm = SignExtend<12>(imm),
      };
      insn_consumer_->OpImm(args);
    } else if ((GetBits<uint16_t, 31, 1>() +
                GetBits<uint16_t, 20 + kShiftFieldSize, 10 - kShiftFieldSize>()) ==
               0) {  // For Canonical Shift Instructions from RV64G the opcode contains all
                     // zeros except for the 30th (second highest) bit.
      uint16_t high_opcode = GetBits<uint16_t, 20 + kShiftFieldSize, 12 - kShiftFieldSize>();
      ShiftOcodeType opcode{int16_t(low_opcode | (high_opcode << 3))};

      const ShiftImmArgsTemplate<ShiftOcodeType> args = {
          .opcode = opcode,
          .dst = GetBits<uint8_t, 7, 5>(),
          .src = GetBits<uint8_t, 15, 5>(),
          .imm = GetBits<uint8_t, 20, kShiftFieldSize>(),
      };
      insn_consumer_->OpImm(args);
    } else {
      uint8_t shamt = GetBits<uint8_t, 20, kShiftFieldSize>();
      uint16_t high_opcode = GetBits<uint16_t, 20 + kShiftFieldSize, 12 - kShiftFieldSize>();
      BitmanipOpCodeType opcode{int16_t(low_opcode | (high_opcode << 3))};
      bool is_shift = false;

      switch ((BitmanipImmOpcode)opcode) {
        case BitmanipImmOpcode::kRori:
          is_shift = true;
          break;
        default:
          break;
      }

      switch ((BitmanipImm32Opcode)opcode) {
        case BitmanipImm32Opcode::kRoriw:
        case BitmanipImm32Opcode::kSlliuw:
          is_shift = true;
          break;
        default:
          break;
      }
      // TODO(b/291851792): Refactor instructions with shamt into ShiftImmArgs
      if (!is_shift) {
        high_opcode = GetBits<uint16_t, 20, 12>();
        opcode = {low_opcode | (high_opcode << 3)};
        shamt = 0;
      }
      const BitmanipImmArgsTemplate<BitmanipOpCodeType> args = {
          .opcode = opcode,
          .dst = GetBits<uint8_t, 7, 5>(),
          .src = GetBits<uint8_t, 15, 5>(),
          .shamt = shamt,
      };
      insn_consumer_->OpImm(args);
    }
  }

  void DecodeBranch() {
    BranchOpcode opcode{GetBits<uint8_t, 12, 3>()};

    // Decode the offset.
    auto low_imm = GetBits<uint16_t, 8, 4>();
    auto mid_imm = GetBits<uint16_t, 25, 6>();
    auto bit11_imm = GetBits<uint16_t, 7, 1>();
    auto bit12_imm = GetBits<uint16_t, 31, 1>();
    auto offset =
        static_cast<int16_t>(low_imm | (mid_imm << 4) | (bit11_imm << 10) | (bit12_imm << 11));

    const BranchArgs args = {
        .opcode = opcode,
        .src1 = GetBits<uint8_t, 15, 5>(),
        .src2 = GetBits<uint8_t, 20, 5>(),
        // The offset is encoded as 2-byte units, we need to multiply by 2.
        .offset = SignExtend<13>(int16_t(offset * 2)),
    };
    insn_consumer_->CompareAndBranch(args);
  }

  void DecodeJumpAndLink() {
    // Decode the offset.
    auto low_imm = GetBits<uint32_t, 21, 10>();
    auto mid_imm = GetBits<uint32_t, 12, 8>();
    auto bit11_imm = GetBits<uint32_t, 20, 1>();
    auto bit20_imm = GetBits<uint32_t, 31, 1>();
    auto offset =
        static_cast<int32_t>(low_imm | (bit11_imm << 10) | (mid_imm << 11) | (bit20_imm << 19));

    const JumpAndLinkArgs args = {
        .dst = GetBits<uint8_t, 7, 5>(),
        // The offset is encoded as 2-byte units, we need to multiply by 2.
        .offset = SignExtend<21>(offset * 2),
        .insn_len = 4,
    };
    insn_consumer_->JumpAndLink(args);
  }

  void DecodeOpFp() {
    // Bit #29 = 1: means rm is an opcode extension and not operand.
    // Bit #30 = 1: means rs2 is an opcode extension and not operand.
    // Bit #31 = 1: selects general purpose register instead of floating point register as target.
    uint8_t operand_type = GetBits<uint8_t, 25, 2>();
    uint8_t opcode_bits = GetBits<uint8_t, 27, 2>();
    uint8_t rd = GetBits<uint8_t, 7, 5>();
    uint8_t rs1 = GetBits<uint8_t, 15, 5>();
    uint8_t rs2 = GetBits<uint8_t, 20, 5>();
    uint8_t rm = GetBits<uint8_t, 12, 3>();
    switch (GetBits<uint8_t, 29, 3>()) {
      case 0b000: {
        const OpFpArgs args = {
            .opcode = OpFpOpcode(opcode_bits),
            .operand_type = FloatOperandType(operand_type),
            .dst = rd,
            .src1 = rs1,
            .src2 = rs2,
            .rm = rm,
        };
        return insn_consumer_->OpFp(args);
      }
      case 0b001: {
        OpFpNoRoundingOpcode no_rounding_opcode = OpFpNoRoundingOpcode((opcode_bits << 3) + rm);
        if (no_rounding_opcode == Decoder::OpFpNoRoundingOpcode::kFSgnj && rs1 == rs2) {
          const OpFpSingleInputNoRoundingArgs args = {
              .opcode = OpFpSingleInputNoRoundingOpcode::kFmv,
              .operand_type = FloatOperandType(operand_type),
              .dst = rd,
              .src = rs1,
          };
          return insn_consumer_->OpFpSingleInputNoRounding(args);
        }
        const OpFpNoRoundingArgs args = {
            .opcode = no_rounding_opcode,
            .operand_type = FloatOperandType(operand_type),
            .dst = rd,
            .src1 = rs1,
            .src2 = rs2,
        };
        return insn_consumer_->OpFpNoRounding(args);
      }
      case 0b010: {
        if (opcode_bits == 0) {
          // Conversion from one float to the same float type is not supported.
          if (operand_type == rs2) {
            return Undefined();
          }
          // Values larger than 0b11 are reserved in Fcvt.
          if (rs2 > 0b11) {
            return Undefined();
          }
          const FcvtFloatToFloatArgs args = {
              .dst_type = FloatOperandType(operand_type),
              .src_type = FloatOperandType(rs2),
              .dst = rd,
              .src = rs1,
              .rm = rm,
          };
          return insn_consumer_->Fcvt(args);
        }
        uint8_t opcode = (opcode_bits << 5) + rs2;
        const OpFpSingleInputArgs args = {
            .opcode = OpFpSingleInputOpcode(opcode),
            .operand_type = FloatOperandType(operand_type),
            .dst = rd,
            .src = rs1,
            .rm = rm,
        };
        return insn_consumer_->OpFpSingleInput(args);
      }
      case 0b101: {
        uint8_t opcode = (opcode_bits << 3) + rm;
        const OpFpGpRegisterTargetNoRoundingArgs args = {
            .opcode = OpFpGpRegisterTargetNoRoundingOpcode(opcode),
            .operand_type = FloatOperandType(operand_type),
            .dst = rd,
            .src1 = rs1,
            .src2 = rs2,
        };
        return insn_consumer_->OpFpGpRegisterTargetNoRounding(args);
      }
      case 0b110:
        switch (opcode_bits) {
          case 0b00: {
            const FcvtFloatToIntegerArgs args = {
                .dst_type = FcvtOperandType(rs2),
                .src_type = FloatOperandType(operand_type),
                .dst = rd,
                .src = rs1,
                .rm = rm,
            };
            return insn_consumer_->Fcvt(args);
          }
          case 0b10: {
            const FcvtIntegerToFloatArgs args = {
                .dst_type = FloatOperandType(operand_type),
                .src_type = FcvtOperandType(rs2),
                .dst = rd,
                .src = rs1,
                .rm = rm,
            };
            return insn_consumer_->Fcvt(args);
          }
          default:
            return Undefined();
        }
      case 0b111: {
        uint16_t opcode = (opcode_bits << 8) + (rs2 << 3) + rm;
        switch (rm) {
          case 0b001: {
            const OpFpGpRegisterTargetSingleInputNoRoundingArgs args = {
                .opcode = OpFpGpRegisterTargetSingleInputNoRoundingOpcode(opcode),
                .operand_type = FloatOperandType(operand_type),
                .dst = rd,
                .src = rs1,
            };
            return insn_consumer_->OpFpGpRegisterTargetSingleInputNoRounding(args);
          }
          case 0b000: {
            if (opcode_bits == 0b00) {
              const FmvFloatToIntegerArgs args = {
                  .operand_type = FloatOperandType(operand_type),
                  .dst = rd,
                  .src = rs1,
              };
              return insn_consumer_->FmvFloatToInteger(args);
            } else if (opcode_bits == 0b10) {
              const FmvIntegerToFloatArgs args = {
                  .operand_type = FloatOperandType(operand_type),
                  .dst = rd,
                  .src = rs1,
              };
              return insn_consumer_->FmvIntegerToFloat(args);
            } else {
              return Undefined();
            }
          }
          default:
            return Undefined();
        }
      }
      default:
        return Undefined();
    }
  }

  void DecodeSystem() {
    uint8_t low_opcode = GetBits<uint8_t, 12, 2>();
    if (low_opcode == 0b00) {
      int32_t opcode = GetBits<uint32_t, 7, 25>();
      const SystemArgs args = {
          .opcode = SystemOpcode(opcode),
      };
      return insn_consumer_->System(args);
    }
    if (GetBits<uint8_t, 14, 1>()) {
      CsrImmOpcode opcode = CsrImmOpcode(low_opcode);
      const CsrImmArgs args = {
          .opcode = opcode,
          .dst = GetBits<uint8_t, 7, 5>(),
          .imm = GetBits<uint8_t, 15, 5>(),
          .csr = CsrRegister(GetBits<uint16_t, 20, 12>()),
      };
      return insn_consumer_->Csr(args);
    }
    CsrOpcode opcode = CsrOpcode(low_opcode);
    const CsrArgs args = {
        .opcode = opcode,
        .dst = GetBits<uint8_t, 7, 5>(),
        .src = GetBits<uint8_t, 15, 5>(),
        .csr = CsrRegister(GetBits<uint16_t, 20, 12>()),
    };
    return insn_consumer_->Csr(args);
  }

  void DecodeJumpAndLinkRegister() {
    if (GetBits<uint8_t, 12, 3>() != 0b000) {
      Undefined();
      return;
    }
    // Decode sign-extend offset.
    int16_t offset = GetBits<uint16_t, 20, 12>();
    offset = static_cast<int16_t>(offset << 4) >> 4;

    const JumpAndLinkRegisterArgs args = {
        .dst = GetBits<uint8_t, 7, 5>(),
        .base = GetBits<uint8_t, 15, 5>(),
        .offset = offset,
        .insn_len = 4,
    };
    insn_consumer_->JumpAndLinkRegister(args);
  }

  enum class BaseOpcode {
    kLoad = 0b00'000,
    kLoadFp = 0b00'001,
    kCustom0 = 0b00'010,
    kMiscMem = 0b00'011,
    kOpImm = 0b00'100,
    kAuipc = 0b00'101,
    kOpImm32 = 0b00'110,
    // Reserved 0b00'111,
    kStore = 0b01'000,
    kStoreFp = 0b01'001,
    kCustom1 = 0b01'010,
    kAmo = 0b01'011,
    kOp = 0b01'100,
    kLui = 0b01'101,
    kOp32 = 0b01'110,
    // Reserved 0b01'111,
    kMAdd = 0b10'000,
    kMSub = 0b10'001,
    kNmSub = 0b10'010,
    kNmAdd = 0b10'011,
    kOpFp = 0b10'100,
    // Reserved 0b10'101,
    kCustom2 = 0b10'110,
    // Reserved 0b10'111,
    kBranch = 0b11'000,
    kJalr = 0b11'001,
    // Reserved 0b11'010,
    kJal = 0b11'011,
    kSystem = 0b11'100,
    // Reserved 0b11'101,
    kCustom3 = 0b11'110,
    // Reserved 0b11'111,
    kMaxValue = 0b11'111,
  };

  enum class CompressedOpcode {
    kAddi4spn = 0b00'000,
    kFld = 0b001'00,
    kLw = 0b010'00,
    kLd = 0b011'00,
    // Reserved 0b00'100
    kFsd = 0b101'00,
    kSw = 0b110'00,
    kSd = 0b111'00,
    kAddi = 0b000'01,
    kAddiw = 0b001'01,
    kLi = 0b010'01,
    kLui_Addi16sp = 0b011'01,
    kMisc_Alu = 0b100'01,
    kJ = 0b101'01,
    kBeqz = 0b110'01,
    kBnez = 0b111'01,
    kSlli = 0b000'10,
    kFldsp = 0b001'10,
    kLwsp = 0b010'10,
    kLdsp = 0b011'10,
    kJr_Jalr_Mv_Add = 0b100'10,
    kFsdsp = 0b101'10,
    kSwsp = 0b110'10,
    kSdsp = 0b111'10,
    // instruction with 0bxxx'11 opcodes are not compressed instruction and can not be in this
    // table.
    kMaxValue = 0b111'11,
  };

  static constexpr std::optional<FloatOperandType> kLoadStoreWidthToFloatOperandType[8] = {
      {},
      {FloatOperandType::kHalf},
      {FloatOperandType::kFloat},
      {FloatOperandType::kDouble},
      {FloatOperandType::kQuad},
      {},
      {},
      {}};

  InsnConsumer* insn_consumer_;
  uint32_t code_;
};

}  // namespace berberis

#endif  // BERBERIS_DECODER_RISCV64_DECODER_H_
