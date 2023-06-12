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

#include <cstdint>

#include "lite_translator.h"

#include "berberis/assembler/common.h"
#include "berberis/assembler/x86_64.h"
#include "berberis/base/checks.h"
#include "berberis/base/macros.h"
#include "berberis/decoder/riscv64/decoder.h"

namespace berberis {

using Register = LiteTranslator::Register;
using Condition = LiteTranslator::Condition;

//
// Instruction implementations.
//

Register LiteTranslator::Op(Decoder::OpOpcode opcode, Register arg1, Register arg2) {
  using OpOpcode = Decoder::OpOpcode;
  Register res = AllocTempReg();
  switch (opcode) {
    case OpOpcode::kAdd:
      as_.Movq(res, arg1);
      as_.Addq(res, arg2);
      break;
    case OpOpcode::kSub:
      as_.Movq(res, arg1);
      as_.Subq(res, arg2);
      break;
    case OpOpcode::kAnd:
      as_.Movq(res, arg1);
      as_.Andq(res, arg2);
      break;
    case OpOpcode::kOr:
      as_.Movq(res, arg1);
      as_.Orq(res, arg2);
      break;
    case OpOpcode::kXor:
      as_.Movq(res, arg1);
      as_.Xorq(res, arg2);
      break;
    case OpOpcode::kSll:
    case OpOpcode::kSrl:
    case OpOpcode::kSra:
      as_.Movq(res, arg1);
      as_.Movq(as_.rcx, arg2);
      if (opcode == OpOpcode::kSrl) {
        as_.ShrqByCl(res);
      } else if (opcode == OpOpcode::kSll) {
        as_.ShlqByCl(res);
      } else if (opcode == OpOpcode::kSra) {
        as_.SarqByCl(res);
      } else {
        FATAL("Unexpected OpOpcode");
      }
      break;
    case OpOpcode::kSlt:
      as_.Xorq(res, res);
      as_.Cmpq(arg1, arg2);
      as_.Setcc(Condition::kLess, res);
      break;
    case OpOpcode::kSltu:
      as_.Xorq(res, res);
      as_.Cmpq(arg1, arg2);
      as_.Setcc(Condition::kBelow, res);
      break;
    case OpOpcode::kMul:
      as_.Movq(res, arg1);
      as_.Imulq(res, arg2);
      break;
    case OpOpcode::kMulh:
      as_.Movq(res, arg1);
      as_.Movq(as_.rax, arg1);
      as_.Imulq(arg2);
      as_.Movq(res, as_.rdx);
      break;
    case OpOpcode::kMulhsu: {
      as_.Movq(res, arg1);
      as_.Movq(as_.rax, arg2);
      as_.Mulq(res);
      as_.Sarq(res, int8_t{63});
      as_.Imulq(res, arg2);
      as_.Addq(res, as_.rdx);
      break;
    }
    case OpOpcode::kMulhu:
      as_.Movq(as_.rax, arg1);
      as_.Mulq(arg2);
      as_.Movq(res, as_.rdx);
      break;
    case OpOpcode::kDiv:
    case OpOpcode::kRem:
      as_.Movq(as_.rax, arg1);
      as_.Movq(as_.rdx, as_.rax);
      as_.Sarq(as_.rdx, int8_t{63});
      as_.Idivq(arg2);
      as_.Movq(res, opcode == OpOpcode::kDiv ? as_.rax : as_.rdx);
      break;
    case OpOpcode::kDivu:
    case OpOpcode::kRemu:
      as_.Movq(as_.rax, arg1);
      as_.Xorq(as_.rdx, as_.rdx);
      as_.Divq(arg2);
      as_.Movq(res, opcode == OpOpcode::kDivu ? as_.rax : as_.rdx);
      break;
    default:
      Unimplemented();
      return {};
  }
  return res;
}

Register LiteTranslator::Op32(Decoder::Op32Opcode opcode, Register arg1, Register arg2) {
  using Op32Opcode = Decoder::Op32Opcode;
  Register res = AllocTempReg();
  switch (opcode) {
    case Op32Opcode::kAddw:
      as_.Movl(res, arg1);
      as_.Addl(res, arg2);
      as_.Movsxlq(res, res);
      break;
    case Op32Opcode::kSubw:
      as_.Movl(res, arg1);
      as_.Subl(res, arg2);
      as_.Movsxlq(res, res);
      break;
    case Op32Opcode::kSllw:
    case Op32Opcode::kSrlw:
    case Op32Opcode::kSraw:
      as_.Movl(res, arg1);
      as_.Movl(as_.rcx, arg2);
      if (opcode == Op32Opcode::kSrlw) {
        as_.ShrlByCl(res);
      } else if (opcode == Op32Opcode::kSllw) {
        as_.ShllByCl(res);
      } else if (opcode == Op32Opcode::kSraw) {
        as_.SarlByCl(res);
      } else {
        FATAL("Unexpected Op32Opcode");
      }
      as_.Movsxlq(res, res);
      break;
    case Op32Opcode::kMulw:
      as_.Movl(res, arg1);
      as_.Imull(res, arg2);
      as_.Movsxlq(res, res);
      break;
    case Op32Opcode::kDivw:
    case Op32Opcode::kRemw:
      as_.Movl(as_.rax, arg1);
      as_.Movl(as_.rdx, as_.rax);
      as_.Sarl(as_.rdx, int8_t{31});
      as_.Idivl(arg2);
      as_.Movsxlq(res, opcode == Op32Opcode::kDivw ? as_.rax : as_.rdx);
      break;
    case Op32Opcode::kDivuw:
    case Op32Opcode::kRemuw:
      as_.Movl(as_.rax, arg1);
      as_.Xorl(as_.rdx, as_.rdx);
      as_.Divl(arg2);
      as_.Movsxlq(res, opcode == Op32Opcode::kDivuw ? as_.rax : as_.rdx);
      break;
    default:
      Unimplemented();
      return {};
  }
  return res;
}

Register LiteTranslator::OpImm(Decoder::OpImmOpcode opcode, Register arg, int16_t imm) {
  using OpImmOpcode = Decoder::OpImmOpcode;
  Register res = AllocTempReg();
  switch (opcode) {
    case OpImmOpcode::kAddi:
      as_.Movq(res, arg);
      as_.Addq(res, imm);
      break;
    case OpImmOpcode::kSlti:
      as_.Xorq(res, res);
      as_.Cmpq(arg, imm);
      as_.Setcc(Condition::kLess, res);
      break;
    case OpImmOpcode::kSltiu:
      as_.Xorq(res, res);
      as_.Cmpq(arg, imm);
      as_.Setcc(Condition::kBelow, res);
      break;
    case OpImmOpcode::kXori:
      as_.Movq(res, arg);
      as_.Xorq(res, imm);
      break;
    case OpImmOpcode::kOri:
      as_.Movq(res, arg);
      as_.Orq(res, imm);
      break;
    case OpImmOpcode::kAndi:
      as_.Movq(res, arg);
      as_.Andq(res, imm);
      break;
    default:
      Unimplemented();
      return {};
  }
  return res;
}

Register LiteTranslator::OpImm32(Decoder::OpImm32Opcode opcode, Register arg, int16_t imm) {
  Register res = AllocTempReg();
  switch (opcode) {
    case Decoder::OpImm32Opcode::kAddiw:
      as_.Movl(res, arg);
      as_.Addl(res, imm);
      as_.Movsxlq(res, res);
      break;
    default:
      Unimplemented();
      return {};
  }
  return res;
}

Register LiteTranslator::ShiftImm(Decoder::ShiftImmOpcode opcode, Register arg, uint16_t imm) {
  using ShiftImmOpcode = Decoder::ShiftImmOpcode;
  Register res = AllocTempReg();
  as_.Movq(res, arg);
  as_.Movq(as_.rcx, imm);
  if (opcode == ShiftImmOpcode::kSrli) {
    as_.ShrqByCl(res);
  } else if (opcode == ShiftImmOpcode::kSlli) {
    as_.ShlqByCl(res);
  } else if (opcode == ShiftImmOpcode::kSrai) {
    as_.SarqByCl(res);
  } else {
    Unimplemented();
    return {};
  }
  return res;
}

Register LiteTranslator::ShiftImm32(Decoder::ShiftImm32Opcode opcode, Register arg, uint16_t imm) {
  using ShiftImm32Opcode = Decoder::ShiftImm32Opcode;
  Register res = AllocTempReg();
  as_.Movl(res, arg);
  as_.Movl(as_.rcx, imm);
  if (opcode == ShiftImm32Opcode::kSrliw) {
    as_.ShrlByCl(res);
  } else if (opcode == ShiftImm32Opcode::kSlliw) {
    as_.ShllByCl(res);
  } else if (opcode == ShiftImm32Opcode::kSraiw) {
    as_.SarlByCl(res);
  } else {
    Unimplemented();
    return {};
  }
  as_.Movsxlq(res, res);
  return res;
}

}  // namespace berberis
