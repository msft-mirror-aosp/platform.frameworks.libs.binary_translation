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

#ifndef BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_H_
#define BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_H_

#include <cstdint>

#include "berberis/assembler/common.h"
#include "berberis/assembler/x86_64.h"
#include "berberis/base/checks.h"
#include "berberis/base/macros.h"
#include "berberis/decoder/riscv64/decoder.h"
#include "berberis/decoder/riscv64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_riscv64.h"

namespace berberis {

class MachindeCode;

class LiteTranslator {
 public:
  using Decoder = Decoder<SemanticsPlayer<LiteTranslator>>;
  using Register = x86_64::Assembler::Register;
  using FpRegister = x86_64::Assembler::XMMRegister;
  using Condition = x86_64::Assembler::Condition;

  explicit LiteTranslator(MachineCode* machine_code)
      : as_(machine_code), success_(true), next_gp_reg_for_alloc_(0){};

  //
  // Instruction implementations.
  //

  Register Op(Decoder::OpOpcode opcode, Register arg1, Register arg2) {
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
        as_.Movq(res, arg1);
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

  Register Op32(Decoder::Op32Opcode opcode, Register arg1, Register arg2) {
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

  Register OpImm(Decoder::OpImmOpcode opcode, Register arg, int16_t imm) {
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

  Register OpImm32(Decoder::OpImm32Opcode opcode, Register arg, int16_t imm) {
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

  Register ShiftImm(Decoder::ShiftImmOpcode opcode, Register arg, uint16_t imm) {
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

  Register ShiftImm32(Decoder::ShiftImm32Opcode opcode, Register arg, uint16_t imm) {
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

  Register Lui(int32_t imm) {
    UNUSED(imm);
    Unimplemented();
    return {};
  }

  Register Auipc(int32_t imm) {
    UNUSED(imm);
    Unimplemented();
    return {};
  }

  Register Load(Decoder::LoadOperandType operand_type, Register arg, int16_t offset) {
    UNUSED(operand_type, arg, offset);
    Unimplemented();
    return {};
  }

  void Store(Decoder::StoreOperandType operand_type, Register arg, int16_t offset, Register data) {
    UNUSED(operand_type, arg, offset, data);
    Unimplemented();
  }

  Register Amo(Decoder::AmoOpcode opcode, Register arg1, Register arg2, bool aq, bool rl) {
    UNUSED(opcode, arg1, arg2, aq, rl);
    Unimplemented();
    return {};
  }

  void Branch(Decoder::BranchOpcode opcode, Register arg1, Register arg2, int16_t offset) {
    UNUSED(opcode, arg1, arg2, offset);
    Unimplemented();
  }

  Register JumpAndLink(int32_t offset, uint8_t insn_len) {
    UNUSED(offset, insn_len);
    Unimplemented();
    return {};
  }

  Register JumpAndLinkRegister(Register base, int16_t offset, uint8_t insn_len) {
    UNUSED(base, offset, insn_len);
    Unimplemented();
    return {};
  }

  Register Ecall(Register syscall_nr,
                 Register arg0,
                 Register arg1,
                 Register arg2,
                 Register arg3,
                 Register arg4,
                 Register arg5) {
    UNUSED(syscall_nr, arg0, arg1, arg2, arg3, arg4, arg5);
    Unimplemented();
    return {};
  }

  void Fence(Decoder::FenceOpcode /*opcode*/,
             Register /*src*/,
             bool sw,
             bool sr,
             bool /*so*/,
             bool /*si*/,
             bool pw,
             bool pr,
             bool /*po*/,
             bool /*pi*/) {
    UNUSED(sw, sr, pw, pr);
    Unimplemented();
  }

  void FenceI(Register /*arg*/, int16_t /*imm*/) { Unimplemented(); }

  void Nop() {}

  //
  // F and D extensions.
  //

  FpRegister OpFp(Decoder::OpFpOpcode opcode,
                  Decoder::FloatOperandType float_size,
                  uint8_t rm,
                  FpRegister arg1,
                  FpRegister arg2) {
    UNUSED(opcode, float_size, rm, arg1, arg2);
    Unimplemented();
    return {};
  }

  FpRegister OpFpNoRounding(Decoder::OpFpNoRoundingOpcode opcode,
                            Decoder::FloatOperandType float_size,
                            FpRegister arg1,
                            FpRegister arg2) {
    UNUSED(opcode, float_size, arg1, arg2);
    Unimplemented();
    return {};
  }

  Register OpFpGpRegisterTargetNoRounding(Decoder::OpFpGpRegisterTargetNoRoundingOpcode opcode,
                                          Decoder::FloatOperandType float_size,
                                          FpRegister arg1,
                                          FpRegister arg2) {
    UNUSED(opcode, float_size, arg1, arg2);
    Unimplemented();
    return {};
  }

  Register OpFpGpRegisterTargetSingleInputNoRounding(
      Decoder::OpFpGpRegisterTargetSingleInputNoRoundingOpcode opcode,
      Decoder::FloatOperandType float_size,
      FpRegister arg) {
    UNUSED(opcode, float_size, arg);
    Unimplemented();
    return {};
  }

  FpRegister OpFpSingleInput(Decoder::OpFpSingleInputOpcode opcode,
                             Decoder::FloatOperandType float_size,
                             uint8_t rm,
                             FpRegister arg) {
    UNUSED(opcode, float_size, rm, arg);
    Unimplemented();
    return {};
  }

  FpRegister Fmv(Register arg) {
    UNUSED(arg);
    Unimplemented();
    return {};
  }

  Register Fmv(Decoder::FloatOperandType float_size, FpRegister arg) {
    UNUSED(float_size, arg);
    Unimplemented();
    return {};
  }

  FpRegister Fcvt(Decoder::FloatOperandType target_operand_size,
                  Decoder::FloatOperandType source_operand_size,
                  uint8_t rm,
                  FpRegister arg) {
    UNUSED(target_operand_size, source_operand_size, rm, arg);
    Unimplemented();
    return {};
  }

  Register Fcvt(Decoder::FcvtOperandType target_operand_size,
                Decoder::FloatOperandType source_operand_size,
                uint8_t rm,
                FpRegister arg) {
    UNUSED(target_operand_size, source_operand_size, rm, arg);
    Unimplemented();
    return {};
  }

  FpRegister Fcvt(Decoder::FloatOperandType target_operand_size,
                  Decoder::FcvtOperandType source_operand_size,
                  uint8_t rm,
                  Register arg) {
    UNUSED(target_operand_size, source_operand_size, rm, arg);
    Unimplemented();
    return {};
  }

  FpRegister Fma(Decoder::FmaOpcode opcode,
                 Decoder::FloatOperandType float_size,
                 uint8_t rm,
                 FpRegister arg1,
                 FpRegister arg2,
                 FpRegister arg3) {
    UNUSED(opcode, float_size, rm, arg1, arg2, arg3);
    Unimplemented();
    return {};
  }

  FpRegister LoadFp(Decoder::FloatOperandType opcode, Register arg, int16_t offset) {
    UNUSED(opcode, arg, offset);
    Unimplemented();
    return {};
  }

  void StoreFp(Decoder::FloatOperandType opcode, Register arg, int16_t offset, FpRegister data) {
    UNUSED(opcode, arg, offset, data);
    Unimplemented();
  }

  Register Csr(Decoder::CsrOpcode opcode, Register arg, Decoder::CsrRegister csr) {
    UNUSED(opcode, arg, csr);
    Unimplemented();
    return {};
  }

  Register Csr(Decoder::CsrImmOpcode opcode, uint8_t imm, Decoder::CsrRegister csr) {
    UNUSED(opcode, imm, csr);
    Unimplemented();
    return {};
  }

  //
  // Guest state getters/setters.
  //

  Register GetReg(uint8_t reg) {
    CHECK_GT(reg, 0);
    CHECK_LT(reg, arraysize(ThreadState::cpu.x));
    Register result = AllocTempReg();
    int32_t offset = offsetof(ThreadState, cpu.x[0]) + reg * 8;
    as_.Movq(result, {.base = as_.rbp, .disp = offset});
    return result;
  }

  void SetReg(uint8_t reg, Register value) {
    CHECK_GT(reg, 0);
    CHECK_LT(reg, arraysize(ThreadState::cpu.x));
    int32_t offset = offsetof(ThreadState, cpu.x[0]) + reg * 8;
    as_.Movq({.base = as_.rbp, .disp = offset}, value);
  }

  FpRegister GetFpReg(uint8_t reg) {
    UNUSED(reg);
    Unimplemented();
    return {};
  }

  FpRegister GetFRegAndUnboxNaN(uint8_t reg, Decoder::FloatOperandType operand_type) {
    UNUSED(reg, operand_type);
    Unimplemented();
    return {};
  }

  FpRegister CanonicalizeNans(FpRegister value, Decoder::FloatOperandType operand_type) {
    UNUSED(value, operand_type);
    Unimplemented();
    return {};
  }

  Register CanonicalizeGpNans(Register value, Decoder::FloatOperandType operand_type) {
    UNUSED(value, operand_type);
    Unimplemented();
    return {};
  }

  void NanBoxAndSetFpReg(uint8_t reg, FpRegister value, Decoder::FloatOperandType operand_type) {
    UNUSED(reg, value, operand_type);
    Unimplemented();
  }

  //
  // Various helper methods.
  //

  Register GetImm(uint64_t imm) {
    UNUSED(imm);
    Unimplemented();
    return {};
  }

  void Unimplemented() { success_ = false; }

  x86_64::Assembler* as() { return &as_; }
  bool success() const { return success_; }

 private:
  Register AllocTempReg() {
    // TODO(286261771): Add rdx to registers, push it on stack in all instances that are clobbering
    // it.
    static constexpr x86_64::Assembler::Register kRegs[] = {x86_64::Assembler::rbx,
                                                            x86_64::Assembler::rsi,
                                                            x86_64::Assembler::rdi,
                                                            x86_64::Assembler::r8,
                                                            x86_64::Assembler::r9,
                                                            x86_64::Assembler::r10,
                                                            x86_64::Assembler::r11,
                                                            x86_64::Assembler::r12,
                                                            x86_64::Assembler::r13,
                                                            x86_64::Assembler::r14,
                                                            x86_64::Assembler::r15};
    CHECK_LT(next_gp_reg_for_alloc_, arraysize(kRegs));

    return kRegs[next_gp_reg_for_alloc_++];
  }

  x86_64::Assembler as_;
  bool success_;
  uint8_t next_gp_reg_for_alloc_;
};

}  // namespace berberis

#endif  // BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_H_
