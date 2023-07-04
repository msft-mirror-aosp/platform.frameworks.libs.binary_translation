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
#include "berberis/guest_state/guest_state.h"
#include "berberis/intrinsics/intrinsics.h"
#include "berberis/intrinsics/intrinsics_float.h"
#include "berberis/intrinsics/macro_assembler.h"
#include "berberis/lite_translator/lite_translate_region.h"

namespace berberis {

class MachindeCode;

class LiteTranslator {
 public:
  using Assembler = MacroAssembler<x86_64::Assembler>;
  using Decoder = Decoder<SemanticsPlayer<LiteTranslator>>;
  using Register = Assembler::Register;
  using FpRegister = Assembler::XMMRegister;
  using Condition = Assembler::Condition;
  using Float32 = intrinsics::Float32;
  using Float64 = intrinsics::Float64;

  explicit LiteTranslator(MachineCode* machine_code, GuestAddr pc, LiteTranslateParams& params)
      : as_(machine_code),
        success_(true),
        next_gp_reg_for_alloc_(0),
        pc_(pc),
        params_(params),
        is_region_end_reached_(false){};

  //
  // Instruction implementations.
  //

  Register Op(Decoder::OpOpcode opcode, Register arg1, Register arg2);
  Register Op32(Decoder::Op32Opcode opcode, Register arg1, Register arg2);
  Register OpImm(Decoder::OpImmOpcode opcode, Register arg, int16_t imm);
  Register OpImm32(Decoder::OpImm32Opcode opcode, Register arg, int16_t imm);
  Register ShiftImm(Decoder::ShiftImmOpcode opcode, Register arg, uint16_t imm);
  Register ShiftImm32(Decoder::ShiftImm32Opcode opcode, Register arg, uint16_t imm);
  Register Lui(int32_t imm);
  Register Auipc(int32_t imm);
  void CompareAndBranch(Decoder::BranchOpcode opcode, Register arg1, Register arg2, int16_t offset);
  void Branch(int32_t offset);
  void BranchRegister(Register base, int16_t offset);
  void ExitRegion(GuestAddr target);
  void ExitRegionIndirect(Register target);
  void Store(Decoder::StoreOperandType operand_type, Register arg, int16_t offset, Register data);
  Register Load(Decoder::LoadOperandType operand_type, Register arg, int16_t offset);

  Register Amo(Decoder::AmoOpcode opcode, Register arg1, Register arg2, bool aq, bool rl) {
    UNUSED(opcode, arg1, arg2, aq, rl);
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

  Register OpFpGpRegisterTargetNoRounding(Decoder::OpFpGpRegisterTargetNoRoundingOpcode opcode,
                                          Decoder::FloatOperandType float_size,
                                          FpRegister arg1,
                                          FpRegister arg2) {
    UNUSED(opcode, float_size, arg1, arg2);
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

  GuestAddr GetInsnAddr() const { return pc_; }

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

  [[nodiscard]] Register GetFrm() {
    Unimplemented();
    return {};
  }

  [[nodiscard]] Register GetImm(uint64_t imm) {
    Register imm_reg = AllocTempReg();
    as_.Movq(imm_reg, imm);
    return imm_reg;
  }

  void Unimplemented() { success_ = false; }

  [[nodiscard]] Assembler* as() { return &as_; }
  [[nodiscard]] bool success() const { return success_; }

#include "berberis/intrinsics/translator_intrinsics_hooks-inl.h"

  bool is_region_end_reached() const { return is_region_end_reached_; }

  void IncrementInsnAddr(uint8_t insn_size) { pc_ += insn_size; }

  void FreeTempRegs() { next_gp_reg_for_alloc_ = 0; }

 private:
  template <auto kFunction, typename AssemblerResType, typename... AssemblerArgType>
  AssemblerResType CallIntrinsic(AssemblerArgType...) {
    Unimplemented();
    return {};
  }

  Register AllocTempReg() {
    // TODO(286261771): Add rdx to registers, push it on stack in all instances that are clobbering
    // it.
    static constexpr Assembler::Register kRegs[] = {Assembler::rbx,
                                                    Assembler::rsi,
                                                    Assembler::rdi,
                                                    Assembler::r8,
                                                    Assembler::r9,
                                                    Assembler::r10,
                                                    Assembler::r11,
                                                    Assembler::r12,
                                                    Assembler::r13,
                                                    Assembler::r14,
                                                    Assembler::r15};
    CHECK_LT(next_gp_reg_for_alloc_, arraysize(kRegs));

    return kRegs[next_gp_reg_for_alloc_++];
  }

  Assembler as_;
  bool success_;
  uint8_t next_gp_reg_for_alloc_;
  GuestAddr pc_;
  const LiteTranslateParams& params_;
  bool is_region_end_reached_;
};

}  // namespace berberis

#endif  // BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_H_
