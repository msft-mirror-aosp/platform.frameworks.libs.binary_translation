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
#include "berberis/base/dependent_false.h"
#include "berberis/base/macros.h"
#include "berberis/decoder/riscv64/decoder.h"
#include "berberis/decoder/riscv64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/intrinsics/intrinsics.h"
#include "berberis/intrinsics/intrinsics_float.h"
#include "berberis/intrinsics/macro_assembler.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/platform.h"

#include "allocator.h"
#include "call_intrinsic.h"
#include "inline_intrinsic.h"

namespace berberis {

class MachindeCode;

class LiteTranslator {
 public:
  using Assembler = MacroAssembler<x86_64::Assembler>;
  using Decoder = Decoder<SemanticsPlayer<LiteTranslator>>;
  using Register = Assembler::Register;
  // Note: on RISC-V architecture FP register and SIMD registers are disjoint, but on x86 they are
  // the same.
  using FpRegister = Assembler::XMMRegister;
  using SimdRegister = Assembler::XMMRegister;
  using Condition = Assembler::Condition;
  using Float32 = intrinsics::Float32;
  using Float64 = intrinsics::Float64;

  explicit LiteTranslator(MachineCode* machine_code, GuestAddr pc, LiteTranslateParams& params)
      : as_(machine_code),
        success_(true),
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
  Register Slli(Register arg, int8_t imm);
  Register Srli(Register arg, int8_t imm);
  Register Srai(Register arg, int8_t imm);
  Register ShiftImm32(Decoder::ShiftImm32Opcode opcode, Register arg, uint16_t imm);
  Register Rori(Register arg, int8_t shamt);
  Register Roriw(Register arg, int8_t shamt);
  Register Lui(int32_t imm);
  Register Auipc(int32_t imm);
  void CompareAndBranch(Decoder::BranchOpcode opcode, Register arg1, Register arg2, int16_t offset);
  void Branch(int32_t offset);
  void BranchRegister(Register base, int16_t offset);
  void ExitRegion(GuestAddr target);
  void ExitRegionIndirect(Register target);
  void Store(Decoder::StoreOperandType operand_type, Register arg, int16_t offset, Register data);
  Register Load(Decoder::LoadOperandType operand_type, Register arg, int16_t offset);

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

  template <typename DataType>
  FpRegister LoadFp(Register arg, int16_t offset) {
    FpRegister res = AllocTempSimdReg();
    as_.Movs<DataType>(res, {.base = arg, .disp = offset});
    return res;
  }

  template <typename DataType>
  void StoreFp(Register arg, int16_t offset, FpRegister data) {
    as_.Movs<DataType>({.base = arg, .disp = offset}, data);
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
    CHECK_LT(reg, arraysize(ThreadState::cpu.f));
    SimdRegister result = AllocTempSimdReg();
    int32_t offset = offsetof(ThreadState, cpu.f) + reg * sizeof(Float64);
    as_.Movsd(result, {.base = Assembler::rbp, .disp = offset});
    return result;
  }

  FpRegister GetFRegAndUnboxNan(uint8_t reg, Decoder::FloatOperandType operand_type) {
    SimdRegister result = GetFpReg(reg);
    switch (operand_type) {
      case Decoder::FloatOperandType::kFloat: {
        SimdRegister unboxed_result = AllocTempSimdReg();
        if (host_platform::kHasAVX) {
          as_.MacroUnboxNanAVX<Float32>(unboxed_result, result);
        } else {
          as_.MacroUnboxNan<Float32>(unboxed_result, result);
        }
        return unboxed_result;
      }
      case Decoder::FloatOperandType::kDouble:
        return result;
      // No support for half-precision and quad-precision operands.
      default:
        Unimplemented();
        return {};
    }
  }

  FpRegister CanonicalizeNan(FpRegister value, Decoder::FloatOperandType operand_type) {
    SimdRegister canonical_result = AllocTempSimdReg();
    switch (operand_type) {
      case Decoder::FloatOperandType::kFloat: {
        if (host_platform::kHasAVX) {
          as_.CanonicalizeNanAVX<Float32>(canonical_result, value);
        } else {
          as_.CanonicalizeNan<Float32>(canonical_result, value);
        }
        return canonical_result;
      }
      case Decoder::FloatOperandType::kDouble: {
        if (host_platform::kHasAVX) {
          as_.CanonicalizeNanAVX<Float64>(canonical_result, value);
        } else {
          as_.CanonicalizeNan<Float64>(canonical_result, value);
        }
        return canonical_result;
      }
      // No support for half-precision and quad-precision operands.
      default:
        Unimplemented();
        return {};
    }
  }

  void NanBoxAndSetFpReg(uint8_t reg, FpRegister value, Decoder::FloatOperandType operand_type) {
    CHECK_LT(reg, arraysize(ThreadState::cpu.f));
    int32_t offset = offsetof(ThreadState, cpu.f) + reg * sizeof(Float64);
    switch (operand_type) {
      case Decoder::FloatOperandType::kFloat:
        if (host_platform::kHasAVX) {
          as_.MacroNanBoxAVX<Float32>(value);
          as_.Vmovsd({.base = Assembler::rbp, .disp = offset}, value);
        } else {
          as_.Movsd({.base = Assembler::rbp, .disp = offset}, value);
        }
        break;
      case Decoder::FloatOperandType::kDouble:
        if (host_platform::kHasAVX) {
          as_.Vmovsd({.base = Assembler::rbp, .disp = offset}, value);
        } else {
          as_.Movsd({.base = Assembler::rbp, .disp = offset}, value);
        }
        break;
      // No support for half-precision and quad-precision operands.
      default:
        return Unimplemented();
    }
  }

  //
  // Various helper methods.
  //

  [[nodiscard]] Register GetFrm() {
    Register frm_reg = AllocTempReg();
    as_.Movb(frm_reg, {.base = Assembler::rbp, .disp = offsetof(ThreadState, cpu.csr_data)});
    as_.Andb(frm_reg, int8_t{0b111});
    return frm_reg;
  }

  [[nodiscard]] Register GetImm(uint64_t imm) {
    Register imm_reg = AllocTempReg();
    as_.Movq(imm_reg, imm);
    return imm_reg;
  }

  void Unimplemented() { success_ = false; }

  [[nodiscard]] Assembler* as() { return &as_; }
  [[nodiscard]] bool success() const { return success_; }

  void FreeTempRegs() {
    gp_allocator_.FreeTemps();
    simd_allocator_.FreeTemps();
  }

#include "berberis/intrinsics/translator_intrinsics_hooks-inl.h"

  bool is_region_end_reached() const { return is_region_end_reached_; }

  void IncrementInsnAddr(uint8_t insn_size) { pc_ += insn_size; }

  Register AllocTempReg() {
    if (auto reg_option = gp_allocator_.AllocTemp()) {
      return reg_option.value();
    }
    success_ = false;
    return {};
  };

  SimdRegister AllocTempSimdReg() {
    if (auto reg_option = simd_allocator_.AllocTemp()) {
      return reg_option.value();
    }
    success_ = false;
    return {};
  };

 private:
  template <auto kFunction, typename AssemblerResType, typename... AssemblerArgType>
  AssemblerResType CallIntrinsic(AssemblerArgType... args) {
    AssemblerResType result;
    if constexpr (std::is_same_v<AssemblerResType, Register>) {
      result = AllocTempReg();
    } else if constexpr (std::is_same_v<AssemblerResType, SimdRegister>) {
      result = AllocTempSimdReg();
    } else {
      // This should not be reached by the compiler. If it is - there is a new result type that
      // needs to be supported.
      static_assert(kDependentTypeFalse<AssemblerResType>, "Unsupported result type");
    }

    if (TryInlineIntrinsic<kFunction>(&as_, this, result, args...)) {
      return result;
    }

    call_intrinsic::CallIntrinsic<AssemblerResType>(as_, kFunction, result, args...);

    return result;
  }

  Assembler as_;
  bool success_;
  GuestAddr pc_;
  Allocator<Register> gp_allocator_;
  Allocator<SimdRegister> simd_allocator_;
  const LiteTranslateParams& params_;
  bool is_region_end_reached_;
};

}  // namespace berberis

#endif  // BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_H_
