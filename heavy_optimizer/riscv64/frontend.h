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

#ifndef BERBERIS_HEAVY_OPTIMIZER_RISCV64_FRONTEND_H_
#define BERBERIS_HEAVY_OPTIMIZER_RISCV64_FRONTEND_H_

#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/base/arena_map.h"
#include "berberis/base/dependent_false.h"
#include "berberis/decoder/riscv64/decoder.h"
#include "berberis/decoder/riscv64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_arch.h"
#include "berberis/intrinsics/intrinsics.h"
#include "berberis/intrinsics/macro_assembler.h"
#include "berberis/runtime_primitives/platform.h"

#include "call_intrinsic.h"
#include "inline_intrinsic.h"
#include "simd_register.h"

namespace berberis {

class HeavyOptimizerFrontend {
 public:
  using CsrName = berberis::CsrName;
  using Decoder = Decoder<SemanticsPlayer<HeavyOptimizerFrontend>>;
  using Register = MachineReg;
  using FpRegister = SimdReg;
  using Float32 = intrinsics::Float32;
  using Float64 = intrinsics::Float64;

  struct MemoryOperand {
    Register base{0};
    // We call the following field "index" even though we do not scale it at the
    // moment.  We can add a scale as the need arises.
    Register index{0};
    uint64_t disp = 0;
  };

  explicit HeavyOptimizerFrontend(x86_64::MachineIR* machine_ir, GuestAddr pc)
      : pc_(pc),
        success_(true),
        builder_(machine_ir),
        flag_register_(machine_ir->AllocVReg()),
        is_uncond_branch_(false),
        branch_targets_(machine_ir->arena()) {
    StartRegion();
  }

  void CompareAndBranch(Decoder::BranchOpcode opcode, Register arg1, Register arg2, int16_t offset);
  void Branch(int32_t offset);
  void BranchRegister(Register base, int16_t offset);

  [[nodiscard]] Register GetImm(uint64_t imm);
  [[nodiscard]] Register Copy(Register value) {
    Register result = AllocTempReg();
    Gen<PseudoCopy>(result, value, 8);
    return result;
  }

  [[nodiscard]] Register GetReg(uint8_t reg);
  void SetReg(uint8_t reg, Register value);

  void Unimplemented();
  //
  // Instruction implementations.
  //
  void Nop();
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

  Register Ecall(Register /* syscall_nr */,
                 Register /* arg0 */,
                 Register /* arg1 */,
                 Register /* arg2 */,
                 Register /* arg3 */,
                 Register /* arg4 */,
                 Register /* arg5 */) {
    Unimplemented();
    return {};
  }

  void Store(Decoder::StoreOperandType operand_type, Register arg, int16_t offset, Register data);
  Register Load(Decoder::LoadOperandType operand_type, Register arg, int16_t offset);

  //
  // Atomic extensions.
  //

  template <typename IntType, bool aq, bool rl>
  Register Lr(Register /* addr */) {
    Unimplemented();
    return {};
  }

  template <typename IntType, bool aq, bool rl>
  Register Sc(Register /* addr */, Register /* data */) {
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

  //
  // F and D extensions.
  //
  [[nodiscard]] FpRegister GetFpReg(uint8_t reg);

  template <typename FloatType>
  [[nodiscard]] FpRegister GetFRegAndUnboxNan(uint8_t reg) {
    CHECK_LE(reg, kNumGuestFpRegs);
    FpRegister result = AllocTempSimdReg();
    builder_.GenGetSimd(result.machine_reg(), reg);
    FpRegister unboxed_result = AllocTempSimdReg();
    if (host_platform::kHasAVX) {
      builder_.Gen<x86_64::MacroUnboxNanFloat32AVX>(unboxed_result.machine_reg(),
                                                    result.machine_reg());
    } else {
      builder_.Gen<x86_64::MacroUnboxNanFloat32>(unboxed_result.machine_reg(),
                                                 result.machine_reg());
    }
    return unboxed_result;
  }

  template <typename FloatType>
  void NanBoxAndSetFpReg(uint8_t reg, FpRegister value) {
    CHECK_LE(reg, kNumGuestFpRegs);
    if (host_platform::kHasAVX) {
      builder_.Gen<x86_64::MacroNanBoxFloat32AVX>(value.machine_reg(), value.machine_reg());
    } else {
      builder_.Gen<x86_64::MacroNanBoxFloat32>(value.machine_reg());
    }

    builder_.GenSetSimd(reg, value.machine_reg());
  }

  template <typename DataType>
  FpRegister LoadFp(Register /* arg */, int16_t /* offset */) {
    Unimplemented();
    return {};
  }

  template <typename DataType>
  void StoreFp(Register /* arg */, int16_t /* offset */, FpRegister /* data */) {
    Unimplemented();
  }

  FpRegister Fmv(FpRegister /* arg */) {
    Unimplemented();
    return {};
  }

  //
  // V extension.
  //

  template <typename VOpArgs, typename... ExtraAegs>
  void OpVector(const VOpArgs& /*args*/, ExtraAegs... /*extra_args*/) {
    // TODO(b/300690740): develop and implement strategy which would allow us to support vector
    // intrinsics not just in the interpreter.
    Unimplemented();
  }

  //
  // Csr
  //

  Register UpdateCsr(Decoder::CsrOpcode /* opcode */, Register /* arg */, Register /* csr */) {
    Unimplemented();
    return {};
  }

  Register UpdateCsr(Decoder::CsrImmOpcode /* opcode */, uint8_t /* imm */, Register /* csr */) {
    Unimplemented();
    return {};
  }

  [[nodiscard]] bool success() const { return success_; }

  //
  // Intrinsic proxy methods.
  //

#include "berberis/intrinsics/translator_intrinsics_hooks-inl.h"

  //
  // Guest state getters/setters.
  //

  [[nodiscard]] GuestAddr GetInsnAddr() const { return pc_; }
  void IncrementInsnAddr(uint8_t insn_size) { pc_ += insn_size; }

  [[nodiscard]] bool IsRegionEndReached() const;
  void StartInsn();
  void Finalize(GuestAddr stop_pc);

  // These methods are exported only for testing.
  [[nodiscard]] const ArenaMap<GuestAddr, MachineInsnPosition>& branch_targets() const {
    return branch_targets_;
  }

  template <CsrName kName>
  [[nodiscard]] Register GetCsr() {
    Unimplemented();
    return {};
  }

  template <CsrName kName>
  void SetCsr(uint8_t /* imm */) {
    Unimplemented();
  }

  template <CsrName kName>
  void SetCsr(Register /* arg */) {
    Unimplemented();
  }

 private:
  // Specialization for AssemblerResType=void
  template <auto kFunction,
            typename AssemblerResType,
            typename... AssemblerArgType,
            std::enable_if_t<std::is_same_v<std::decay_t<AssemblerResType>, void>, bool> = true>
  void CallIntrinsic(AssemblerArgType... args) {
    if (TryInlineIntrinsicForHeavyOptimizer<kFunction>(&builder_, GetFlagsRegister(), args...)) {
      return;
    }

    CallIntrinsicImpl(&builder_, kFunction, GetFlagsRegister(), args...);
  }

  template <auto kFunction,
            typename AssemblerResType,
            typename... AssemblerArgType,
            std::enable_if_t<!std::is_same_v<std::decay_t<AssemblerResType>, void>, bool> = true>
  AssemblerResType CallIntrinsic(AssemblerArgType... args) {
    AssemblerResType result;

    if constexpr (std::is_same_v<AssemblerResType, Register>) {
      result = AllocTempReg();
    } else if constexpr (std::is_same_v<AssemblerResType, SimdReg>) {
      result = AllocTempSimdReg();
    } else if constexpr (std::is_same_v<AssemblerResType, std::tuple<Register, Register>>) {
      result = {AllocTempReg(), AllocTempReg()};
    } else if constexpr (std::is_same_v<AssemblerResType, std::tuple<SimdReg, Register>>) {
      result = {AllocTempSimdReg(), AllocTempReg()};
    } else if constexpr (std::is_same_v<AssemblerResType, std::tuple<SimdReg, SimdReg>>) {
      result = {AllocTempSimdReg(), AllocTempSimdReg()};
    } else if constexpr (std::is_same_v<AssemblerResType, std::tuple<SimdReg, SimdReg, SimdReg>>) {
      result = {AllocTempSimdReg(), AllocTempSimdReg(), AllocTempSimdReg()};
    } else if constexpr (std::is_same_v<AssemblerResType,
                                        std::tuple<SimdReg, SimdReg, SimdReg, SimdReg>>) {
      result = {AllocTempSimdReg(), AllocTempSimdReg(), AllocTempSimdReg(), AllocTempSimdReg()};
    } else {
      // This should not be reached by the compiler. If it is - there is a new result type that
      // needs to be supported.
      static_assert(kDependentTypeFalse<AssemblerResType>, "Unsupported result type");
    }

    if (TryInlineIntrinsicForHeavyOptimizer<kFunction>(
            &builder_, result, GetFlagsRegister(), args...)) {
      return result;
    }

    CallIntrinsicImpl(&builder_, kFunction, result, GetFlagsRegister(), args...);
    return result;
  }

  // Syntax sugar.
  template <typename InsnType, typename... Args>
  /*may_discard*/ InsnType* Gen(Args... args) {
    return builder_.Gen<InsnType, Args...>(args...);
  }

  static x86_64::Assembler::Condition ToAssemblerCond(Decoder::BranchOpcode opcode);

  [[nodiscard]] Register AllocTempReg();
  [[nodiscard]] SimdReg AllocTempSimdReg();
  [[nodiscard]] Register GetFlagsRegister() const { return flag_register_; };

  void GenJump(GuestAddr target);
  void ExitGeneratedCode(GuestAddr target);
  void ExitRegionIndirect(Register target);

  void GenRecoveryBlockForLastInsn();

  void ResolveJumps();
  void ReplaceJumpWithBranch(MachineBasicBlock* bb, MachineBasicBlock* target_bb);
  void UpdateBranchTargetsAfterSplit(GuestAddr addr,
                                     const MachineBasicBlock* old_bb,
                                     MachineBasicBlock* new_bb);

  void StartRegion() {
    auto* region_entry_bb = builder_.ir()->NewBasicBlock();
    auto* cont_bb = builder_.ir()->NewBasicBlock();
    builder_.ir()->AddEdge(region_entry_bb, cont_bb);
    builder_.StartBasicBlock(region_entry_bb);
    Gen<PseudoBranch>(cont_bb);
    builder_.StartBasicBlock(cont_bb);
  }

  GuestAddr pc_;
  bool success_;
  x86_64::MachineIRBuilder builder_;
  MachineReg flag_register_;
  bool is_uncond_branch_;
  // Contains IR positions of all guest instructions of the current region.
  // Also contains all branch targets which the current region jumps to.
  // If the target is outside of the current region the position is uninitialized,
  // i.e. it's basic block (position.first) is nullptr.
  ArenaMap<GuestAddr, MachineInsnPosition> branch_targets_;
};

}  // namespace berberis

#endif /* BERBERIS_HEAVY_OPTIMIZER_RISCV64_FRONTEND_H_ */
