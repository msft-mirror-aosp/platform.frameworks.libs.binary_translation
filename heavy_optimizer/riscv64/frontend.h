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
#include "berberis/decoder/riscv64/decoder.h"
#include "berberis/decoder/riscv64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

class HeavyOptimizerFrontend {
 public:
  using CsrName = berberis::CsrName;
  using Decoder = Decoder<SemanticsPlayer<HeavyOptimizerFrontend>>;
  using Register = MachineReg;
  using FpRegister = MachineReg;
  using Float32 = intrinsics::Float32;
  using Float64 = intrinsics::Float64;

  explicit HeavyOptimizerFrontend(x86_64::MachineIR* machine_ir, GuestAddr pc)
      : pc_(pc),
        builder_(machine_ir),
        is_uncond_branch_(false),
        branch_targets_(machine_ir->arena()) {
    StartRegion();
  }

  //
  // Various helper methods.
  //

  [[nodiscard]] GuestAddr GetInsnAddr() const { return pc_; }
  void IncrementInsnAddr(uint8_t insn_size) { pc_ += insn_size; }

  [[nodiscard]] bool IsRegionEndReached() const;
  void StartInsn();
  void Finalize(GuestAddr stop_pc);

 private:
  // Syntax sugar.
  template <typename InsnType, typename... Args>
  /*may_discard*/ InsnType* Gen(Args... args) {
    return builder_.Gen<InsnType, Args...>(args...);
  }

  void GenJump(GuestAddr target);
  void ExitGeneratedCode(GuestAddr target);

  void ResolveJumps();

  void Unimplemented();

  void StartRegion() {
    auto* region_entry_bb = builder_.ir()->NewBasicBlock();
    auto* cont_bb = builder_.ir()->NewBasicBlock();
    builder_.ir()->AddEdge(region_entry_bb, cont_bb);
    builder_.StartBasicBlock(region_entry_bb);
    Gen<PseudoBranch>(cont_bb);
    builder_.StartBasicBlock(cont_bb);
  }

  GuestAddr pc_;
  x86_64::MachineIRBuilder builder_;
  bool is_uncond_branch_;
  // Contains IR positions of all guest instructions of the current region.
  // Also contains all branch targets which the current region jumps to.
  // If the target is outside of the current region the position is uninitialized,
  // i.e. it's basic block (position.first) is nullptr.
  ArenaMap<GuestAddr, MachineInsnPosition> branch_targets_;
};

}  // namespace berberis

#endif /* BERBERIS_HEAVY_OPTIMIZER_RISCV64_FRONTEND_H_ */
