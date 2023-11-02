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

#include "frontend.h"

#include <cstddef>

#include "berberis/assembler/x86_64.h"
#include "berberis/backend/common/machine_ir.h"
#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/guest_state/guest_state_arch.h"
#include "berberis/runtime_primitives/platform.h"

namespace berberis {

using BranchOpcode = HeavyOptimizerFrontend::Decoder::BranchOpcode;
using FpRegister = HeavyOptimizerFrontend::FpRegister;
using Register = HeavyOptimizerFrontend::Register;

void HeavyOptimizerFrontend::CompareAndBranch(BranchOpcode opcode,
                                              Register arg1,
                                              Register arg2,
                                              int16_t offset) {
  auto ir = builder_.ir();
  auto cur_bb = builder_.bb();
  MachineBasicBlock* then_bb = ir->NewBasicBlock();
  MachineBasicBlock* else_bb = ir->NewBasicBlock();
  ir->AddEdge(cur_bb, then_bb);
  ir->AddEdge(cur_bb, else_bb);

  Gen<x86_64::CmpqRegReg>(arg1, arg2, GetFlagsRegister());
  Gen<PseudoCondBranch>(ToAssemblerCond(opcode), then_bb, else_bb, GetFlagsRegister());

  builder_.StartBasicBlock(then_bb);
  GenJump(pc_ + offset);

  builder_.StartBasicBlock(else_bb);
}

void HeavyOptimizerFrontend::Branch(int32_t offset) {
  is_uncond_branch_ = true;
  GenJump(pc_ + offset);
}

void HeavyOptimizerFrontend::BranchRegister(Register src, int16_t offset) {
  is_uncond_branch_ = true;
  Register target = AllocTempReg();
  Gen<PseudoCopy>(target, src, 8);
  // Avoid the extra insn if unneeded.
  if (offset != 0) {
    Gen<x86_64::AddqRegImm>(target, offset, GetFlagsRegister());
  }
  // TODO(b/232598137) Maybe move this to translation cache?
  Gen<x86_64::AndqRegImm>(target, ~int32_t{1}, GetFlagsRegister());
  ExitRegionIndirect(target);
}

x86_64::Assembler::Condition HeavyOptimizerFrontend::ToAssemblerCond(BranchOpcode opcode) {
  switch (opcode) {
    case BranchOpcode::kBeq:
      return x86_64::Assembler::Condition::kEqual;
    case BranchOpcode::kBne:
      return x86_64::Assembler::Condition::kNotEqual;
    case BranchOpcode::kBlt:
      return x86_64::Assembler::Condition::kLess;
    case BranchOpcode::kBge:
      return x86_64::Assembler::Condition::kGreaterEqual;
    case BranchOpcode::kBltu:
      return x86_64::Assembler::Condition::kBelow;
    case BranchOpcode::kBgeu:
      return x86_64::Assembler::Condition::kAboveEqual;
  }
}

Register HeavyOptimizerFrontend::GetImm(uint64_t imm) {
  Register result = AllocTempReg();
  Gen<x86_64::MovqRegImm>(result, imm);
  return result;
}

Register HeavyOptimizerFrontend::AllocTempReg() {
  return builder_.ir()->AllocVReg();
}

SimdReg HeavyOptimizerFrontend::AllocTempSimdReg() {
  return SimdReg{builder_.ir()->AllocVReg()};
}

void HeavyOptimizerFrontend::GenJump(GuestAddr target) {
  auto map_it = branch_targets_.find(target);
  if (map_it == branch_targets_.end()) {
    // Remember that this address was taken to help region formation. If we
    // translate it later the data will be overwritten with the actual location.
    branch_targets_[target] = MachineInsnPosition{};
  }

  // Checking pending signals only on back jumps guarantees no infinite loops
  // without pending signal checks.
  auto kind = target <= GetInsnAddr() ? PseudoJump::Kind::kJumpWithPendingSignalsCheck
                                      : PseudoJump::Kind::kJumpWithoutPendingSignalsCheck;

  Gen<PseudoJump>(target, kind);
}

void HeavyOptimizerFrontend::ExitGeneratedCode(GuestAddr target) {
  Gen<PseudoJump>(target, PseudoJump::Kind::kExitGeneratedCode);
}

void HeavyOptimizerFrontend::ExitRegionIndirect(Register target) {
  Gen<PseudoIndirectJump>(target);
}
void HeavyOptimizerFrontend::Unimplemented() {
  success_ = false;
  ExitGeneratedCode(GetInsnAddr());
  // We don't require region to end here as control flow may jump around
  // the undefined instruction, so handle it as an unconditional branch.
  is_uncond_branch_ = true;
}

bool HeavyOptimizerFrontend::IsRegionEndReached() const {
  if (!is_uncond_branch_) {
    return false;
  }

  auto map_it = branch_targets_.find(GetInsnAddr());
  // If this instruction following an unconditional branch isn't reachable by
  // some other branch - it's a region end.
  return map_it == branch_targets_.end();
}

void HeavyOptimizerFrontend::ResolveJumps() {
  if (!config::kLinkJumpsWithinRegion) {
    return;
  }
  auto ir = builder_.ir();

  MachineBasicBlockList bb_list_copy(ir->bb_list());
  for (auto bb : bb_list_copy) {
    if (bb->is_recovery()) {
      // Recovery blocks must exit region, do not try to resolve it into a local branch.
      continue;
    }

    const MachineInsn* last_insn = bb->insn_list().back();
    if (last_insn->opcode() != kMachineOpPseudoJump) {
      continue;
    }

    auto* jump = static_cast<const PseudoJump*>(last_insn);
    if (jump->kind() == PseudoJump::Kind::kSyscall ||
        jump->kind() == PseudoJump::Kind::kExitGeneratedCode) {
      // Syscall or generated code exit must always exit region.
      continue;
    }

    GuestAddr target = jump->target();
    auto map_it = branch_targets_.find(target);
    // All PseudoJump insns must add their targets to branch_targets.
    CHECK(map_it != branch_targets_.end());

    MachineInsnPosition pos = map_it->second;
    MachineBasicBlock* target_containing_bb = pos.first;
    if (!target_containing_bb) {
      // Branch target is not in the current region
      continue;
    }

    CHECK(pos.second.has_value());
    auto target_insn_it = pos.second.value();
    MachineBasicBlock* target_bb;
    if (target_insn_it == target_containing_bb->insn_list().begin()) {
      // We don't need to split if target_insn_it is at the beginning of target_containing_bb.
      target_bb = target_containing_bb;
    } else {
      // target_bb is split from target_containing_bb.
      target_bb = ir->SplitBasicBlock(target_containing_bb, target_insn_it);
      UpdateBranchTargetsAfterSplit(target, target_containing_bb, target_bb);

      // Make sure target_bb is also considered for jump resolution. Otherwise we may leave code
      // referenced by it unlinked from the rest of the IR.
      bb_list_copy.push_back(target_bb);

      // If bb is equal to target_containing_bb, then the branch instruction at the end of bb
      // is moved to the new target_bb, so we replace the instruction at the end of the
      // target_bb instead of bb.
      if (bb == target_containing_bb) {
        bb = target_bb;
      }
    }

    ReplaceJumpWithBranch(bb, target_bb);
  }
}

void HeavyOptimizerFrontend::ReplaceJumpWithBranch(MachineBasicBlock* bb,
                                                   MachineBasicBlock* target_bb) {
  auto ir = builder_.ir();
  const auto* last_insn = bb->insn_list().back();
  CHECK_EQ(last_insn->opcode(), kMachineOpPseudoJump);
  auto* jump = static_cast<const PseudoJump*>(last_insn);
  GuestAddr target = static_cast<const PseudoJump*>(jump)->target();
  // Do not invalidate this iterator as it may be a target for another jump.
  // Instead overwrite the instruction.
  auto jump_it = std::prev(bb->insn_list().end());

  if (jump->kind() == PseudoJump::Kind::kJumpWithoutPendingSignalsCheck) {
    // Simple branch for forward jump.
    *jump_it = ir->NewInsn<PseudoBranch>(target_bb);
    ir->AddEdge(bb, target_bb);
  } else {
    CHECK(jump->kind() == PseudoJump::Kind::kJumpWithPendingSignalsCheck);
    // See EmitCheckSignalsAndMaybeReturn.
    auto* exit_bb = ir->NewBasicBlock();
    // Note that we intentionally don't mark exit_bb as recovery and therefore don't request its
    // reordering away from hot code spots. target_bb is a back branch and is unlikely to be a
    // fall-through jump for the current bb. At the same time exit_bb can be a fall-through jump
    // and benchmarks benefit from it.
    const size_t offset = offsetof(ThreadState, pending_signals_status);
    auto* cmpb = ir->NewInsn<x86_64::CmpbMemBaseDispImm>(
        x86_64::kMachineRegRBP, offset, kPendingSignalsPresent, GetFlagsRegister());
    *jump_it = cmpb;
    auto* cond_branch = ir->NewInsn<PseudoCondBranch>(
        x86_64::Assembler::Condition::kEqual, exit_bb, target_bb, GetFlagsRegister());
    bb->insn_list().push_back(cond_branch);

    builder_.StartBasicBlock(exit_bb);
    ExitGeneratedCode(target);

    ir->AddEdge(bb, exit_bb);
    ir->AddEdge(bb, target_bb);
  }
}

void HeavyOptimizerFrontend::UpdateBranchTargetsAfterSplit(GuestAddr addr,
                                                           const MachineBasicBlock* old_bb,
                                                           MachineBasicBlock* new_bb) {
  auto map_it = branch_targets_.find(addr);
  CHECK(map_it != branch_targets_.end());
  while (map_it != branch_targets_.end() && map_it->second.first == old_bb) {
    map_it->second.first = new_bb;
    map_it++;
  }
}

Register HeavyOptimizerFrontend::GetReg(uint8_t reg) {
  CHECK_LT(reg, kNumGuestRegs);
  Register dst = AllocTempReg();
  builder_.GenGet(dst, reg);
  return dst;
}

void HeavyOptimizerFrontend::SetReg(uint8_t reg, Register value) {
  CHECK_LT(reg, kNumGuestRegs);
  builder_.GenPut(reg, value);
}

FpRegister HeavyOptimizerFrontend::GetFpReg(uint8_t reg) {
  FpRegister result = AllocTempSimdReg();
  builder_.GenGetSimd(result.machine_reg(), reg);
  return result;
}

void HeavyOptimizerFrontend::Nop() {}

Register HeavyOptimizerFrontend::Op(Decoder::OpOpcode opcode, Register arg1, Register arg2) {
  using OpOpcode = Decoder::OpOpcode;
  using Condition = x86_64::Assembler::Condition;
  auto res = AllocTempReg();
  switch (opcode) {
    case OpOpcode::kAdd:
      Gen<PseudoCopy>(res, arg1, 8);
      Gen<x86_64::AddqRegReg>(res, arg2, GetFlagsRegister());
      break;
    case OpOpcode::kSub:
      Gen<PseudoCopy>(res, arg1, 8);
      Gen<x86_64::SubqRegReg>(res, arg2, GetFlagsRegister());
      break;
    case OpOpcode::kAnd:
      Gen<PseudoCopy>(res, arg1, 8);
      Gen<x86_64::AndqRegReg>(res, arg2, GetFlagsRegister());
      break;
    case OpOpcode::kOr:
      Gen<PseudoCopy>(res, arg1, 8);
      Gen<x86_64::OrqRegReg>(res, arg2, GetFlagsRegister());
      break;
    case OpOpcode::kXor:
      Gen<PseudoCopy>(res, arg1, 8);
      Gen<x86_64::XorqRegReg>(res, arg2, GetFlagsRegister());
      break;
    case OpOpcode::kSll:
      Gen<PseudoCopy>(res, arg1, 8);
      Gen<x86_64::ShlqRegReg>(res, arg2, GetFlagsRegister());
      break;
    case OpOpcode::kSrl:
      Gen<PseudoCopy>(res, arg1, 8);
      Gen<x86_64::ShrqRegReg>(res, arg2, GetFlagsRegister());
      break;
    case OpOpcode::kSra:
      Gen<PseudoCopy>(res, arg1, 8);
      Gen<x86_64::SarqRegReg>(res, arg2, GetFlagsRegister());
      break;
    case OpOpcode::kSlt: {
      Gen<x86_64::CmpqRegReg>(arg1, arg2, GetFlagsRegister());
      auto temp = AllocTempReg();
      Gen<x86_64::SetccReg>(Condition::kLess, temp, GetFlagsRegister());
      Gen<x86_64::MovzxbqRegReg>(res, temp);
      break;
    }
    case OpOpcode::kSltu: {
      Gen<x86_64::CmpqRegReg>(arg1, arg2, GetFlagsRegister());
      auto temp = AllocTempReg();
      Gen<x86_64::SetccReg>(Condition::kBelow, temp, GetFlagsRegister());
      Gen<x86_64::MovzxbqRegReg>(res, temp);
      break;
    }
    case OpOpcode::kMul:
      Gen<PseudoCopy>(res, arg1, 8);
      Gen<x86_64::ImulqRegReg>(res, arg2, GetFlagsRegister());
      break;
    case OpOpcode::kMulh: {
      auto rax = AllocTempReg();
      auto rdx = AllocTempReg();
      Gen<PseudoCopy>(rax, arg1, 8);
      Gen<x86_64::ImulqRegRegReg>(rax, rdx, arg2, GetFlagsRegister());
      Gen<PseudoCopy>(res, rdx, 8);
    } break;
    case OpOpcode::kMulhsu: {
      Gen<PseudoCopy>(res, arg1, 8);
      auto rax = AllocTempReg();
      auto rdx = AllocTempReg();
      Gen<PseudoCopy>(rax, arg2, 8);
      Gen<x86_64::MulqRegRegReg>(rax, rdx, res, GetFlagsRegister());
      Gen<x86_64::SarqRegImm>(res, 63, GetFlagsRegister());
      Gen<x86_64::ImulqRegReg>(res, arg2, GetFlagsRegister());
      Gen<x86_64::AddqRegReg>(res, rdx, GetFlagsRegister());
    } break;
    case OpOpcode::kMulhu: {
      auto rax = AllocTempReg();
      auto rdx = AllocTempReg();
      Gen<PseudoCopy>(rax, arg1, 8);
      Gen<x86_64::MulqRegRegReg>(rax, rdx, arg2, GetFlagsRegister());
      Gen<PseudoCopy>(res, rdx, 8);
    } break;
    case OpOpcode::kDiv:
    case OpOpcode::kRem: {
      auto rax = AllocTempReg();
      auto rdx = AllocTempReg();
      Gen<PseudoCopy>(rax, arg1, 8);
      Gen<PseudoCopy>(rdx, rax, 8);
      Gen<x86_64::SarqRegImm>(rdx, 63, GetFlagsRegister());
      Gen<x86_64::IdivqRegRegReg>(rax, rdx, arg2, GetFlagsRegister());
      Gen<PseudoCopy>(res, opcode == OpOpcode::kDiv ? rax : rdx, 8);
    } break;
    case OpOpcode::kDivu:
    case OpOpcode::kRemu: {
      auto rax = AllocTempReg();
      auto rdx = AllocTempReg();
      Gen<PseudoCopy>(rax, arg1, 8);
      // Pseudo-def for use-def operand of XOR to make sure data-flow is integrate.
      Gen<PseudoDefReg>(rdx);
      Gen<x86_64::XorqRegReg>(rdx, rdx, GetFlagsRegister());
      Gen<x86_64::DivqRegRegReg>(rax, rdx, arg2, GetFlagsRegister());
      Gen<PseudoCopy>(res, opcode == OpOpcode::kDivu ? rax : rdx, 8);
    } break;
    case OpOpcode::kAndn:
      if (host_platform::kHasBMI) {
        Gen<x86_64::AndnqRegRegReg>(res, arg2, arg1, GetFlagsRegister());
      } else {
        Gen<PseudoCopy>(res, arg2, 8);
        Gen<x86_64::NotqReg>(res);
        Gen<x86_64::AndqRegReg>(res, arg1, GetFlagsRegister());
      }
      break;
    case OpOpcode::kOrn:
      Gen<PseudoCopy>(res, arg2, 8);
      Gen<x86_64::NotqReg>(res);
      Gen<x86_64::OrqRegReg>(res, arg1, GetFlagsRegister());
      break;
    case OpOpcode::kXnor:
      Gen<PseudoCopy>(res, arg2, 8);
      Gen<x86_64::XorqRegReg>(res, arg1, GetFlagsRegister());
      Gen<x86_64::NotqReg>(res);
      break;
    default:
      Unimplemented();
      return {};
  }

  return res;
}

Register HeavyOptimizerFrontend::Op32(Decoder::Op32Opcode opcode, Register arg1, Register arg2) {
  using Op32Opcode = Decoder::Op32Opcode;
  auto res = AllocTempReg();
  auto unextended_res = res;
  switch (opcode) {
    case Op32Opcode::kAddw:
      Gen<PseudoCopy>(res, arg1, 4);
      Gen<x86_64::AddlRegReg>(res, arg2, GetFlagsRegister());
      break;
    case Op32Opcode::kSubw:
      Gen<PseudoCopy>(res, arg1, 4);
      Gen<x86_64::SublRegReg>(res, arg2, GetFlagsRegister());
      break;
    case Op32Opcode::kSllw:
    case Op32Opcode::kSrlw:
    case Op32Opcode::kSraw: {
      auto rcx = AllocTempReg();
      Gen<PseudoCopy>(res, arg1, 4);
      Gen<PseudoCopy>(rcx, arg2, 4);
      if (opcode == Op32Opcode::kSllw) {
        Gen<x86_64::ShllRegReg>(res, rcx, GetFlagsRegister());
      } else if (opcode == Op32Opcode::kSrlw) {
        Gen<x86_64::ShrlRegReg>(res, rcx, GetFlagsRegister());
      } else {
        Gen<x86_64::SarlRegReg>(res, rcx, GetFlagsRegister());
      }
    } break;
    case Op32Opcode::kMulw:
      Gen<PseudoCopy>(res, arg1, 4);
      Gen<x86_64::ImullRegReg>(res, arg2, GetFlagsRegister());
      break;
    case Op32Opcode::kDivw:
    case Op32Opcode::kRemw: {
      auto rax = AllocTempReg();
      auto rdx = AllocTempReg();
      Gen<PseudoCopy>(rax, arg1, 4);
      Gen<PseudoCopy>(rdx, rax, 4);
      Gen<x86_64::SarlRegImm>(rdx, int8_t{31}, GetFlagsRegister());
      Gen<x86_64::IdivlRegRegReg>(rax, rdx, arg2, GetFlagsRegister());
      unextended_res = opcode == Op32Opcode::kDivw ? rax : rdx;
    } break;
    case Op32Opcode::kDivuw:
    case Op32Opcode::kRemuw: {
      auto rax = AllocTempReg();
      auto rdx = AllocTempReg();
      Gen<PseudoCopy>(rax, arg1, 4);
      // Pseudo-def for use-def operand of XOR to make sure data-flow is integrate.
      Gen<PseudoDefReg>(rdx);
      Gen<x86_64::XorlRegReg>(rdx, rdx, GetFlagsRegister());
      Gen<x86_64::DivlRegRegReg>(rax, rdx, arg2, GetFlagsRegister());
      unextended_res = opcode == Op32Opcode::kDivuw ? rax : rdx;
    } break;
    default:
      Unimplemented();
      return {};
  }
  Gen<x86_64::MovsxlqRegReg>(res, unextended_res);
  return res;
}

//
//  Methods that are not part of SemanticsListener implementation.
//
void HeavyOptimizerFrontend::StartInsn() {
  if (is_uncond_branch_) {
    auto* ir = builder_.ir();
    builder_.StartBasicBlock(ir->NewBasicBlock());
  }

  is_uncond_branch_ = false;
  // The iterators in branch_targets are the last iterators before generating an insn.
  // We advance iterators by one step in Finalize(), as we'll use it to iterate
  // the sub-list of instructions starting from the first one for the given
  // guest address.

  // If a basic block is empty before generating insn, an empty optional typed
  // value is returned. We will resolve it to the first insn of the basic block
  // in Finalize().
  branch_targets_[GetInsnAddr()] = builder_.GetMachineInsnPosition();
}

void HeavyOptimizerFrontend::Finalize(GuestAddr stop_pc) {
  // Make sure the last basic block isn't empty before fixing iterators in
  // branch_targets.
  if (builder_.bb()->insn_list().empty() ||
      !builder_.ir()->IsControlTransfer(builder_.bb()->insn_list().back())) {
    GenJump(stop_pc);
  }

  // This loop advances the iterators in the branch_targets by one. Because in
  // StartInsn(), we saved the iterator to the last insn before we generate the
  // first insn for each guest address. If an insn is saved as an empty optional,
  // then the basic block is empty before we generate the first insn for the
  // guest address. So we resolve it to the first insn in the basic block.
  for (auto& [unused_address, insn_pos] : branch_targets_) {
    auto& [bb, insn_it] = insn_pos;
    if (!bb) {
      // Branch target is not in the current region.
      continue;
    }

    if (insn_it.has_value()) {
      insn_it.value()++;
    } else {
      // Make sure bb isn't still empty.
      CHECK(!bb->insn_list().empty());
      insn_it = bb->insn_list().begin();
    }
  }

  ResolveJumps();
}

}  // namespace berberis