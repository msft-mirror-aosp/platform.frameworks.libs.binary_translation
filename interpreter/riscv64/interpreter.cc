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

#include "berberis/interpreter/riscv64/interpreter.h"

#include <cstdint>
#include <cstring>

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"
#include "berberis/base/logging.h"
#include "berberis/base/macros.h"
#include "berberis/decoder/riscv64/decoder.h"
#include "berberis/decoder/riscv64/semantics_player.h"
#include "berberis/guest_state/guest_state_riscv64.h"

namespace berberis {

namespace {

class Interpreter {
 public:
  using Decoder = Decoder<SemanticsPlayer<Interpreter>>;
  using Register = uint64_t;

  explicit Interpreter(ThreadState* state)
      : state_(state),
        branch_taken_(false) {}

  //
  // Instruction implementations.
  //

  Register Op(Decoder::OpOpcode opcode, Register arg1, Register arg2) {
    switch (opcode) {
      case Decoder::OpOpcode::kAdd:
        return arg1 + arg2;
      case Decoder::OpOpcode::kSub:
        return arg1 - arg2;
      case Decoder::OpOpcode::kAnd:
        return arg1 & arg2;
      case Decoder::OpOpcode::kOr:
        return arg1 | arg2;
      case Decoder::OpOpcode::kXor:
        return arg1 ^ arg2;
      case Decoder::OpOpcode::kSll:
        return arg1 << arg2;
      case Decoder::OpOpcode::kSlr:
        return arg1 >> arg2;
      case Decoder::OpOpcode::kSra:
        return bit_cast<int64_t>(arg1) >> arg2;
      case Decoder::OpOpcode::kSlt:
        return bit_cast<int64_t>(arg1) < bit_cast<int64_t>(arg2) ? 1 : 0;
      case Decoder::OpOpcode::kSltu:
        return arg1 < arg2 ? 1 : 0;
      default:
        Unimplemented();
        break;
    }
  }

  Register Load(Decoder::LoadOpcode opcode, Register arg, uint16_t offset) {
    void* ptr = bit_cast<void*>(arg + offset);
    switch (opcode) {
      case Decoder::LoadOpcode::kLbu:
        return Load<uint8_t>(ptr);
      case Decoder::LoadOpcode::kLhu:
        return Load<uint16_t>(ptr);
      case Decoder::LoadOpcode::kLwu:
        return Load<uint32_t>(ptr);
      case Decoder::LoadOpcode::kLd:
        return Load<uint64_t>(ptr);
      case Decoder::LoadOpcode::kLb:
        return Load<int8_t>(ptr);
      case Decoder::LoadOpcode::kLh:
        return Load<int16_t>(ptr);
      case Decoder::LoadOpcode::kLw:
        return Load<int32_t>(ptr);
    }
  }

  void Store(Decoder::StoreOpcode opcode, Register arg, uint16_t offset, Register data) {
    void* ptr = bit_cast<void*>(arg + offset);
    switch (opcode) {
      case Decoder::StoreOpcode::kSb:
        Store<uint8_t>(ptr, data);
        break;
      case Decoder::StoreOpcode::kSh:
        Store<uint16_t>(ptr, data);
        break;
      case Decoder::StoreOpcode::kSw:
        Store<uint32_t>(ptr, data);
        break;
      case Decoder::StoreOpcode::kSd:
        Store<uint64_t>(ptr, data);
        break;
    }
  }

  void Branch(Decoder::BranchOpcode opcode, Register arg1, Register arg2, int16_t offset) {
    bool cond_value;
    switch (opcode) {
      case Decoder::BranchOpcode::kBeq:
        cond_value = arg1 == arg2;
        break;
      case Decoder::BranchOpcode::kBne:
        cond_value = arg1 != arg2;
        break;
      case Decoder::BranchOpcode::kBltu:
        cond_value = arg1 < arg2;
        break;
      case Decoder::BranchOpcode::kBgeu:
        cond_value = arg1 >= arg2;
        break;
      case Decoder::BranchOpcode::kBlt:
        cond_value = bit_cast<int64_t>(arg1) < bit_cast<int64_t>(arg2);
        break;
      case Decoder::BranchOpcode::kBge:
        cond_value = bit_cast<int64_t>(arg1) >= bit_cast<int64_t>(arg2);
        break;
    }

    if (cond_value) {
      state_->cpu.insn_addr += offset;
      branch_taken_ = true;
    }
  }

  Register JumpAndLink(int32_t offset, uint8_t insn_len) {
    uint64_t pc = state_->cpu.insn_addr;
    state_->cpu.insn_addr += offset;
    branch_taken_ = true;
    return pc + insn_len;
  }

  void Unimplemented() {
    FATAL("Unimplemented riscv64 instruction");
  }

  //
  // Guest state getters/setters.
  //

  uint64_t GetReg(uint8_t reg) const {
    CheckRegIsValid(reg);
    return state_->cpu.x[reg - 1];
  }

  void SetReg(uint8_t reg, Register value) {
    CheckRegIsValid(reg);
    state_->cpu.x[reg - 1] = value;
  }

  //
  // Various helper methods.
  //

  uint64_t GetImm(uint64_t imm) const { return imm; }

  void FinalizeInsn(uint8_t insn_len) {
    if (!branch_taken_) {
      state_->cpu.insn_addr += insn_len;
    }
  }

 private:
  template <typename DataType>
  uint64_t Load(const void * ptr) const {
    DataType data;
    memcpy(&data, ptr, sizeof(data));
    // Signed types automatically sign-extend to int64_t.
    return static_cast<uint64_t>(data);
  }

  template <typename DataType>
  void Store(void* ptr, uint64_t data) const {
    memcpy(ptr, &data, sizeof(DataType));
  }

  void CheckRegIsValid(uint8_t reg) const {
    CHECK_GT(reg, 0u);
    CHECK_LE(reg, arraysize(state_->cpu.x));
  }

  ThreadState* state_;
  bool branch_taken_;
};

}  // namespace

void InterpretInsn(ThreadState* state) {
  GuestAddr pc = state->cpu.insn_addr;

  Interpreter interpreter(state);
  SemanticsPlayer sem_player(&interpreter);
  Decoder decoder(&sem_player);
  uint8_t insn_len = decoder.Decode(bit_cast<const uint16_t*>(pc));
  interpreter.FinalizeInsn(insn_len);
}

}  // namespace berberis
