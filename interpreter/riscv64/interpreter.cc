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
      : state_(state) {}

  //
  // Instruction implementations.
  //

  Register Op(Decoder::OpOpcode opcode, Register arg1, Register arg2) {
    switch (opcode) {
      case Decoder::OpOpcode::kAdd:
        return arg1 + arg2;
      default:
        Unimplemented();
        break;
    }
  }

  void Unimplemented() {
    LOG_ALWAYS_FATAL("Unimplemented riscv64 instruction");
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
      state_->cpu.insn_addr += insn_len;
  }

 private:
  void CheckRegIsValid(uint8_t reg) const {
    CHECK_GT(reg, 0u);
    CHECK_LE(reg, arraysize(state_->cpu.x));
  }

  ThreadState* state_;
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
