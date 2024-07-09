/*
 * Copyright (C) 2024 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file excenaupt in compliance with the License.
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
#include <cstdlib>

#include "berberis/decoder/riscv64/decoder.h"
#include "berberis/decoder/riscv64/semantics_player_arm64.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

class Interpreter {
 public:
  using Decoder = Decoder<SemanticsPlayer<Interpreter>>;
  using Register = uint64_t;

  explicit Interpreter(ProcessState* state) : state_(state) {}

  //
  // Instruction implementations.
  //

  Register Op(Decoder::OpOpcode opcode, Register arg1, Register arg2) {
    switch (opcode) {
      case Decoder::OpOpcode::kAdd:
        return arg1 + arg2;
      default:
        Undefined();
        return {};
    }
  }

  void Undefined() {
    // If there is a guest handler registered for SIGILL we'll delay its processing until the next
    // sync point (likely the main dispatching loop) due to enabled pending signals. Thus we must
    // ensure that insn_addr isn't automatically advanced in FinalizeInsn.
    exception_raised_ = true;
    abort();
  }

  void Unimplemented() {
    // TODO(b/265372622): Replace with fatal from logging.h.
    abort();
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

  void FinalizeInsn(uint8_t insn_len) { state_->cpu.insn_addr += insn_len; }

  [[nodiscard]] GuestAddr GetInsnAddr() const { return state_->cpu.insn_addr; }

 private:
  void CheckRegIsValid(uint8_t reg) const {
    // TODO(b/265372622): Replace with checks from logging.h.
    if (reg == 0 || (reg > sizeof(state_->cpu.x) / sizeof(state_->cpu.x[0]))) {
      abort();
    }
  }

  ProcessState* state_;
  bool exception_raised_;
};

}  // namespace berberis
