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
      : pc_(pc), builder_(machine_ir) {}

  bool IsRegionEndReached() const;

  [[nodiscard]] GuestAddr GetInsnAddr() const { return pc_; }
  void IncrementInsnAddr(uint8_t insn_size) { pc_ += insn_size; }

 private:
  GuestAddr pc_;
  x86_64::MachineIRBuilder builder_;
};

}  // namespace berberis

#endif /* BERBERIS_HEAVY_OPTIMIZER_RISCV64_FRONTEND_H_ */
