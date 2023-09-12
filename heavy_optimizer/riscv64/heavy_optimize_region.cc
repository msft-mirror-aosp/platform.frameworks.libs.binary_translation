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

#include "berberis/heavy_optimizer/riscv64/heavy_optimize_region.h"

#include <cstdint>
#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/backend/x86_64/code_gen.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/base/config_globals.h"
#include "berberis/base/tracing.h"
#include "berberis/decoder/riscv64/decoder.h"
#include "berberis/decoder/riscv64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"

#include "frontend.h"

namespace berberis {

std::tuple<GuestAddr, size_t> HeavyOptimizeRegion(GuestAddr pc, MachineCode* machine_code) {
  Arena arena;
  x86_64::MachineIR machine_ir(&arena);
  HeavyOptimizerFrontend frontend(&machine_ir, pc);
  SemanticsPlayer sem_player(&frontend);
  Decoder decoder(&sem_player);
  size_t number_of_instructions = 0;

  while (!frontend.IsRegionEndReached()) {
    // TODO(b/291126189) start insn
    // TODO(b/291126189) decode insn, pass size to incr below.
    frontend.IncrementInsnAddr(4);
    number_of_instructions++;
  }

  auto stop_pc = frontend.GetInsnAddr();
  // TODO(b/291126189) finalize insn

  if (IsConfigFlagSet(kVerboseTranslation)) {
    // Trace only after all the potential failure points.
    TRACE("Heavy optimizing 0x%lx (%lu bytes)", pc, stop_pc - pc);
  }

  x86_64::GenCode(&machine_ir, machine_code);

  return {stop_pc, number_of_instructions};
}

}  // namespace berberis
