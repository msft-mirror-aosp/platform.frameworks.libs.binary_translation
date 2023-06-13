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

#include <cstdint>
#include <limits>
#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/checks.h"
#include "berberis/code_gen_lib/code_gen_lib.h"
#include "berberis/decoder/riscv64/decoder.h"
#include "berberis/decoder/riscv64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"

#include "riscv64_to_x86_64/lite_translator.h"

namespace berberis {

namespace {

void Finalize(LiteTranslator* translator, GuestAddr pc) {
  EmitDirectDispatch(translator->as(), pc, /* check_pending_signals */ true);
  translator->as()->Finalize();
}

// Returns the success status and
// - in case of success, the pc of the next instruction past the translated region
// - in case of failure, the pc of the failed instruction
// Specifically, returnes input pc if we cannot translate even the first instruction.
std::tuple<bool, GuestAddr> TryLiteTranslateRegionImpl(GuestAddr start_pc,
                                                       GuestAddr end_pc,
                                                       MachineCode* machine_code) {
  CHECK_LT(start_pc, end_pc);
  LiteTranslator translator(machine_code, start_pc);
  SemanticsPlayer sem_player(&translator);
  Decoder decoder(&sem_player);

  uint8_t insn_size = decoder.Decode(ToHostAddr<const uint16_t>(start_pc));

  Finalize(&translator, start_pc + insn_size);

  return {translator.success(), start_pc + insn_size};
}

}  // namespace

bool LiteTranslateRange(GuestAddr start_pc, GuestAddr end_pc, MachineCode* machine_code) {
  auto [success, stop_pc] = TryLiteTranslateRegionImpl(start_pc, end_pc, machine_code);
  return success;
}

std::tuple<bool, GuestAddr> TryLiteTranslateRegion(GuestAddr start_pc, MachineCode* machine_code) {
  // This effectively makes translating code at max guest address impossible, but we
  // assume that it's not practically significant.
  return TryLiteTranslateRegionImpl(start_pc, std::numeric_limits<GuestAddr>::max(), machine_code);
}

}  // namespace berberis
