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

#include "berberis/runtime/translator.h"

#include <cstdlib>
#include <tuple>

#include "berberis/base/config_globals.h"
#include "berberis/base/logging.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/interpreter/riscv64/interpreter.h"
#include "berberis/runtime_primitives/host_call_frame.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/runtime_primitives/translation_cache.h"

namespace berberis {

namespace {

// Syntax sugar.
GuestCodeEntry::Kind kSpecialHandler = GuestCodeEntry::Kind::kSpecialHandler;
GuestCodeEntry::Kind kInterpreted = GuestCodeEntry::Kind::kInterpreted;

enum class TranslationMode { kInterpretOnly, kNumModes };

TranslationMode g_translation_mode = TranslationMode::kInterpretOnly;

void UpdateTranslationMode() {
  // Indices must match TranslationMode enum.
  constexpr const char* kTranslationModeNames[] = {
      "interpret-only",
  };
  static_assert(static_cast<int>(TranslationMode::kNumModes) ==
                sizeof(kTranslationModeNames) / sizeof(char*));

  const char* config_mode = GetTranslationModeConfig();
  if (!config_mode) {
    return;
  }

  for (int i = 0; i < static_cast<int>(TranslationMode::kNumModes); i++) {
    if (0 == strcmp(config_mode, kTranslationModeNames[i])) {
      g_translation_mode = TranslationMode(i);
      TRACE("translation mode is manually set to '%s'", config_mode);
      return;
    }
  }

  LOG_ALWAYS_FATAL("Unrecognized translation mode '%s'", config_mode);
}

// Use aligned address of this variable as the default stop address for guest execution.
// It should never coincide with any guest address or address of a wrapped host symbol.
// Unwinder might examine nearby insns.
alignas(4) uint32_t g_native_bridge_call_guest[] = {
    // <native_bridge_call_guest>:
    0xd503201f,  // nop
    0xd503201f,  // nop  <--
    0xd503201f,  // nop
};

}  // namespace

void InitTranslator() {
  UpdateTranslationMode();
  InitHostCallFrameGuestPC(ToGuestAddr(g_native_bridge_call_guest + 1));
  // TODO(b/232598137) Setup recovery for interpreter then init here.
  // InitInterpreter();
}

void TranslateRegion(GuestAddr pc) {
  TranslationCache* cache = TranslationCache::GetInstance();

  GuestCodeEntry* entry;
  entry = cache->AddAndLockForTranslation(pc, 0);
  if (!entry) {
    return;
  }

  GuestMapShadow* guest_map_shadow = GuestMapShadow::GetInstance();

  // Check if 1st insn is in executable memory (we don't know yet how many instructions will be able
  // to be translated).
  if (!guest_map_shadow->IsExecutable(pc, 4)) {
    cache->SetTranslatedAndUnlock(pc, entry, 4, kSpecialHandler, {kEntryNoExec, 0});
    return;
  }

  HostCodePiece host_code_piece;
  size_t size;
  GuestCodeEntry::Kind kind;
  if (g_translation_mode == TranslationMode::kInterpretOnly) {
    std::tie(host_code_piece, size, kind) =
        std::make_tuple(HostCodePiece{kEntryInterpret, 0}, 4, kInterpreted);
  } else {
    LOG_ALWAYS_FATAL("Unsupported translation mode %u", g_translation_mode);
  }

  // Now that we know the size of the translated block, make sure the entire memory block has
  // executable permission before saving it to the cache.
  // TODO(b/232598137): installing kEntryNoExec for the *current* pc is completely incorrect as
  // we've checked that it's executable above. The straightforward thing to do would be to
  // check executability of each instruction while translating, and generating signal raise
  // for non-executable ones. This handles the case when region contains conditional branch
  // to non-executable code.
  if (!guest_map_shadow->IsExecutable(pc, size)) {
    TRACE("setting partly executable region at [0x%zx, 0x%zx) as not executable!", pc, pc + size);
    cache->SetTranslatedAndUnlock(pc, entry, size, kSpecialHandler, {kEntryNoExec, 0});
    return;
  }

  cache->SetTranslatedAndUnlock(pc, entry, size, kind, host_code_piece);
}

// A wrapper to export a template function.
void TranslateRegionAtFirstGear(GuestAddr pc) {
  TranslateRegion(pc);
}

// ATTENTION: This symbol gets called directly, without PLT. To keep text
// sharable we should prevent preemption of this symbol, so do not export it!
// TODO(b/232598137): may be set default visibility to protected instead?
extern "C" __attribute__((__visibility__("hidden"))) void berberis_HandleNotTranslated(
    ThreadState* state) {
  if (g_translation_mode == TranslationMode::kInterpretOnly) {
    InterpretInsn(state);
    return;
  }
  TranslateRegion(state->cpu.insn_addr);
}

extern "C" __attribute__((__visibility__("hidden"))) void berberis_HandleInterpret(
    ThreadState* state) {
  InterpretInsn(state);
}

extern "C" __attribute__((__visibility__("hidden"))) const void* berberis_GetDispatchAddress(
    ThreadState* state) {
  if (ArePendingSignalsPresent(state)) {
    return kEntryExitGeneratedCode;
  }
  return TranslationCache::GetInstance()->GetHostCodePtr(state->cpu.insn_addr)->load();
}

}  // namespace berberis
