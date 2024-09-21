/*
 * Copyright (C) 2024 The Android Open Source Project
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

#include "berberis/runtime_primitives/runtime_library.h"

#include "berberis/runtime_primitives/code_pool.h"

#if defined(__x86_64__)
#include "berberis/assembler/machine_code.h"
#include "berberis/assembler/x86_64.h"
#endif

namespace berberis {

HostCode kEntryInterpret;
HostCode kEntryExitGeneratedCode;
HostCode kEntryStop;
HostCode kEntryNoExec;
HostCode kEntryNotTranslated;
HostCode kEntryTranslating;
HostCode kEntryInvalidating;
HostCode kEntryWrapping;

namespace {
#if defined(__x86_64__)
// This function installs a trampoline in the CodePool address space.
// This needed to ensure that all entries in the translation cache
// are always pointing to the memory allocated via CodePool.
HostCode InstallEntryTrampoline(HostCode target_function_ptr) {
  MachineCode mc;
  x86_64::Assembler as(&mc);
  as.Jmp(target_function_ptr);
  as.Finalize();
  return GetDefaultCodePoolInstance()->Add(&mc);
}
#endif
}  // namespace

void InitHostEntries() {
#if defined(__x86_64__)
  kEntryInterpret = InstallEntryTrampoline(AsHostCode(berberis_entry_Interpret));
  kEntryExitGeneratedCode = InstallEntryTrampoline(AsHostCode(berberis_entry_ExitGeneratedCode));
  kEntryStop = InstallEntryTrampoline(AsHostCode(berberis_entry_Stop));
  kEntryNoExec = InstallEntryTrampoline(AsHostCode(berberis_entry_NoExec));
  kEntryNotTranslated = InstallEntryTrampoline(AsHostCode(berberis_entry_NotTranslated));
  kEntryTranslating = InstallEntryTrampoline(AsHostCode(berberis_entry_Translating));
  kEntryInvalidating = InstallEntryTrampoline(AsHostCode(berberis_entry_Invalidating));
  kEntryWrapping = InstallEntryTrampoline(AsHostCode(berberis_entry_Wrapping));
#else
  kEntryInterpret = AsHostCode(berberis_entry_Interpret);
  kEntryExitGeneratedCode = AsHostCode(berberis_entry_ExitGeneratedCode);
  kEntryStop = AsHostCode(berberis_entry_Stop);
  kEntryNoExec = AsHostCode(berberis_entry_NoExec);
  kEntryNotTranslated = AsHostCode(berberis_entry_NotTranslated);
  kEntryTranslating = AsHostCode(berberis_entry_Translating);
  kEntryInvalidating = AsHostCode(berberis_entry_Invalidating);
  kEntryWrapping = AsHostCode(berberis_entry_Wrapping);
#endif
}

}  // namespace berberis
