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

#include <cstdlib>

#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

// "Calling conventions" among generated code and trampolines
// ==========================================================
//
// Introduction
// ------------
//
// To ensure the high performance of our generated code, we employ a couple of
// techniques:
//
// - We allow generated regions to jump among them without transferring control
//   back to Berberis runtime.
//
// - We use custom "calling conventions" that are different from the standard
//   aapcs64 calling conventions, with some items passed in registers.
//
// Entry and exits
// ---------------
//
// Upon entry into generated code and trampoline adapters, we must have:
//
// - x29 pointing to ThreadState,
//
// - every field in ThreadState up to date, except insn_addr, and
//
// - x0 containing up-to-date value for potentially stale ThreadState::insn_addr.
//
// Since we jump among generated code and trampolines, each region must adhere
// to the "calling conventions" above as it exits.
//
// Each region is allowed to use the stack pointed to by sp. However, it must
// restore sp before exiting.
//
// x19-x30 and the lower 64 bits of v8-v15 are callee saved. All other registers,
// and the upper 64 bits of v8-v15, are caller saved. That is, regions are
// allowed to use them without restoring their original values.
//
// Berberis -> generated code
// ---------------------------------
//
// If we are transferring control to generated code and trampolines from the
// Berberis runtime, such as ExecuteGuest, then we must do so via
// berberis_RunGeneratedCode, which is responsible for setting up registers for
// the "calling conventions".
//
// Generated code -> Berberis
// ---------------------------------
//
// When we are exiting generate code, we must do so via END_GENERATED_CODE macro
// defined in this file. The macro ensures that ThreadState is fully up to date,
// including insn_addr, before transferring control back to the Berberis
// runtime.

namespace {

// Number of bytes used for storing callee-saved registers on the stack when
// entering and exiting generated code. There are a total of 20 64-bit
// callee-saved registers.
constexpr size_t kCalleeSavedFrameSize = 8 * 20;

}  // namespace

extern "C" {

// Perform all the steps needed to exit generated code except return, which is
// up to the users of this macro. The users of this macro may choose to perform
// a sibling call as necessary.
// clang-format off
#define END_GENERATED_CODE(EXIT_INSN)                                   \
  asm(                                                                  \
      /* Sync insn_addr. */                                             \
      "str x0, [x29, %[InsnAddr]]\n"                                    \
      /* Set kOutsideGeneratedCode residence. */                        \
      "mov w28, %[OutsideGeneratedCode]\n"                              \
      "strb w28, [x29, %[Residence]]\n"                                 \
                                                                        \
      /* Set x0 to the pointer to the guest state so that               \
       * we can perform a sibling call to functions like                \
       * berberis_HandleNotTranslated.                                  \
       */                                                               \
      "mov x0, x29\n"                                                   \
                                                                        \
      /* Epilogue */                                                    \
      "ldp d15, d14, [sp]\n"                                            \
      "ldp d13, d12, [sp, 16]\n"                                        \
      "ldp d11, d10, [sp, 32]\n"                                        \
      "ldp d9, d8, [sp, 48]\n"                                          \
      "ldp x29, x28, [sp, 64]\n"                                        \
      "ldp x27, x26, [sp, 80]\n"                                        \
      "ldp x25, x24, [sp, 96]\n"                                        \
      "ldp x23, x22, [sp, 112]\n"                                       \
      "ldp x21, x20, [sp, 128]\n"                                       \
      "ldp x19, lr, [sp, 144]\n"                                        \
      "add sp, %[CalleeSavedFrameSize]\n"                               \
                                                                        \
      EXIT_INSN                                                         \
      ::[InsnAddr] "p"(offsetof(berberis::ThreadState, cpu.insn_addr)), \
      [Residence] "p"(offsetof(berberis::ThreadState, residence)),      \
      [OutsideGeneratedCode] "M"(berberis::kOutsideGeneratedCode),      \
      [CalleeSavedFrameSize] "J"(kCalleeSavedFrameSize))
// clang-format on

[[gnu::naked]] [[gnu::noinline]] void berberis_RunGeneratedCode(ThreadState* state, HostCode code) {
  // Parameters are in x0 - state and x1 - code
  //
  // In aapcs64, the stack must be aligned on 16 at every call instruction (sp mod 16 = 0).
  // See https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst (6.4.5.1)

  // clang-format off
  asm(
    // Prologue
    "sub sp, %[CalleeSavedFrameSize]\n"
    "stp x19, lr, [sp, 144]\n"
    "stp x21, x20, [sp, 128]\n"
    "stp x23, x22, [sp, 112]\n"
    "stp x25, x24, [sp, 96]\n"
    "stp x27, x26, [sp, 80]\n"
    "stp x29, x28, [sp, 64]\n"
    "stp d9, d8, [sp, 48]\n"
    "stp d11, d10, [sp, 32]\n"
    "stp d13, d12, [sp, 16]\n"
    "stp d15, d14, [sp]\n"

    // Set state pointer
    "mov x29, x0\n"

    // Set insn_addr.
    "ldr x0, [x29, %[InsnAddr]]\n"
    // Set kInsideGeneratedCode residence.
    "mov w28, %[InsideGeneratedCode]\n"
    "strb w28, [x29, %[Residence]]\n"

    // Jump to entry
    "br x1"
    ::[InsnAddr] "p"(offsetof(ThreadState, cpu.insn_addr)),
    [Residence] "p"(offsetof(ThreadState, residence)),
    [InsideGeneratedCode] "M"(kInsideGeneratedCode),
    [CalleeSavedFrameSize] "J"(kCalleeSavedFrameSize));
  // clang-format on
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Interpret() {
  // clang-format off
  asm(
    // Sync insn_addr.
    "str x0, [x29, %[InsnAddr]]\n"
    // Set kOutsideGeneratedCode residence. */
    "mov w28, %[OutsideGeneratedCode]\n"
    "strb w28, [x29, %[Residence]]\n"

    // x29 holds the pointer to state which is the argument to the call.
    "mov x0, x29\n"
    "bl berberis_HandleInterpret\n"

    // x0 may be clobbered by the call above, so init it again.
    "mov x0, x29\n"
    "bl berberis_GetDispatchAddress\n"
    "mov x1, x0\n"

    // Set insn_addr.
    "ldr x0, [x29, %[InsnAddr]]\n"
    // Set kInsideGeneratedCode residence.
    "mov w28, %[InsideGeneratedCode]\n"
    "strb w28, [x29, %[Residence]]\n"

    "br x1\n"
    ::[InsnAddr] "p"(offsetof(berberis::ThreadState, cpu.insn_addr)),
    [Residence] "p"(offsetof(berberis::ThreadState, residence)),
    [OutsideGeneratedCode] "M"(berberis::kOutsideGeneratedCode),
    [InsideGeneratedCode] "J"(berberis::kInsideGeneratedCode));
  // clang-format on
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_ExitGeneratedCode() {
  END_GENERATED_CODE("ret");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Stop() {
  END_GENERATED_CODE("ret");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_NoExec() {
  END_GENERATED_CODE("b berberis_HandleNoExec");
  // void berberis_HandleNoExec(ThreadState*);
  // Perform a sibling call to berberis_HandleNoExec. The only parameter
  // is state which is saved in x0 by END_GENERATED_CODE.
  // TODO(b/232598137): Remove state from HandleNoExec parameters. Get it from
  // the guest thread instead.
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_NotTranslated() {
  END_GENERATED_CODE("b berberis_HandleNotTranslated");
  // void berberis_HandleNotTranslated(ThreadState*);
  // See the comment above about the sibling call.
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Translating() {
  // TODO(b/232598137): Run interpreter while translation is in progress.
  END_GENERATED_CODE("ret");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Invalidating() {
  // TODO(b/232598137): maybe call sched_yield() here.
  END_GENERATED_CODE("ret");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Wrapping() {
  // TODO(b/232598137): maybe call sched_yield() here.
  END_GENERATED_CODE("ret");
}

}  // extern "C"

}  // namespace berberis
