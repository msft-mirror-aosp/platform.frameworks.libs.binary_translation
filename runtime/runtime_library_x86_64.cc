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

#include "berberis/runtime_primitives/runtime_library.h"

#include "berberis/base/checks.h"
#include "berberis/guest_state/guest_state.h"

// Perform all the steps needed to exit generated code except return, which is
// up to the users of this macro. The users of this macro may choose to perform
// a sibling call as necessary.
// clang-format off
#define END_GENERATED_CODE(EXIT_INSN)                                   \
  asm(                                                                  \
      /* Sync insn_addr. */                                             \
      "mov %%rax, %[InsnAddr](%%rbp)\n"                                 \
      /* Set kOutsideGeneratedCode residence. */                        \
      "movb %[OutsideGeneratedCode], %[Residence](%%rbp)\n"             \
                                                                        \
      /* Set %rdi to the pointer to the guest state so that             \
       * we can perform a sibling call to functions like                \
       * berberis_HandleNotTranslated.                                  \
       */                                                               \
      "mov %%rbp, %%rdi\n"                                              \
                                                                        \
      /* Restore stack */                                               \
      "add %[FrameSizeAtTranslatedCode], %%rsp\n"                       \
                                                                        \
      /* Epilogue */                                                    \
      "pop %%r15\n"                                                     \
      "pop %%r14\n"                                                     \
      "pop %%r13\n"                                                     \
      "pop %%r12\n"                                                     \
      "pop %%rbx\n"                                                     \
      "pop %%rbp\n"                                                     \
      EXIT_INSN                                                        \
      ::[InsnAddr] "p"(offsetof(berberis::ThreadState, cpu.insn_addr)), \
      [Residence] "p"(offsetof(berberis::ThreadState, residence)),      \
      [OutsideGeneratedCode] "J"(berberis::kOutsideGeneratedCode),      \
      [FrameSizeAtTranslatedCode] "J"(8))
// TODO(b/278926583): create and use
// berberis::config::kFrameSizeAtTranslatedCode instead of "8"
// clang-format on

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
//   x86_64 calling conventions, with some items passed in registers.
//
// Entry and exits
// ---------------
//
// Upon entry into generated code and trampoline adapters, we must have:
//
// - %rbp pointing to CPUState,
//
// - every field in CPUState up to date, except insn_addr, and
//
// - %rax containing up-to-date value for potentially stale CPUState::insn_addr.
//
// Since we jump among generated code and trampolines, each region must adhere
// to the "calling conventions" above as it exits.
//
// Each region is allowed to use the stack pointed to by %rsp. However, it must
// restore %rsp before exiting.
//
// %rbx, %rbp, and %r12-%r15 are callee saved, all other registers are
// "caller saved". That is, regions are allowed to use them without restoring
// their original values.
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
// defined in this file. The macro ensures that CPUState is fully up to date,
// including insn_addr, before transferring control back to the Berberis
// runtime.

extern "C" {

// ATTENTION: this symbol gets called directly, without PLT. To keep text
// sharable we should prevent preemption of this symbol, so do not export it!
// TODO(b/232598137): may be set default visibility to protected instead?
__attribute__((__visibility__("hidden"))) void berberis_HandleNoExec(ThreadState* /* state */) {
  // TODO(b/278926583): Add implementation.
  FATAL("berberis_HandleNoExec not yet implemented");
}

// ATTENTION: this symbol gets called directly, without PLT. To keep text
// sharable we should prevent preemption of this symbol, so do not export it!
// TODO(b/232598137): may be set default visibility to protected instead?
__attribute__((__visibility__("hidden"))) void berberis_HandleNotTranslated(
    ThreadState* /* state */) {
  // TODO(b/278926583): move to translator.
  FATAL("berberis_HandleNotTranslated not yet implemented");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Stop() {
  END_GENERATED_CODE("ret");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_NoExec() {
  END_GENERATED_CODE("jmp berberis_HandleNoExec");
  // void berberis_HandleNoExec(ThreadState*);
  // Perform a sibling call to berberis_HandleNoExec. The only parameter
  // is state which is saved in %rdi by END_GENERATED_CODE. We could call the
  // function here instead of jumping to it, but it would be more work to do
  // so because we would have to align the stack and issue the "ret"
  // instruction after the call.
  // TODO(b/232598137): Remove state from HandleNoExec parameters. Get it from
  // the guest thread instead.
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_NotTranslated() {
  END_GENERATED_CODE("jmp berberis_HandleNotTranslated");
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
