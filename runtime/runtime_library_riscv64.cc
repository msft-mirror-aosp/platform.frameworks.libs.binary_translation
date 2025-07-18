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

#include "berberis/base/config.h"
#include "berberis/guest_state/guest_state.h"

extern "C" void berberis_HandleNotTranslated(berberis::ThreadState* state);
extern "C" void berberis_GetDispatchAddress(berberis::ThreadState* state);
extern "C" void berberis_HandleInterpret(berberis::ThreadState* state);

// Helpers ensure that the functions below are available in PLT.
__attribute__((used, __visibility__("hidden"))) extern "C" void helper_NotTranslated(
    berberis::ThreadState* state) {
  berberis_HandleNotTranslated(state);
}

__attribute__((used, __visibility__("hidden"))) extern "C" void helper_GetDispatchAddress(
    berberis::ThreadState* state) {
  berberis_GetDispatchAddress(state);
}

__attribute__((used, __visibility__("hidden"))) extern "C" void helper_HandleInterpret(
    berberis::ThreadState* state) {
  berberis_HandleInterpret(state);
}

// Perform all the steps needed to exit generated code except return, which is
// up to the users of this macro. The users of this macro may choose to perform
// a sibling call as necessary.
// clang-format off
#define END_GENERATED_CODE(EXIT_INSN)                                   \
  asm(                                                                  \
      /* Sync insn_addr. */                                             \
      "sd s11, %[InsnAddr](fp)\n"                                       \
      /* Set kOutsideGeneratedCode residence. */                        \
      "li t1, %[OutsideGeneratedCode]\n"                                \
      "sb t1, %[Residence](fp)\n"                                       \
                                                                        \
      /* Set a0 to the pointer to the guest state so that               \
       * we can perform a sibling call to functions like                \
       * berberis_HandleNotTranslated.                                  \
       */                                                               \
      "mv a0, fp\n"                                                     \
                                                                        \
      /* Epilogue */                                                    \
      "ld ra, 96(sp)\n"                                                 \
      "ld fp, 88(sp)\n"                                                 \
      "ld s1, 80(sp)\n"                                                 \
      "ld s2, 72(sp)\n"                                                 \
      "ld s3, 64(sp)\n"                                                 \
      "ld s4, 56(sp)\n"                                                 \
      "ld s5, 48(sp)\n"                                                 \
      "ld s6, 40(sp)\n"                                                 \
      "ld s7, 32(sp)\n"                                                 \
      "ld s8, 24(sp)\n"                                                 \
      "ld s9, 16(sp)\n"                                                 \
      "ld s10, 8(sp)\n"                                                 \
      "ld s11, 0(sp)\n"                                                 \
      "addi sp, sp, 112\n"                                              \
                                                                        \
      EXIT_INSN                                                         \
      ::[InsnAddr] "I"(offsetof(berberis::ThreadState, cpu.insn_addr)), \
      [Residence] "I"(offsetof(berberis::ThreadState, residence)),      \
      [OutsideGeneratedCode] "I"(berberis::kOutsideGeneratedCode))
// clang-format on

namespace berberis {

extern "C" {

[[gnu::naked]] [[gnu::noinline]] void berberis_RunGeneratedCode(ThreadState* state, HostCode code) {
  // Parameters are in a0 - state and a1 - code.
  // Instruction address is saved in s11. This is also the last register to be allocated within a
  // region. This approach maximizes the chance of s11 being not clobbered and thus facilitates
  // debugging.
  //
  // On riscv64 Linux, stack should be aligned on 16 at every call insn.
  // That means stack is always 0 mod 16 on function entry.
  // See https://riscv.org/wp-content/uploads/2015/01/riscv-calling.pdf (18.2).
  //
  // We are saving all general purpose callee saved registers.
  // TODO(b/352784623): Save fp registers when we start using them.

  // clang-format off
  asm(
    // Prologue
      "addi sp, sp, -112\n"
      "sd s11, 0(sp)\n"
      "sd s10, 8(sp)\n"
      "sd s9, 16(sp)\n"
      "sd s8, 24(sp)\n"
      "sd s7, 32(sp)\n"
      "sd s6, 40(sp)\n"
      "sd s5, 48(sp)\n"
      "sd s4, 56(sp)\n"
      "sd s3, 64(sp)\n"
      "sd s2, 72(sp)\n"
      "sd s1, 80(sp)\n"
      "sd fp, 88(sp)\n"
      "sd ra, 96(sp)\n"

      // Set state pointer.
      "mv fp, a0\n"  // kStateRegister, kOmitFramePointer

      // Set insn_addr.
      "ld s11, %[InsnAddr](fp)\n"
      // Set kInsideGeneratedCode residence.
      "li t1, %[InsideGeneratedCode]\n"
      "sb t1, %[Residence](fp)\n"

      // Jump to entry.
      "jr a1\n"
      ::[InsnAddr] "I"(offsetof(ThreadState, cpu.insn_addr)),
  [Residence] "I"(offsetof(ThreadState, residence)),
  [InsideGeneratedCode] "I"(kInsideGeneratedCode));
  // clang-format on
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Interpret() {
  // clang-format off
  asm(
    //Sync insn_addr.
      "sd s11, %[InsnAddr](fp)\n"
      // Set kOutsideGeneratedCode residence.
      "li t0, %[OutsideGeneratedCode]\n"
      "sb t0, %[Residence](fp)\n"

      // fp holds the pointer to state which is the argument to the call.
      "mv a0, fp\n"
      "call berberis_HandleInterpret@plt\n"

      // a0 may be clobbered by the call abobe, so init it again.
      "mv a0, fp\n"
      "call berberis_GetDispatchAddress@plt\n"
      "mv t1, a0\n"

      // Set insn_addr.
      "ld s11, %[InsnAddr](fp)\n"
      // Set kInsideGeneratedCode residence.
      "li t0, %[InsideGeneratedCode]\n"
      "sb t0, %[Residence](fp)\n"

      "jr t1\n"
      ::[InsnAddr] "I"(offsetof(berberis::ThreadState, cpu.insn_addr)),
  [Residence] "I"(offsetof(berberis::ThreadState, residence)),
  [OutsideGeneratedCode] "I"(berberis::kOutsideGeneratedCode),
  [InsideGeneratedCode] "I"(berberis::kInsideGeneratedCode));
  // clang-format on
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_ExitGeneratedCode() {
  END_GENERATED_CODE("ret");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Stop() {
  END_GENERATED_CODE("ret");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_NoExec() {
  asm("unimp");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_NotTranslated() {
  // @plt is needed since the symbol is dynamically linked.
  END_GENERATED_CODE("tail berberis_HandleNotTranslated@plt");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Translating() {
  asm("unimp");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Invalidating() {
  asm("unimp");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Wrapping() {
  asm("unimp");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_HandleLiteCounterThresholdReached() {
  asm("unimp");
}

}  // extern "C"

}  // namespace berberis
