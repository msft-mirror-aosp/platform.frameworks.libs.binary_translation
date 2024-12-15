/*
 * Copyright (C) 2019 The Android Open Source Project
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

#include "berberis/code_gen_lib/gen_wrapper.h"

#include "berberis/assembler/machine_code.h"
#include "berberis/assembler/x86_32.h"
#include "berberis/base/bit_util.h"
#include "berberis/base/logging.h"
#include "berberis/guest_abi/guest_arguments.h"
#include "berberis/guest_abi/guest_call.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

using x86_32::Assembler;

void GenWrapGuestFunction(MachineCode* mc,
                          GuestAddr pc,
                          const char* signature,
                          HostCode guest_runner,
                          const char* name) {
  UNUSED(name);

  // Stack frame
  // -----------
  // esp, aligned on 16             -> [argument 0: pc]
  //                                   [argument 1: guest argument buffer addr]
  // aligned on 4                   -> [guest argument buffer]
  //                                   [...]
  // esp after prologue             -> [saved ebp]
  // esp after call                 -> [return addr]
  // esp before call, aligned on 16 -> [parameter 0]
  //                                   [...]

  Assembler as(mc);

  // On function entry, esp + 4 is a multiple of 16.
  // Right before next function call, esp is a multiple of 16.

  // Default prologue.
  as.Push(Assembler::ebp);
  as.Movl(Assembler::ebp, Assembler::esp);

  static_assert(alignof(GuestArgumentBuffer) <= 4, "unexpected GuestArgumentBuffer alignment");

  // Estimate guest argument buffer size.
  // Each argument can be 2 4-bytes at most. Result can be 2 4-bytes at most.
  // First 4-byte is in the GuestArgumentBuffer.
  // TODO(eaeltsin): maybe run parameter passing to calculate exactly?
  size_t max_argv_size = strlen(signature) * 8;
  size_t guest_argument_buffer_size = sizeof(GuestArgumentBuffer) - 4 + max_argv_size;

  // Stack frame size is guest argument buffer + 2 4-bytes for guest runner arguments.
  size_t frame_size = guest_argument_buffer_size + 8;

  // Curr esp + 8 is a multiple of 16.
  // New esp is a multiple of 16.
  size_t aligned_frame_size = AlignUp(frame_size + 8, 16) - 8;

  // Allocate stack frame.
  as.Subl(Assembler::esp, aligned_frame_size);

  constexpr int kArgcOffset = 8 + offsetof(GuestArgumentBuffer, argc);
  constexpr int kRescOffset = 8 + offsetof(GuestArgumentBuffer, resc);
  constexpr int kArgvOffset = 8 + offsetof(GuestArgumentBuffer, argv);

  const int params_offset = aligned_frame_size + 8;

  // Convert parameters and set argc.
  int host_argc = 0;
  int argc = 0;
  for (size_t i = 1; signature[i] != '\0'; ++i) {
    if (signature[i] == 'z' || signature[i] == 'b' || signature[i] == 's' || signature[i] == 'c' ||
        signature[i] == 'i' || signature[i] == 'p' || signature[i] == 'f') {
      as.Movl(Assembler::eax, {.base = Assembler::esp, .disp = params_offset + 4 * host_argc});
      ++host_argc;
      as.Movl({.base = Assembler::esp, .disp = kArgvOffset + 4 * argc}, Assembler::eax);
      ++argc;
    } else if (signature[i] == 'l' || signature[i] == 'd') {
      as.Movl(Assembler::eax, {.base = Assembler::esp, .disp = params_offset + 4 * host_argc});
      as.Movl(Assembler::edx, {.base = Assembler::esp, .disp = params_offset + 4 * host_argc + 4});
      host_argc += 2;
      argc = AlignUp(argc, 2);
      as.Movl({.base = Assembler::esp, .disp = kArgvOffset + 4 * argc}, Assembler::eax);
      as.Movl({.base = Assembler::esp, .disp = kArgvOffset + 4 * argc + 4}, Assembler::edx);
      argc += 2;
    } else {
      FATAL("signature char '%c' not supported", signature[i]);
    }
  }
  as.Movl({.base = Assembler::esp, .disp = kArgcOffset}, argc);

  // Set resc.
  if (signature[0] == 'z' || signature[0] == 'b' || signature[0] == 's' ||
      signature[0] == 'c' | signature[0] == 'i' || signature[0] == 'p' || signature[0] == 'f') {
    as.Movl({.base = Assembler::esp, .disp = kRescOffset}, 1);
  } else if (signature[0] == 'l' || signature[0] == 'd') {
    as.Movl({.base = Assembler::esp, .disp = kRescOffset}, 2);
  } else {
    CHECK_EQ('v', signature[0]);
    as.Movl({.base = Assembler::esp, .disp = kRescOffset}, 0);
  }

  // Call guest runner.
  as.Movl({.base = Assembler::esp, .disp = 0}, pc);
  as.Leal(Assembler::eax, {.base = Assembler::esp, .disp = 8});
  as.Movl({.base = Assembler::esp, .disp = 4}, Assembler::eax);
  as.Call(guest_runner);

  // Get the result.
  if (signature[0] == 'z' || signature[0] == 'b' || signature[0] == 's' ||
      signature[0] == 'c' | signature[0] == 'i' || signature[0] == 'p') {
    as.Movl(Assembler::eax, {.base = Assembler::esp, .disp = kArgvOffset});
  } else if (signature[0] == 'l') {
    as.Movl(Assembler::eax, {.base = Assembler::esp, .disp = kArgvOffset});
    as.Movl(Assembler::edx, {.base = Assembler::esp, .disp = kArgvOffset + 4});
  } else if (signature[0] == 'f') {
    as.Flds({.base = Assembler::esp, .disp = kArgvOffset});
  } else if (signature[0] == 'd') {
    as.Fldl({.base = Assembler::esp, .disp = kArgvOffset});
  }

  // Free stack frame.
  as.Addl(Assembler::esp, aligned_frame_size);

  // Default epilogue.
  as.Pop(Assembler::ebp);
  as.Ret();

  as.Finalize();
}

}  // namespace berberis
