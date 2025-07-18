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
#include "berberis/assembler/x86_64.h"
#include "berberis/base/bit_util.h"
#include "berberis/base/logging.h"
#include "berberis/guest_abi/guest_arguments.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

using x86_64::Assembler;

void GenWrapGuestFunction(MachineCode* mc,
                          GuestAddr pc,
                          const char* signature,
                          HostCode guest_runner,
                          const char* name) {
  UNUSED(name);

  Assembler as(mc);

  // On function entry, rsp + 8 is a multiple of 16.
  // Right before next function call, rsp is a multiple of 16.

  // Default prologue.
  as.Push(Assembler::rbp);
  as.Movq(Assembler::rbp, Assembler::rsp);

  static_assert(alignof(GuestArgumentBuffer) <= 16, "unexpected GuestArgumentBuffer alignment");

  // Estimate guest argument buffer size.
  // Each argument can be 2 8-bytes at most. Result can be 2 8-bytes at most.
  // At least 8 arguments go to registers in GuestArgumentBuffer.
  // First 8-byte of stack is in GuestArgumentBuffer.
  // Result is return on registers in GuestArgumentBuffer.
  // TODO(eaeltsin): maybe run parameter passing to calculate exactly?
  size_t num_args = strlen(signature) - 1;
  size_t max_stack_argv_size = (num_args > 8 ? num_args - 8 : 0) * 16;
  size_t guest_argument_buffer_size = sizeof(GuestArgumentBuffer) - 8 + max_stack_argv_size;

  size_t aligned_frame_size = AlignUp(guest_argument_buffer_size, 16);

  // Allocate stack frame.
  as.Subq(Assembler::rsp, static_cast<int32_t>(aligned_frame_size));

  // rsp is 16-bytes aligned and points to GuestArgumentBuffer.

  constexpr int kArgcOffset = offsetof(GuestArgumentBuffer, argc);
  constexpr int kRescOffset = offsetof(GuestArgumentBuffer, resc);
  constexpr int kArgvOffset = offsetof(GuestArgumentBuffer, argv);
  constexpr int kSimdArgcOffset = offsetof(GuestArgumentBuffer, simd_argc);
  constexpr int kSimdRescOffset = offsetof(GuestArgumentBuffer, simd_resc);
  constexpr int kSimdArgvOffset = offsetof(GuestArgumentBuffer, simd_argv);
  constexpr int kStackArgcOffset = offsetof(GuestArgumentBuffer, stack_argc);
  constexpr int kStackArgvOffset = offsetof(GuestArgumentBuffer, stack_argv);

  const int params_offset = aligned_frame_size + 16;

  // Convert parameters and set argc.
  int argc = 0;
  int simd_argc = 0;
  int stack_argc = 0;
  int host_stack_argc = 0;
  for (size_t i = 1; signature[i] != '\0'; ++i) {
    if (signature[i] == 'z' || signature[i] == 'b' || signature[i] == 's' || signature[i] == 'c' ||
        signature[i] == 'i' || signature[i] == 'p' || signature[i] == 'l') {
      static constexpr Assembler::Register kParamRegs[] = {
          Assembler::rdi,
          Assembler::rsi,
          Assembler::rdx,
          Assembler::rcx,
          Assembler::r8,
          Assembler::r9,
      };
      if (argc < static_cast<int>(std::size(kParamRegs))) {
        as.Movq({.base = Assembler::rsp, .disp = kArgvOffset + argc * 8}, kParamRegs[argc]);
      } else if (argc < 8) {
        as.Movq(Assembler::rax,
                {.base = Assembler::rsp, .disp = params_offset + host_stack_argc * 8});
        ++host_stack_argc;
        as.Movq({.base = Assembler::rsp, .disp = kArgvOffset + argc * 8}, Assembler::rax);
      } else {
        as.Movq(Assembler::rax,
                {.base = Assembler::rsp, .disp = params_offset + host_stack_argc * 8});
        ++host_stack_argc;
        as.Movq({.base = Assembler::rsp, .disp = kStackArgvOffset + stack_argc * 8},
                Assembler::rax);
        ++stack_argc;
      }
      ++argc;
    } else if (signature[i] == 'f' || signature[i] == 'd') {
      static constexpr Assembler::XMMRegister kParamRegs[] = {
          Assembler::xmm0,
          Assembler::xmm1,
          Assembler::xmm2,
          Assembler::xmm3,
          Assembler::xmm4,
          Assembler::xmm5,
          Assembler::xmm6,
          Assembler::xmm7,
      };
      if (simd_argc < static_cast<int>(std::size(kParamRegs))) {
        as.Movq({.base = Assembler::rsp, .disp = kSimdArgvOffset + simd_argc * 16},
                kParamRegs[simd_argc]);
      } else {
        as.Movq(Assembler::rax,
                {.base = Assembler::rsp, .disp = params_offset + host_stack_argc * 8});
        ++host_stack_argc;
        as.Movq({.base = Assembler::rsp, .disp = kStackArgvOffset + stack_argc * 8},
                Assembler::rax);
        ++stack_argc;
      }
      ++simd_argc;
    } else {
      FATAL("signature char '%c' not supported", signature[i]);
    }
  }
  as.Movl({.base = Assembler::rsp, .disp = kArgcOffset}, std::min(argc, 8));
  as.Movl({.base = Assembler::rsp, .disp = kSimdArgcOffset}, std::min(simd_argc, 8));
  // ATTENTION: GuestArgumentBuffer::stack_argc is in bytes!
  as.Movl({.base = Assembler::rsp, .disp = kStackArgcOffset}, stack_argc * 8);

  // Set resc.
  if (signature[0] == 'z' || signature[0] == 'b' || signature[0] == 's' ||
      signature[0] == 'c' | signature[0] == 'i' || signature[0] == 'p' || signature[0] == 'l') {
    as.Movl({.base = Assembler::rsp, .disp = kRescOffset}, 1);
    as.Movl({.base = Assembler::rsp, .disp = kSimdRescOffset}, 0);
  } else if (signature[0] == 'f' || signature[0] == 'd') {
    as.Movl({.base = Assembler::rsp, .disp = kRescOffset}, 0);
    as.Movl({.base = Assembler::rsp, .disp = kSimdRescOffset}, 1);
  } else {
    CHECK_EQ('v', signature[0]);
    as.Movl({.base = Assembler::rsp, .disp = kRescOffset}, 0);
    as.Movl({.base = Assembler::rsp, .disp = kSimdRescOffset}, 0);
  }

  // Call guest runner.
  as.Movq(Assembler::rdi, pc);
  as.Movq(Assembler::rsi, Assembler::rsp);
  as.Call(guest_runner);

  // Get the result.
  if (signature[0] == 'z' || signature[0] == 'b' || signature[0] == 's' ||
      signature[0] == 'c' | signature[0] == 'i' || signature[0] == 'p' || signature[0] == 'l') {
    as.Movq(Assembler::rax, {.base = Assembler::rsp, .disp = kArgvOffset});
  } else if (signature[0] == 'f' || signature[0] == 'd') {
    as.Movq(Assembler::xmm0, {.base = Assembler::rsp, .disp = kSimdArgvOffset});
  } else {
    CHECK_EQ('v', signature[0]);
  }

  // Free stack frame.
  as.Addq(Assembler::rsp, static_cast<int32_t>(aligned_frame_size));

  // Default epilogue.
  as.Pop(Assembler::rbp);
  as.Ret();

  as.Finalize();
}

}  // namespace berberis
