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

#include <linux/unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>

#include "berberis/base/checks.h"
#include "berberis/base/macros.h"
#include "berberis/base/scoped_errno.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

namespace {

long RunGuestSyscallImpl(long guest_nr, long arg_1, long arg_2, long arg_3, long arg_4, long arg_5,
                         long arg_6) {
  UNUSED(arg_4, arg_5, arg_6);
  switch (guest_nr) {
    case 64:  // __NR_write
      return syscall(1, arg_1, arg_2, arg_3);
    default:
      FATAL("Unsupported guest syscall %ld", guest_nr);
  }
}

}  // namespace

void RunGuestSyscall(ThreadState* state) {
  ScopedErrno scoped_errno;

  // RISCV Linux takes arguments in a0-a5 and syscall number in a7.
  long result =
      RunGuestSyscallImpl(state->cpu.x[17], state->cpu.x[10], state->cpu.x[11], state->cpu.x[12],
                          state->cpu.x[13], state->cpu.x[14], state->cpu.x[15]);
  // The result is returned in a0.
  if (result == -1) {
    state->cpu.x[10] = -errno;
  } else {
    state->cpu.x[10] = result;
  }
}

}  // namespace berberis
