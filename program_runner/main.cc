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

#include <malloc.h>
#include <sys/auxv.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"
#include "berberis/base/file.h"
#include "berberis/guest_state/guest_addr.h"

#if defined(__x86_64__)
#include "berberis/guest_loader/guest_loader.h"
#include "berberis/program_runner/program_runner.h"
#include "berberis/runtime/berberis.h"
#elif defined(__aarch64__)
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/riscv64/interpreter.h"
#include "berberis/tiny_loader/loaded_elf_file.h"
#include "berberis/tiny_loader/tiny_loader.h"
#endif

// Program runner meant for testing and manual invocation.

namespace berberis {

namespace {

void Usage(const char* argv_0) {
  printf(
      "Usage: %s [-h] guest_executable [arg1 [arg2 ...]]\n"
      "  -h             - print this message\n"
      "  guest_executable - path to the guest executable\n",
      argv_0);
}

struct Options {
  const char* guest_executable;
  bool print_help_and_exit;
};

Options ParseArgs(int argc, char* argv[]) {
  CHECK_GE(argc, 1);

  Options opts{};

  while (true) {
    int c = getopt(argc, argv, "+h:");
    if (c < 0) {
      break;
    }
    switch (c) {
      case 'h':
        return Options{.print_help_and_exit = true};
      default:
        UNREACHABLE();
    }
  }

  if (optind >= argc) {
    return Options{.print_help_and_exit = true};
  }

  opts.print_help_and_exit = false;
  return opts;
}

}  // namespace

}  // namespace berberis

int main(int argc, char* argv[], [[maybe_unused]] char* envp[]) {
#if defined(__GLIBC__)
  // Disable brk in glibc-malloc.
  //
  // By default GLIBC uses brk in malloc which may lead to conflicts with
  // executables that use brk for their own needs. See http://b/64720148 for
  // example.
  mallopt(M_MMAP_THRESHOLD, 0);
  mallopt(M_TRIM_THRESHOLD, -1);
#endif

  berberis::Options opts = berberis::ParseArgs(argc, argv);

  if (opts.print_help_and_exit) {
    berberis::Usage(argv[0]);
    return -1;
  }

  std::string error_msg;
#if defined(__x86_64__)
  if (!berberis::Run(
          // TODO(b/276787135): Make vdso and loader configurable via command line arguments.
          /* vdso_path */ nullptr,
          /* loader_path */ nullptr,
          argc - optind,
          const_cast<const char**>(argv + optind),
          envp,
          &error_msg)) {
    fprintf(stderr, "unable to start executable: %s\n", error_msg.c_str());
    return -1;
  }
#elif defined(__aarch64__)
  LoadedElfFile elf_file;
  if (!TinyLoader::LoadFromFile(opts.guest_executable, &elf_file, &error_msg)) {
    fprintf(stderr, "unable to start load file: %s\n", error_msg.c_str());
    return -1;
  }
  if (elf_file->e_type() != ET_EXEC) {
    fprintf(stderr, "this is not a static executable file: %s\n", elf_file->e_type().c_str());
    return -1;
  }

  berberis::ThreadState state{};
  state.cpu.insn_addr = elf_file->entry_point();
  while (true) {
    InterpretInsn(&state);
  }
#endif

  return 0;
}
