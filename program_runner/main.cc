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

#include <sys/auxv.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <tuple>

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"
#include "berberis/base/macros.h"
#include "berberis/base/tracing.h"
#include "berberis/calling_conventions/calling_conventions_riscv64.h"
#include "berberis/guest_os_primitives/guest_thread_manager.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/kernel_api/exec_emulation.h"
#include "berberis/runtime/execute_guest.h"
#include "berberis/tiny_loader/loaded_elf_file.h"
#include "berberis/tiny_loader/tiny_loader.h"

namespace berberis {

namespace {

void Usage(const char* argv_0) {
  printf(
      "Usage: %s [-h] [-a start_addr] guest_executable [arg1 [arg2 ...]]\n"
      "  -h             - print this message\n"
      "  -a start_addr  - start execution at start_addr\n"
      "  guest_executable - path to the guest executable\n",
      argv_0);
}

std::tuple<GuestAddr, bool> ParseGuestAddr(const char* addr_cstr) {
  char* end_ptr = nullptr;
  errno = 0;
  GuestAddr addr = bit_cast<GuestAddr>(strtoull(addr_cstr, &end_ptr, 16));

  // Warning: setting errno on failure is implementation defined. So we also use extra heuristics.
  if (errno != 0 || (*end_ptr != '\n' && *end_ptr != '\0')) {
    printf("Cannot convert \"%s\" to integer: %s\n", addr_cstr,
           errno != 0 ? strerror(errno) : "unexpected end of string");
    return {kNullGuestAddr, false};
  }
  return {addr, true};
}

struct Options {
  const char* guest_executable;
  GuestAddr start_addr;
  bool print_help_and_exit;
};

Options ParseArgs(int argc, char* argv[]) {
  CHECK_GE(argc, 1);

  Options opts{.start_addr = kNullGuestAddr};

  while (true) {
    int c = getopt(argc, argv, "ha:");
    if (c < 0) {
      break;
    }
    switch (c) {
      case 'a': {
        auto [addr, success] = ParseGuestAddr(optarg);
        if (!success) {
          return Options{.print_help_and_exit = true};
        }
        opts.start_addr = addr;
        break;
      }
      case 'h':
        return Options{.print_help_and_exit = true};
      default:
        UNREACHABLE();
    }
  }

  if (optind >= argc) {
    return Options{.print_help_and_exit = true};
  }

  opts.guest_executable = argv[optind];
  opts.print_help_and_exit = false;
  return opts;
}

// TODO(b/279068747): Move this to guest_loader.
GuestAddr InitKernelArgs(GuestAddr guest_sp,
                         size_t argc,
                         char* argv[],
                         char* envp[],
                         GuestAddr linker_base_addr,
                         GuestAddr main_executable_entry_point,
                         GuestAddr phdr,
                         size_t phdr_count) {
  static uint8_t kRandomBytes[16];
  std::independent_bits_engine<std::default_random_engine, CHAR_BIT, uint8_t> engine;
  std::generate(&kRandomBytes[0], &kRandomBytes[0] + sizeof(kRandomBytes), std::ref(engine));

  // TODO(b/279068747): Provide meaningful values for disabled arguments.
  // clang-format off
  const uint64_t auxv[] = {
      // AT_HWCAP,        kRiscv64ValueHwcap,
      // AT_HWCAP2,       kRiscv64ValueHwcap2,
      AT_RANDOM,       ToGuestAddr(&kRandomBytes[0]),
      AT_SECURE,       false,
      AT_BASE,         linker_base_addr,
      AT_PHDR,         phdr,
      AT_PHNUM,        phdr_count,
      AT_ENTRY,        main_executable_entry_point,
      AT_PAGESZ,       static_cast<uint64_t>(sysconf(_SC_PAGESIZE)),
      AT_CLKTCK,       static_cast<uint64_t>(sysconf(_SC_CLK_TCK)),
      // AT_SYSINFO_EHDR, ehdr_vdso,
      AT_UID,          getuid(),
      AT_EUID,         geteuid(),
      AT_GID,          getgid(),
      AT_EGID,         getegid(),
      AT_NULL,
  };
  // clang-format on

  size_t envp_count = 0;  // number of environment variables + nullptr
  while (envp[envp_count++] != nullptr) {
  }

  guest_sp -= sizeof(uint64_t) +               // argc
              sizeof(uint64_t) * (argc + 1) +  // argv + nullptr
              sizeof(uint64_t) * envp_count +  // envp + nullptr
              sizeof(auxv);                    // auxv

  guest_sp = AlignDown(guest_sp, riscv64::CallingConventions::kStackAlignmentBeforeCall);

  uint64_t* curr = ToHostAddr<uint64_t>(guest_sp);

  // argc
  *curr++ = argc;

  // argv
  for (size_t i = 0; i < argc; ++i) {
    *curr++ = reinterpret_cast<uint64_t>(argv[i]);
  }
  *curr++ = kNullGuestAddr;

  // envp
  curr = reinterpret_cast<uint64_t*>(DemangleGuestEnvp(reinterpret_cast<char**>(curr), envp));

  // auxv
  memcpy(curr, auxv, sizeof(auxv));

  return guest_sp;
}
}  // namespace

}  // namespace berberis

int main(int argc, char* argv[], char* envp[]) {
  berberis::Options opts = berberis::ParseArgs(argc, argv);

  if (opts.print_help_and_exit) {
    berberis::Usage(argv[0]);
    return -1;
  }

  LoadedElfFile elf_file;
  std::string error_msg;
  if (!TinyLoader::LoadFromFile(opts.guest_executable, &elf_file, &error_msg)) {
    printf("%s\n", error_msg.c_str());
    return -1;
  }

  // TODO(b/276786584): define InitBerberis instead.
  berberis::InitGuestThreadManager();
  berberis::Tracing::Init();

  auto* thread = berberis::GetCurrentGuestThread();
  auto& cpu_state = thread->state()->cpu;
  if (opts.start_addr != berberis::kNullGuestAddr) {
    cpu_state.insn_addr = opts.start_addr;
  } else {
    cpu_state.insn_addr = berberis::ToGuestAddr(elf_file.entry_point());
  }

  cpu_state.x[berberis::SP] =
      berberis::InitKernelArgs(cpu_state.x[berberis::SP],
                               argc,
                               argv,
                               envp,
                               berberis::ToGuestAddr(elf_file.base_addr()),
                               berberis::ToGuestAddr(elf_file.entry_point()),
                               berberis::ToGuestAddr(elf_file.phdr_table()),
                               elf_file.phdr_count());

  ExecuteGuest(thread->state(), berberis::kNullGuestAddr);

  return 0;
}
