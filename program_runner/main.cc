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

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <tuple>

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"
#include "berberis/base/file.h"
#include "berberis/guest_loader/guest_loader.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime/berberis.h"

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

bool Run(const char* vdso_path,
         const char* loader_path,
         int argc,
         const char* argv[],
         char* envp[],
         std::string* error_msg) {
  InitBerberis();

  std::string executable_realpath;
  if (!Realpath(argv[0], &executable_realpath)) {
    *error_msg = std::string("Unable to get realpath of ") + argv[0];
    return false;
  }

  GuestLoader::StartExecutable(
      executable_realpath.c_str(), vdso_path, loader_path, argc, argv, envp, error_msg);

  return false;
}

}  // namespace

}  // namespace berberis

int main(int argc, char* argv[], char* envp[]) {
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

  return 0;
}
