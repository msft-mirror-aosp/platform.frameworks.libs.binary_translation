/*
 * Copyright (C) 2014 The Android Open Source Project
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

#include "berberis/kernel_api/open_emulation.h"

#include <fcntl.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "berberis/base/fd.h"
#include "berberis/base/file.h"
#include "berberis/base/strings.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/kernel_api/main_executable_real_path_emulation.h"

namespace berberis {

namespace {

// It's macro since we use it as string literal below.
#define PROC_SELF_MAPS "/proc/self/maps"

// Note that dirfd, flags and mode are only used to fallback to
// host's openat in case of failure.
int OpenatProcSelfMapsForGuest(int dirfd, int flags, mode_t mode) {
  TRACE("Openat for " PROC_SELF_MAPS);

  std::string file_data;
  bool success = ReadFileToString(PROC_SELF_MAPS, &file_data);
  if (!success) {
    TRACE("Cannot read " PROC_SELF_MAPS ", falling back to host's openat");
    return openat(dirfd, PROC_SELF_MAPS, flags, mode);
  }

  int mem_fd = CreateMemfdOrDie("[guest " PROC_SELF_MAPS "]");

  auto* maps_shadow = GuestMapShadow::GetInstance();

  std::vector<std::string> lines = Split(file_data, "\n");
  std::string guest_maps;
  for (size_t i = 0; i < lines.size(); i++) {
    uintptr_t start;
    uintptr_t end;
    int prot_offset;
    if (sscanf(lines.at(i).c_str(), "%" SCNxPTR "-%" SCNxPTR " %n", &start, &end, &prot_offset) !=
        2) {
      if (!lines[i].empty()) {
        TRACE("Cannot parse " PROC_SELF_MAPS " line : %s", lines.at(i).c_str());
      }
      guest_maps.append(lines.at(i) + "\n");
      continue;
    }
    BitValue exec_status = maps_shadow->GetExecutable(GuestAddr(start), end - start);
    if (exec_status == kBitMixed) {
      // When we strip guest executable bit from host mappings the kernel may merge r-- and r-x
      // mappings, resulting in kBitMixed executability state. We are avoiding such merging by
      // SetVmaAnonName in MmapForGuest/MprotectForGuest. This isn't strictly guaranteed to work, so
      // issue a warning if it doesn't, or if we got kBitMixed for another reason to investigate.
      // TODO(b/322873334): Instead split such host mapping into several guest mappings.
      TRACE("Unexpected " PROC_SELF_MAPS " mapping with mixed guest executability");
    }
    // prot_offset points to "rwxp", so offset of "x" is 2 symbols away.
    lines.at(i).at(prot_offset + 2) = (exec_status == kBitSet) ? 'x' : '-';

    guest_maps.append(lines.at(i) + "\n");
  }

  TRACE("--------\n%s\n--------", guest_maps.c_str());

  WriteFullyOrDie(mem_fd, guest_maps.c_str(), guest_maps.size());

  lseek(mem_fd, 0, 0);

  return mem_fd;
}

}  // namespace

int OpenatForGuest(int dirfd, const char* path, int guest_flags, mode_t mode) {
  int host_flags = ToHostOpenFlags(guest_flags);

  if (strcmp(path, PROC_SELF_MAPS) == 0) {
    return OpenatProcSelfMapsForGuest(dirfd, host_flags, mode);
  }

  const char* real_path = nullptr;
  if ((host_flags & AT_SYMLINK_NOFOLLOW) == 0) {
    real_path = TryReadLinkToMainExecutableRealPath(path);
  }

  return openat(dirfd, real_path ? real_path : path, host_flags, mode);
}

int OpenForGuest(const char* path, int flags, mode_t mode) {
  return OpenatForGuest(AT_FDCWD, path, flags, mode);
}

}  // namespace berberis
