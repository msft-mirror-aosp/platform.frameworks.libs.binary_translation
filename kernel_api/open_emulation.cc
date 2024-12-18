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
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <utility>

#include "berberis/base/arena_alloc.h"
#include "berberis/base/arena_map.h"
#include "berberis/base/arena_string.h"
#include "berberis/base/arena_vector.h"
#include "berberis/base/checks.h"
#include "berberis/base/fd.h"
#include "berberis/base/forever_alloc.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/kernel_api/main_executable_real_path_emulation.h"

namespace berberis {

namespace {

class EmulatedFileDescriptors {
 public:
  explicit EmulatedFileDescriptors() : fds_(&arena_) {}

  static EmulatedFileDescriptors* GetInstance() {
    static auto* g_emulated_proc_self_maps_fds = NewForever<EmulatedFileDescriptors>();
    return g_emulated_proc_self_maps_fds;
  }

  // Not copyable or movable.
  EmulatedFileDescriptors(const EmulatedFileDescriptors&) = delete;
  EmulatedFileDescriptors& operator=(const EmulatedFileDescriptors&) = delete;

  void Add(int fd) {
    std::lock_guard lock(mutex_);
    auto [unused_it, inserted] = fds_.insert(std::make_pair(fd, 0));
    if (!inserted) {
      // We expect every fd to be added at most once. But if it breaks let's consider it non-fatal.
      TRACE("Detected duplicated fd in EmulatedFileDescriptors");
    }
  }

  bool Contains(int fd) {
    std::lock_guard lock(mutex_);
    return fds_.find(fd) != fds_.end();
  }

  void Remove(int fd) {
    std::lock_guard lock(mutex_);
    auto it = fds_.find(fd);
    if (it != fds_.end()) {
      fds_.erase(it);
    }
  }

 private:
  std::mutex mutex_;
  Arena arena_;
  // We use it as a set because we don't have ArenaSet, so client data isn't really used.
  ArenaMap<int, int> fds_;
};

// It's macro since we use it as string literal below.
#define PROC_SELF_MAPS "/proc/self/maps"

// Reader that works with custom allocator strings. Based on android::base::ReadFileToString.
template <typename String>
bool ReadProcSelfMapsToString(String& content) {
  int fd = open(PROC_SELF_MAPS, O_RDONLY);
  if (fd == -1) {
    return false;
  }
  char buf[4096] __attribute__((__uninitialized__));
  ssize_t n;
  while ((n = read(fd, &buf[0], sizeof(buf))) > 0) {
    content.append(buf, n);
  }
  close(fd);
  return n == 0;
}

// String split that works with custom allocator strings. Based on android::base::Split.
template <typename String>
ArenaVector<String> SplitLines(Arena* arena, const String& content) {
  ArenaVector<String> lines(arena);
  size_t base = 0;
  size_t found;
  while (true) {
    found = content.find_first_of('\n', base);
    lines.emplace_back(content, base, found - base, content.get_allocator());
    if (found == content.npos) break;
    base = found + 1;
  }
  return lines;
}

// Note that dirfd, flags and mode are only used to fallback to
// host's openat in case of failure.
// Avoid mallocs since bionic tests use it under malloc_disable (b/338211718).
int OpenatProcSelfMapsForGuest(int dirfd, int flags, mode_t mode) {
  TRACE("Openat for " PROC_SELF_MAPS);

  Arena arena;
  ArenaString file_data(&arena);
  bool success = ReadProcSelfMapsToString(file_data);
  if (!success) {
    TRACE("Cannot read " PROC_SELF_MAPS ", falling back to host's openat");
    return openat(dirfd, PROC_SELF_MAPS, flags, mode);
  }

  int mem_fd = CreateMemfdOrDie("[guest " PROC_SELF_MAPS "]");

  auto* maps_shadow = GuestMapShadow::GetInstance();

  auto lines = SplitLines(&arena, file_data);
  ArenaString guest_maps(&arena);
  for (size_t i = 0; i < lines.size(); i++) {
    uintptr_t start;
    uintptr_t end;
    int prot_offset;
    auto& cur_line = lines.at(i);
    if (sscanf(cur_line.c_str(), "%" SCNxPTR "-%" SCNxPTR " %n", &start, &end, &prot_offset) != 2) {
      if (!cur_line.empty()) {
        TRACE("Cannot parse " PROC_SELF_MAPS " line : %s", cur_line.c_str());
      }
      guest_maps.append(cur_line + "\n");
      continue;
    }
    // Split the line into guest exec / no-exec chunks.
    uintptr_t original_start = start;
    while (start < end) {
      auto [is_exec, region_size] =
          maps_shadow->GetExecutableRegionSize(GuestAddr(start), end - start);
      // prot_offset points to "rwxp", so offset of "x" is 2 symbols away.
      cur_line.at(prot_offset + 2) = is_exec ? 'x' : '-';
      if ((start == original_start) && ((start + region_size) >= end)) {
        // Most often we should be able to just take the whole host line.
        guest_maps.append(cur_line);
        guest_maps.append("\n");
        break;
      }
      // We cannot print into cur_line in place since we don't want the terminating null. Also the
      // new range can theoretically be longer than the old one. E.g. if "a000-ba000" (len=10) is
      // split into "a000-aa000" (len=10) and "aa000-ba000" (len=11).
      // At max, for 64-bit pointers, we need 16(ptr)+1(-)+16(ptr)+1(\0)=34 symbols buffer,
      // so 64-bytes should be more than enough.
      char addr_range_buf[64];
      int chars_num = snprintf(addr_range_buf,
                               sizeof(addr_range_buf),
                               "%" PRIxPTR "-%" PRIxPTR,
                               start,
                               start + region_size);
      CHECK_LT(static_cast<size_t>(chars_num), sizeof(addr_range_buf));
      guest_maps.append(addr_range_buf);
      // Append the rest of the line starting from protections and including the front space.
      guest_maps.append(cur_line.data() + prot_offset - 1);
      guest_maps.append("\n");
      start += region_size;
    }
  }

  // Normally /proc/self/maps doesn't have newline at the end.
  // It's simpler to remove it than to not add it in the loop.
  CHECK_EQ(guest_maps.back(), '\n');
  guest_maps.pop_back();

  TRACE("--------\n%s\n--------", guest_maps.c_str());

  WriteFullyOrDie(mem_fd, guest_maps.c_str(), guest_maps.size());

  lseek(mem_fd, 0, 0);

  EmulatedFileDescriptors::GetInstance()->Add(mem_fd);

  return mem_fd;
}

bool IsProcSelfMaps(const char* path, int flags) {
  struct stat cur_stat;
  struct stat proc_stat;
  // This check works for /proc/self/maps itself as well as symlinks (unless AT_SYMLINK_NOFOLLOW is
  // requested). As an added benefit it gracefully handles invalid pointers in path.
  return stat(path, &cur_stat) == 0 && stat(PROC_SELF_MAPS, &proc_stat) == 0 &&
         !(S_ISLNK(cur_stat.st_mode) && (flags & AT_SYMLINK_NOFOLLOW)) &&
         cur_stat.st_ino == proc_stat.st_ino && cur_stat.st_dev == proc_stat.st_dev;
}

// In zygote this is not needed because native bridge is mounting
// /proc/cpuinfo to point to the emulated version. But for executables
// this does not happen and they end up reading host cpuinfo.
//
// Note that current selinux policies prevent us from mounting /proc/cpuinfo
// (replicating what zygote process does) for executables hence we need to
// emulate it here.
const char* TryTranslateProcCpuinfoPath(const char* path, int flags) {
#if defined(__ANDROID__)
  struct stat cur_stat;
  struct stat cpuinfo_stat;

  if (stat(path, &cur_stat) == -1 || stat("/proc/cpuinfo", &cpuinfo_stat) == -1) {
    return nullptr;
  }

  if (S_ISLNK(cur_stat.st_mode) && (flags & AT_SYMLINK_NOFOLLOW)) {
    return nullptr;
  }

  if ((cur_stat.st_ino == cpuinfo_stat.st_ino) && (cur_stat.st_dev == cpuinfo_stat.st_dev)) {
    TRACE("openat: Translating %s to %s", path, kGuestCpuinfoPath);
    return kGuestCpuinfoPath;
  }
#else
  UNUSED(path, flags);
#endif
  return nullptr;
}

}  // namespace

bool IsFileDescriptorEmulatedProcSelfMaps(int fd) {
  return EmulatedFileDescriptors::GetInstance()->Contains(fd);
}

void CloseEmulatedProcSelfMapsFileDescriptor(int fd) {
  EmulatedFileDescriptors::GetInstance()->Remove(fd);
}

int OpenatForGuest(int dirfd, const char* path, int guest_flags, mode_t mode) {
  int host_flags = ToHostOpenFlags(guest_flags);

  if (IsProcSelfMaps(path, host_flags)) {
    return OpenatProcSelfMapsForGuest(dirfd, host_flags, mode);
  }

  const char* real_path = nullptr;
  if ((host_flags & AT_SYMLINK_NOFOLLOW) == 0) {
    real_path = TryReadLinkToMainExecutableRealPath(path);
  }

  if (real_path == nullptr) {
    real_path = TryTranslateProcCpuinfoPath(path, host_flags);
  }

  return openat(dirfd, real_path != nullptr ? real_path : path, host_flags, mode);
}

int OpenForGuest(const char* path, int flags, mode_t mode) {
  return OpenatForGuest(AT_FDCWD, path, flags, mode);
}

}  // namespace berberis
