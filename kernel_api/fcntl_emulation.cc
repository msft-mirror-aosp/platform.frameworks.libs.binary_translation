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

// We need 32-bit functions here.  Suppress unconditional use of 64-bit offsets.
// Functions with 64-bit offsets are still available when used with "64" suffix.
//
// Note: this is actually only needed for host build since Android build system
// insists on defining _FILE_OFFSET_BITS=64 for host-host binaries.
//
// _FILE_OFFSET_BITS is NOT defined when we are building target-host binaries.
#ifdef _FILE_OFFSET_BITS
#undef _FILE_OFFSET_BITS
#endif

#include "fcntl_emulation.h"

#include <fcntl.h>
#include <sys/file.h>

#include <cerrno>

#include "berberis/base/logging.h"

#include "guest_types.h"
#include "open_emulation.h"

namespace berberis {

#define GUEST_F_GETLK 5
#define GUEST_F_SETLK 6
#define GUEST_F_SETLKW 7

int GuestFcntl(int fd, int cmd, long arg_3) {
  auto [processed, result] = GuestFcntlArch(fd, cmd, arg_3);
  if (processed) {
    return result;
  }

  switch (cmd) {
    case F_GETFD:
    case F_GETOWN:
    case F_GETSIG:
    case F_GETLEASE:
      return fcntl(fd, cmd);
    case F_GETFL: {
      auto result = fcntl(fd, cmd);
      if (result < 0) {
        return result;
      }
      return ToGuestOpenFlags(result);
    }
    case F_DUPFD:
    case F_DUPFD_CLOEXEC:
    case F_SETFD:
    case F_SETOWN:
    case F_SETSIG:
    case F_SETLEASE:
    case F_NOTIFY:
    case F_GETOWN_EX:
    case F_SETOWN_EX:
#if defined(F_ADD_SEALS)
    case F_ADD_SEALS:
#endif
#if defined(F_GET_SEALS)
    case F_GET_SEALS:
#endif
    case GUEST_F_SETLK:
    case GUEST_F_SETLKW:
    case GUEST_F_GETLK:
      return fcntl(fd, cmd, arg_3);
    case F_SETFL:
      return fcntl(fd, cmd, ToHostOpenFlags(arg_3));
    default:
      ALOGE("Unknown fcntl command: %d", cmd);
      errno = ENOSYS;
      return -1;
  }
}

}  // namespace berberis
