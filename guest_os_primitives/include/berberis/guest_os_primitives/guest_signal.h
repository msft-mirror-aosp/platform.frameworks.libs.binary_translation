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

#ifndef BERBERIS_GUEST_OS_PRIMITIVES_GUEST_SIGNAL_H_
#define BERBERIS_GUEST_OS_PRIMITIVES_GUEST_SIGNAL_H_

#include <csignal>
#include <cstring>

#include "berberis/base/struct_check.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

// Maximum number of signals for the guest kernel.
constexpr int Guest__KERNEL__NSIG = 64;

const GuestAddr Guest_SIG_DFL = 0;
const GuestAddr Guest_SIG_IGN = 1;
const GuestAddr Guest_SIG_ERR = -1;

// Guest siginfo_t, as expected by guest rt_sigqueueinfo syscall.
using Guest_siginfo_t = siginfo_t;

// Guest sigset_t, as expected by guest rt_sigprocmask syscall.
#if defined(BERBERIS_GUEST_LP64)
typedef struct {
  unsigned long __bits[1];
} Guest_sigset_t;
CHECK_STRUCT_LAYOUT(Guest_sigset_t, 64, 64);
#else
// TODO(b/283352810): Explicitly support ILP32 guest data model.
// This condition currently assumes ILP32 support.
typedef struct {
  unsigned long __bits[2];
} Guest_sigset_t;
CHECK_STRUCT_LAYOUT(Guest_sigset_t, 64, 32);
#endif

// TODO(b/280551353): check other SA_* flags!
static_assert(SA_NODEFER == 0x40000000, "Host and guest SA_NODEFER don't match");

template <typename SmallSigset, typename BigSigset>
inline void ConvertToSmallSigset(const BigSigset& big_sigset, SmallSigset* small_sigset) {
  static_assert(sizeof(SmallSigset) <= sizeof(BigSigset), "wrong sigset size");
  memcpy(small_sigset, &big_sigset, sizeof(SmallSigset));
}

template <typename SmallSigset, typename BigSigset>
inline void ConvertToBigSigset(const SmallSigset& small_sigset, BigSigset* big_sigset) {
  static_assert(sizeof(SmallSigset) <= sizeof(BigSigset), "wrong sigset size");
  memset(big_sigset, 0, sizeof(BigSigset));
  memcpy(big_sigset, &small_sigset, sizeof(SmallSigset));
}

// TODO(b/283697533): Define Guest_sigaction differently depending on guest platform.
// Guest struct (__kernel_)sigaction, as expected by rt_sigaction syscall.
struct Guest_sigaction {
  // Prefix avoids conflict with original 'sa_sigaction' defined as macro.
  GuestAddr guest_sa_sigaction;
  unsigned long sa_flags;
  Guest_sigset_t sa_mask;
};
#if defined(BERBERIS_GUEST_LP64)
CHECK_STRUCT_LAYOUT(Guest_sigaction, 192, 64);
CHECK_FIELD_LAYOUT(Guest_sigaction, guest_sa_sigaction, 0, 64);
CHECK_FIELD_LAYOUT(Guest_sigaction, sa_flags, 64, 64);
CHECK_FIELD_LAYOUT(Guest_sigaction, sa_mask, 128, 64);
#endif
// TODO(b/283352810): Add checks for ILP32 guest data model.

struct Guest_sigaction;
bool SetGuestSignalHandler(int signal,
                           const Guest_sigaction* act,
                           Guest_sigaction* old_act,
                           int* error);

}  // namespace berberis

#endif  // BERBERIS_GUEST_OS_PRIMITIVES_GUEST_SIGNAL_H_