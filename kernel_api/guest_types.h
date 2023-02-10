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

#ifndef BERBERIS_KERNEL_API_GUEST_TYPES_H_
#define BERBERIS_KERNEL_API_GUEST_TYPES_H_

#include <sys/epoll.h>
#include <sys/file.h>

#include <climits>
#include <cstdint>

#include "berberis/base/struct_check.h"

// Architecture-specific guest types.
#include "guest_types_arch.h"

namespace berberis {

static_assert(EPOLL_CTL_ADD == 1);
static_assert(EPOLL_CTL_DEL == 2);
static_assert(EPOLL_CTL_MOD == 3);
static_assert(EPOLL_CLOEXEC == 02000000);

struct Guest_epoll_event {
  uint32_t events;
  alignas(64 / CHAR_BIT) uint64_t data;
};

// Verify precise layouts so ConvertHostEPollEventArrayToGuestInPlace is safe.
CHECK_STRUCT_LAYOUT(Guest_epoll_event, 128, 64);
CHECK_FIELD_LAYOUT(Guest_epoll_event, events, 0, 32);
CHECK_FIELD_LAYOUT(Guest_epoll_event, data, 64, 64);
static_assert(sizeof(epoll_event) <= sizeof(Guest_epoll_event));
static_assert(alignof(epoll_event) <= alignof(Guest_epoll_event));

}  // namespace berberis

#endif  // BERBERIS_KERNEL_API_GUEST_TYPES_H_
