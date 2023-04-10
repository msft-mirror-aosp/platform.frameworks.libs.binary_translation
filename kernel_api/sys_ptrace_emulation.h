/*
 * Copyright (C) 2018 The Android Open Source Project
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

#ifndef BERBERIS_KERNEL_API_SYS_PTRACE_EMULATION_H_
#define BERBERIS_KERNEL_API_SYS_PTRACE_EMULATION_H_

#include <sys/ptrace.h>
#include <sys/types.h>

#include <tuple>

namespace berberis {

int PtraceForGuest(int request, pid_t pid, void* addr, void* data);
std::tuple<bool, int> PtraceForGuestArch(int request, pid_t pid, void* addr, void* data);

}  // namespace berberis

#endif  // BERBERIS_KERNEL_API_SYS_PTRACE_EMULATION_H_
