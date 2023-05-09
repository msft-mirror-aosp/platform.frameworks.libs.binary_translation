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

#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/base/macros.h"

namespace berberis {

GuestThread* GuestThread::CreatePthread(void* stack, size_t stack_size, size_t guard_size) {
  // TODO(b/280551726): Implement.
  UNUSED(stack, stack_size, guard_size);
  return nullptr;
}

void GuestThread::Destroy(GuestThread* thread) {
  // TODO(b/280551726): Implement.
  UNUSED(thread);
}

void GuestThread::InitStaticTls() {
  // TODO(b/280551726): Implement.
}

}  // namespace berberis