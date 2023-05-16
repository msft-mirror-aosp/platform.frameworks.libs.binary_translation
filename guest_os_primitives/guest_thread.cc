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

#include <sys/mman.h>  // mprotect

#if defined(__BIONIC__)
#include "private/bionic_constants.h"
#endif

#include "berberis/base/mmap.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_state/guest_addr.h"  // ToGuestAddr
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/runtime_primitives/host_stack.h"
#include "native_bridge_support/linker/static_tls_config.h"

namespace berberis {

NativeBridgeStaticTlsConfig g_static_tls_config;

namespace {

constexpr size_t kGuestThreadPageAlignedSize = AlignUpPageSize(sizeof(GuestThread));

}  // namespace

// static
GuestThread* GuestThread::Create() {
  // ATTENTION: GuestThread is aligned on 16, as fp registers in CPUState are aligned on 16, for
  // efficient handling with aligned SSE memory access instructions. Thus, avoid using 'new', as
  // it might not honor alignment! See b/64554026.
  //
  // ATTENTION: Bionic allocates thread internal data together with thread stack.
  // In case of user provided stack, thread internal data goes there.
  void* thread_storage = Mmap(kGuestThreadPageAlignedSize);
  if (thread_storage == MAP_FAILED) {
    return nullptr;
  }

  GuestThread* thread = new (thread_storage) GuestThread;
  CHECK(thread);

  thread->state_ = CreateThreadState();
  if (!thread->state_) {
    TRACE("failed to allocate thread state");
    Destroy(thread);
    return nullptr;
  }
  InitThreadState(thread->state_);
  SetGuestThread(thread->state_, thread);

  return thread;
}

// static
GuestThread* GuestThread::CreatePthread(void* stack, size_t stack_size, size_t guard_size) {
  GuestThread* thread = Create();
  if (thread == nullptr) {
    return nullptr;
  }

  if (!thread->AllocStack(stack, stack_size, guard_size)) {
    Destroy(thread);
    return nullptr;
  }
  // TODO(b/280551726): Implement.
  // SetStackRegister(&thread->state_->cpu, thread->stack_top_);

  // TODO(b/281859262): Implement shadow call stack initialization.

  // Static TLS must be in an independent mapping, because on creation of main thread its config
  // is yet unknown. Loader sets main thread's static TLS explicitly later.
  if (!thread->AllocStaticTls()) {
    Destroy(thread);
    return nullptr;
  }

  return thread;
}

// static
void GuestThread::Destroy(GuestThread* thread) {
  // ATTENTION: Don't run guest code from here!
  if (ArePendingSignalsPresent(thread->state_)) {
    TRACE("thread destroyed with pending signals, signals ignored!");
  }

  if (thread->host_stack_) {
    // This happens only on cleanup after failed creation.
    MunmapOrDie(thread->host_stack_, GetStackSizeForTranslation());
  }
  if (thread->mmap_size_) {
    MunmapOrDie(thread->stack_, thread->mmap_size_);
  }
#if defined(__BIONIC__)
  if (thread->static_tls_ != nullptr) {
    MunmapOrDie(thread->static_tls_, AlignUpPageSize(g_static_tls_config.size));
  }
  if (thread->scs_region_ != nullptr) {
    MunmapOrDie(thread->scs_region_, SCS_GUARD_REGION_SIZE);
  }
#endif  // defined(__BIONIC__)
  if (thread->state_) {
    DestroyThreadState(thread->state_);
  }
  MunmapOrDie(thread, kGuestThreadPageAlignedSize);
}

bool GuestThread::AllocStack(void* stack, size_t stack_size, size_t guard_size) {
  // Here is what bionic does, see bionic/pthread_create.cpp:
  //
  // For user-provided stack, it assumes guard_size is included in stack size.
  //
  // For new stack, it adds given guard and stack sizes to get actual stack size:
  //   |<- guard_size ->|<- stack_size -------------------->|
  //   | guard          | stack        | pthread_internal_t | tls | GUARD |
  //   |<- actual stack_size --------->|
  //   ^ stack_base                    ^ stack_top

  if (stack) {
    // User-provided stack.
    stack_ = nullptr;  // Do not unmap in Destroy!
    mmap_size_ = 0;
    guard_size_ = guard_size;
    stack_size_ = stack_size;
    stack_top_ = ToGuestAddr(stack) + stack_size_;
    return true;
  }

  guard_size_ = AlignUpPageSize(guard_size);
  mmap_size_ = guard_size_ + AlignUpPageSize(stack_size);
  stack_size_ = mmap_size_;

  stack_ = Mmap(mmap_size_);
  if (stack_ == MAP_FAILED) {
    TRACE("failed to allocate stack!");
    stack_ = nullptr;  // Do not unmap in Destroy!
    return false;
  }

  if (mprotect(stack_, guard_size_, PROT_NONE) != 0) {
    TRACE("failed to protect stack!");
    return false;
  }

  stack_top_ = ToGuestAddr(stack_) + stack_size_ - 16;
  return true;
}

bool GuestThread::AllocShadowCallStack() {
  // TODO(b/281859262): Implement.
  return true;
}

bool GuestThread::AllocStaticTls() {
  // For the main thread, this function is called twice.

  CHECK_EQ(nullptr, static_tls_);

#if defined(__BIONIC__)
  if (g_static_tls_config.size > 0) {
    static_tls_ = Mmap(AlignUpPageSize(g_static_tls_config.size));
    if (static_tls_ == MAP_FAILED) {
      TRACE("failed to allocate static tls!");
      static_tls_ = nullptr;  // Do not unmap in Destroy!
      return false;
    }
  }
#endif  // defined(__BIONIC__)

  return true;
}

}  // namespace berberis