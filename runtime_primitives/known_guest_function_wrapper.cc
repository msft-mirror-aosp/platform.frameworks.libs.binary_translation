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

#include "berberis/runtime_primitives/known_guest_function_wrapper.h"

#include <map>
#include <mutex>
#include <string>

#include "berberis/base/forever_alloc.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

namespace {

class GuestFunctionWrapper {
 public:
  static GuestFunctionWrapper* GetInstance() {
    static auto* g_wrapper = NewForever<GuestFunctionWrapper>();
    return g_wrapper;
  }

  void RegisterKnown(const char* name, HostCode (*wrapper)(GuestAddr)) {
    std::lock_guard<std::mutex> guard(mutex_);
    wrappers_.insert({name, wrapper});
  }

  HostCode WrapKnown(GuestAddr guest_addr, const char* name) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto wrapper = wrappers_.find(name);
    if (wrapper == end(wrappers_)) {
      return nullptr;
    }
    return wrapper->second(guest_addr);
  }

 private:
  GuestFunctionWrapper() = default;
  GuestFunctionWrapper(const GuestFunctionWrapper&) = delete;
  GuestFunctionWrapper& operator=(const GuestFunctionWrapper&) = delete;

  friend GuestFunctionWrapper* NewForever<GuestFunctionWrapper>();

  std::map<std::string, HostCode (*)(GuestAddr)> wrappers_;
  std::mutex mutex_;
};

}  // namespace

void RegisterKnownGuestFunctionWrapper(const char* name, HostCode (*wrapper)(GuestAddr)) {
  GuestFunctionWrapper::GetInstance()->RegisterKnown(name, wrapper);
}

HostCode WrapKnownGuestFunction(GuestAddr guest_addr, const char* name) {
  return GuestFunctionWrapper::GetInstance()->WrapKnown(guest_addr, name);
}

};  // namespace berberis
