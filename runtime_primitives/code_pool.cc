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

#include "berberis/runtime_primitives/code_pool.h"

#include <algorithm>
#include <cstring>
#include <mutex>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/bit_util.h"
#include "berberis/base/exec_region.h"
#include "berberis/base/exec_region_anonymous.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

template <typename ExecRegionFactory>
HostCode CodePool<ExecRegionFactory>::Add(MachineCode* code) {
  std::lock_guard<std::mutex> lock(mutex_);

  uint32_t size = code->install_size();

  // This is the start of a generated code region which is always a branch
  // target. Align on 16-bytes as recommended by Intel.
  // TODO(b/232598137) Extract this into host specified behavior.
  current_address_ = AlignUp(current_address_, 16);

  if (exec_.end() < current_address_ + size) {
    exec_.Detach();
    exec_ = ExecRegionFactory::Create(std::max(size, ExecRegionFactory::kExecRegionSize));
    current_address_ = exec_.begin();
  }

  const uint8_t* result = current_address_;
  current_address_ += size;

  code->Install(&exec_, result, &recovery_map_);
  return result;
}

template <typename ExecRegionFactory>
uintptr_t CodePool<ExecRegionFactory>::FindRecoveryCode(uintptr_t fault_addr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = recovery_map_.find(fault_addr);
  if (it != recovery_map_.end()) {
    return it->second;
  }
  return 0;
}

void* DataPool::AddRaw(const void* ptr, uint32_t size) {
  void* result;
  // Take the lock to allocate in arena.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    result = arena_.Alloc(size, /* align = */ 16);
  }
  memcpy(result, ptr, size);
  return result;
}

// Specialize supported exec regions for CodePool
template class CodePool<ExecRegionAnonymousFactory>;
#if defined(__BIONIC__)
template class CodePool<ExecRegionElfBackedFactory>;
#endif

CodePool<ExecRegionAnonymousFactory>* GetDefaultCodePoolInstance() {
  static CodePool<ExecRegionAnonymousFactory> g_code_pool;
  return &g_code_pool;
}

#if defined(__BIONIC__)
CodePool<ExecRegionElfBackedFactory>* GetFunctionWrapperCodePoolInstance() {
  static CodePool<ExecRegionElfBackedFactory> g_code_pool;
  return &g_code_pool;
}
#else
CodePool<ExecRegionAnonymousFactory>* GetFunctionWrapperCodePoolInstance() {
  return GetDefaultCodePoolInstance();
}
#endif

DataPool* DataPool::GetInstance() {
  static DataPool g_data_pool;
  return &g_data_pool;
}

}  // namespace berberis
