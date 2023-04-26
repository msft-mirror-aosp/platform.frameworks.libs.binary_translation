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

#ifndef BERBERIS_RUNTIME_PRIMITIVES_CODE_POOL_H_
#define BERBERIS_RUNTIME_PRIMITIVES_CODE_POOL_H_

#include <cstdint>
#include <mutex>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/arena_alloc.h"
#include "berberis/base/exec_region.h"
#include "berberis/base/exec_region_anonymous.h"
#include "berberis/runtime_primitives/host_code.h"

#if defined(__BIONIC__)
#include "berberis/base/exec_region_elf_backed.h"
#endif

namespace berberis {

// Code pool is an arena used to store fragments of generated code.
// TODO(b/232598137): Consider freeing allocated regions.
template <typename ExecRegionConstructor>
class CodePool {
 public:
  CodePool() = default;

  // Not copyable or movable
  CodePool(const CodePool&) = delete;
  CodePool& operator=(const CodePool&) = delete;
  CodePool(CodePool&&) = delete;
  CodePool& operator=(CodePool&&) = delete;

  [[nodiscard]] HostCode Add(MachineCode* code);

  [[nodiscard]] uintptr_t FindRecoveryCode(uintptr_t fault_addr) const;

 private:
  ExecRegion exec_;
  const uint8_t* current_address_ = nullptr;
  // TODO(b/232598137): have recovery map for each region instead!
  RecoveryMap recovery_map_;
  mutable std::mutex mutex_;
};

// Stored data for generated code.
class DataPool {
 public:
  // Returns default data pool.
  static DataPool* GetInstance();

  DataPool() = default;

  template <typename T>
  T* Add(const T& v) {
    return reinterpret_cast<T*>(AddRaw(&v, sizeof(T)));
  }

  void* AddRaw(const void* ptr, uint32_t size);

 private:
  Arena arena_;
  std::mutex mutex_;
};

// Returns default code pool.
[[nodiscard]] CodePool<ExecRegionAnonymousFactory>* GetDefaultCodePoolInstance();

#if defined(__BIONIC__)
[[nodiscard]] CodePool<ExecRegionElfBackedFactory>* GetFunctionWrapperCodePoolInstance();
#else
[[nodiscard]] CodePool<ExecRegionAnonymousFactory>* GetFunctionWrapperCodePoolInstance();
#endif

}  // namespace berberis

#endif  // BERBERIS_RUNTIME_PRIMITIVES_CODE_POOL_H_
