/*
 * Copyright (C) 2024 The Android Open Source Project
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

#include "berberis/runtime_primitives/exec_region_anonymous.h"
#include "berberis/runtime_primitives/exec_region_elf_backed.h"

namespace berberis {

// For static executables we cannot use dlopen_ext.
// Use anonymous factory instead. Please do not use
// this outside of static tests.
ExecRegion ExecRegionElfBackedFactory::Create(size_t size) {
  return ExecRegionAnonymousFactory::Create(size);
}

}  // namespace berberis
