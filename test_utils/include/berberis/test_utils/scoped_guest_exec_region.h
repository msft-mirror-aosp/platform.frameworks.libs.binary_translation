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

#ifndef BERBERIS_TEST_UTILS_SCOPED_GUEST_EXEC_REGION_H_
#define BERBERIS_TEST_UTILS_SCOPED_GUEST_EXEC_REGION_H_

#include <cstddef>

#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

// Attention: We are setting and clearing executability for the whole page!
class ScopedGuestExecRegion {
 public:
  ScopedGuestExecRegion(GuestAddr pc, size_t size) : pc_(pc), size_(size) {
    GuestMapShadow::GetInstance()->SetExecutable(pc, size);
  }

  ScopedGuestExecRegion(const ScopedGuestExecRegion&) = delete;
  ScopedGuestExecRegion& operator=(const ScopedGuestExecRegion&) = delete;
  ScopedGuestExecRegion(const ScopedGuestExecRegion&&) = delete;
  ScopedGuestExecRegion& operator=(const ScopedGuestExecRegion&&) = delete;

  ~ScopedGuestExecRegion() { GuestMapShadow::GetInstance()->ClearExecutable(pc_, size_); }

 private:
  GuestAddr pc_;
  size_t size_;
};

}  // namespace berberis

#endif  // BERBERIS_TEST_UTILS_SCOPED_GUEST_EXEC_REGION_H_
