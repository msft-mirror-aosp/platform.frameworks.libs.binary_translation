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

#include "guest_signal_action.h"

#include "berberis/base/host_signal.h"
#include "berberis/base/macros.h"
#include "berberis/guest_os_primitives/guest_signal.h"

namespace berberis {

bool GuestSignalAction::Change(int sig,
                               const Guest_sigaction* new_sa,
                               host_sa_sigaction_t claimed_host_sa_sigaction,
                               Guest_sigaction* old_sa,
                               int* error) {
  // TODO(b/283499233): Implement.
  UNUSED(sig, new_sa, claimed_host_sa_sigaction, old_sa, error);
  return false;
}

}  // namespace berberis