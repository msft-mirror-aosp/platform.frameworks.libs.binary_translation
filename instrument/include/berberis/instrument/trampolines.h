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

#ifndef BERBERIS_INSTRUMENT_TRAMPOLINES_H_
#define BERBERIS_INSTRUMENT_TRAMPOLINES_H_

#include "berberis/instrument/instrument.h"

#include "berberis/guest_state/guest_state_opaque.h"

namespace berberis {

inline constexpr bool kInstrumentTrampolines = false;

using OnTrampolineFunc = void (*)(ThreadState*, const void*);

OnTrampolineFunc GetOnTrampolineCall(const char* name);
OnTrampolineFunc GetOnTrampolineReturn(const char* name);

}  // namespace berberis

#endif  // BERBERIS_INSTRUMENT_TRAMPOLINES_H_
