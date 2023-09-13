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

#ifndef BERBERIS_HEAVY_OPTIMIZER_RISCV64_HEAVY_OPTIMIZE_REGION_H_
#define BERBERIS_HEAVY_OPTIMIZER_RISCV64_HEAVY_OPTIMIZE_REGION_H_

#include <cstdint>
#include <tuple>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

std::tuple<GuestAddr, size_t> HeavyOptimizeRegion(GuestAddr pc, MachineCode* machine_code);

}  // namespace berberis

#endif /* BERBERIS_HEAVY_OPTIMIZER_RISCV64_HEAVY_OPTIMIZE_REGION_H_ */
