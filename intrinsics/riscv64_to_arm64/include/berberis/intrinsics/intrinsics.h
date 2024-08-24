/*
 * Copyright (C) 2024 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file excenaupt in compliance with the License.
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

#ifndef BERBERIS_INTRINSICS_RISCV64_TO_ARM64_INTRINSICS_H_
#define BERBERIS_INTRINSICS_RISCV64_TO_ARM64_INTRINSICS_H_

#include <cstdint>
#include <tuple>

#include "berberis/intrinsics/common/intrinsics.h"

namespace berberis::intrinsics {

template <typename Type0,
          bool kBool1,
          bool kBool2,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
std::tuple<Type0> AmoAdd(int64_t, Type0);
// Atomic and, like __atomic_fetch_and. Three template arguments: type, aq, rl..
template <typename Type0,
          bool kBool1,
          bool kBool2,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
std::tuple<Type0> AmoAnd(int64_t, Type0);
// Atomic maximum, like __atomic_fetch_max. Three template arguments: type, aq, rl..
template <typename Type0,
          bool kBool1,
          bool kBool2,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
std::tuple<Type0> AmoMax(int64_t, Type0);
// Atomic minimum, like __atomic_fetch_min. Three template arguments: type, aq, rl..
template <typename Type0,
          bool kBool1,
          bool kBool2,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
std::tuple<Type0> AmoMin(int64_t, Type0);
// Atomic or, like __atomic_fetch_or. Three template arguments: type, aq, rl..
template <typename Type0,
          bool kBool1,
          bool kBool2,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
std::tuple<Type0> AmoOr(int64_t, Type0);
// Atomic exchange, like __atomic_exchange_n. Three template arguments: type, aq, rl..
template <typename Type0,
          bool kBool1,
          bool kBool2,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
std::tuple<Type0> AmoSwap(int64_t, Type0);
// Atomic exclusive or, like __atomic_fetch_xor. Three template arguments: type, aq, rl..
template <typename Type0,
          bool kBool1,
          bool kBool2,
          enum PreferredIntrinsicsImplementation = kUseAssemblerImplementationIfPossible>
std::tuple<Type0> AmoXor(int64_t, Type0);

}  // namespace berberis::intrinsics

#include "berberis/intrinsics/intrinsics_atomics_impl.h"

#endif  // BERBERIS_INTRINSICS_RISCV64_TO_ARM64_INTRINSICS_H_
