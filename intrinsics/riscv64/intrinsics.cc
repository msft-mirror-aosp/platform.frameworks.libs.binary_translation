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

#include "berberis/intrinsics/intrinsics.h"
#include "berberis/intrinsics/guest_fp_flags.h"

#include <cstdint>
#include <tuple>

namespace berberis::intrinsics {

std::tuple<uint64_t> Bclri(uint64_t src, uint8_t imm) {
  return {src & ~(uint64_t{1} << imm)};
}

std::tuple<uint64_t> Bexti(uint64_t src, uint8_t imm) {
  return {(src >> imm) & uint64_t{1}};
}

std::tuple<uint64_t> Binvi(uint64_t src, uint8_t imm) {
  return {src ^ (uint64_t{1} << imm)};
}

std::tuple<uint64_t> Bseti(uint64_t src, uint8_t imm) {
  return {src | (uint64_t{1} << imm)};
}

// TODO(b/260725458): stop using __builtin_popcount after C++20 would become available.
template <>
std::tuple<int64_t> Cpop<int32_t>(int32_t src) {
  return {__builtin_popcount(src)};
}

// TODO(b/260725458): stop using __builtin_popcountll after C++20 would become available.
template <>
std::tuple<int64_t> Cpop<int64_t>(int64_t src) {
  return {__builtin_popcountll(src)};
}

void FeSetRound(uint64_t rm) {
  if (rm <= FPFlags::RM_MAX) {
    std::fesetround(intrinsics::ToHostRoundingMode(rm));
  }
}

std::tuple<uint64_t> Slliuw(uint32_t src, uint8_t imm) {
  return {uint64_t{src} << imm};
}

}  // namespace berberis::intrinsics