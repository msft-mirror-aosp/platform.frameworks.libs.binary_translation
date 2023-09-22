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

#include "gtest/gtest.h"

#include <cstdint>

#include "berberis/guest_abi/guest_abi.h"

namespace berberis {

namespace {

TEST(GuestAbi_riscv64, GuestArgumentInt8) {
  uint64_t value = 0;
  auto& param = *reinterpret_cast<GuestAbi::GuestArgument<int8_t>*>(&value);

  value = 0xffff'ffff'ffff'fff9U;
  EXPECT_EQ(param, -7);

  value = 7;
  EXPECT_EQ(param, 7);

  param = -123;
  EXPECT_EQ(value, 0xffff'ffff'ffff'ff85U);

  param = 127;
  EXPECT_EQ(value, 0x0000'0000'0000'007fU);
}

TEST(GuestAbi_riscv64, GuestArgumentUInt8) {
  uint64_t value = 0;
  auto& param = *reinterpret_cast<GuestAbi::GuestArgument<uint8_t>*>(&value);

  value = 0x0000'0000'0000'00f9U;
  EXPECT_EQ(param, 249);

  value = 7;
  EXPECT_EQ(param, 7);

  param = 123;
  EXPECT_EQ(value, 0x0000'0000'0000'007bU);

  param = 255;
  EXPECT_EQ(value, 0x0000'0000'0000'00ffU);
}

TEST(GuestAbi_riscv64, GuestArgumentUInt32) {
  uint64_t value = 0;
  auto& param = *reinterpret_cast<GuestAbi::GuestArgument<uint32_t>*>(&value);

  value = 0xffff'ffff'ffff'ffffU;
  EXPECT_EQ(param, 0xffff'ffffU);

  value = 7;
  EXPECT_EQ(param, 7U);

  param = 0xf123'4567U;
  EXPECT_EQ(value, 0xffff'ffff'f123'4567U);

  param += 1;
  EXPECT_EQ(value, 0xffff'ffff'f123'4568U);
}

}  // namespace

}  // namespace berberis
