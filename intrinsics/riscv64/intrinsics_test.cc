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
#include <tuple>

#include "berberis/intrinsics/intrinsics.h"

namespace berberis::intrinsics {

namespace {

TEST(Intrinsics, Div) {
  ASSERT_EQ(std::get<0>(Div<int8_t>(int8_t{-128}, int8_t{0})), int8_t{-1});
  ASSERT_EQ(std::get<0>(Div<int8_t>(int8_t{-128}, int8_t{-1})), int8_t{-128});
  ASSERT_EQ(std::get<0>(Div<int8_t>(int8_t{-128}, int8_t{-2})), int8_t{64});
  ASSERT_EQ(std::get<0>(Div<uint8_t>(uint8_t{128}, uint8_t{0})), uint8_t{255});
  ASSERT_EQ(std::get<0>(Div<uint8_t>(uint8_t{128}, uint8_t{1})), uint8_t{128});
  ASSERT_EQ(std::get<0>(Div<uint8_t>(uint8_t{128}, uint8_t{2})), uint8_t{64});
  ASSERT_EQ(std::get<0>(Div<int16_t>(int16_t{-32768}, int16_t{0})), int16_t{-1});
  ASSERT_EQ(std::get<0>(Div<int16_t>(int16_t{-32768}, int16_t{-1})), int16_t{-32768});
  ASSERT_EQ(std::get<0>(Div<int16_t>(int16_t{-32768}, int16_t{-2})), int16_t{16384});
  ASSERT_EQ(std::get<0>(Div<uint16_t>(uint16_t{32768}, uint16_t{0})), uint16_t{65535});
  ASSERT_EQ(std::get<0>(Div<uint16_t>(uint16_t{32768}, uint16_t{1})), uint16_t{32768});
  ASSERT_EQ(std::get<0>(Div<uint16_t>(uint16_t{32768}, uint16_t{2})), uint16_t{16384});
  ASSERT_EQ(std::get<0>(Div<int32_t>(int32_t{-2147483648}, int32_t{0})), int32_t{-1});
  ASSERT_EQ(std::get<0>(Div<int32_t>(int32_t{-2147483648}, int32_t{-1})), int32_t{-2147483648});
  ASSERT_EQ(std::get<0>(Div<int32_t>(int32_t{-2147483648}, int32_t{-2})), int32_t{1073741824});
  ASSERT_EQ(std::get<0>(Div<uint32_t>(uint32_t{2147483648}, uint32_t{0})), uint32_t{4294967295});
  ASSERT_EQ(std::get<0>(Div<uint32_t>(uint32_t{2147483648}, uint32_t{1})), uint32_t{2147483648});
  ASSERT_EQ(std::get<0>(Div<uint32_t>(uint32_t{2147483648}, uint32_t{2})), uint32_t{1073741824});
  ASSERT_EQ(std::get<0>(Div<int64_t>(int64_t{-9223372036854775807 - 1}, int64_t{0})), int64_t{-1});
  ASSERT_EQ(std::get<0>(Div<int64_t>(int64_t{-9223372036854775807 - 1}, int64_t{-1})),
            int64_t{-9223372036854775807 - 1});
  ASSERT_EQ(std::get<0>(Div<int64_t>(int64_t{-9223372036854775807 - 1}, int64_t{-2})),
            int64_t{4611686018427387904});
  ASSERT_EQ(std::get<0>(Div<uint64_t>(uint64_t{9223372036854775808U}, uint64_t{0})),
            uint64_t{18446744073709551615U});
  ASSERT_EQ(std::get<0>(Div<uint64_t>(uint64_t{9223372036854775808U}, uint64_t{1})),
            uint64_t{9223372036854775808U});
  ASSERT_EQ(std::get<0>(Div<uint64_t>(uint64_t{9223372036854775808U}, uint64_t{2})),
            uint64_t{4611686018427387904});
}

}  // namespace

}  // namespace berberis::intrinsics
