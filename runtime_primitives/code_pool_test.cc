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

#include "berberis/runtime_primitives/code_pool.h"

namespace berberis {

namespace {

TEST(DataPool, Smoke) {
  DataPool data_pool;
  static uint32_t kConst1 = 0x12345678;
  static uint32_t kConst2 = 0x87654321;
  uint32_t kVar = kConst2;
  uint32_t* ptr = data_pool.Add(kVar);
  EXPECT_EQ(kConst2, *ptr);
  *ptr = kConst1;
  EXPECT_EQ(kConst1, *ptr);
}

}  // namespace

}  // namespace berberis
