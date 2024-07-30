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

#include "gtest/gtest.h"

#include "sys/mman.h"

#include <cstdint>

#include "berberis/base/maps_snapshot.h"

namespace berberis {

namespace {

void Foo() {};

TEST(MapsSnapshot, Basic) {
  auto* maps_snapshot = MapsSnapshot::GetInstance();

  maps_snapshot->ClearForTesting();

  // No mappings can be found before snapshot is taken by Update().
  auto no_mappings_result = maps_snapshot->FindMappedObjectName(reinterpret_cast<uintptr_t>(&Foo));
  ASSERT_FALSE(no_mappings_result.has_value());

  maps_snapshot->Update();

  auto result = maps_snapshot->FindMappedObjectName(reinterpret_cast<uintptr_t>(&Foo));
  ASSERT_TRUE(result.has_value());
  ASSERT_FALSE(result.value().empty());
}

TEST(MapsSnapshot, AnonymousMapping) {
  auto* maps_snapshot = MapsSnapshot::GetInstance();

  void* addr = mmap(nullptr, 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
  ASSERT_NE(addr, MAP_FAILED);
  maps_snapshot->Update();
  auto result = maps_snapshot->FindMappedObjectName(reinterpret_cast<uintptr_t>(addr));
  munmap(addr, 4096);

  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result.value().empty());
}

}  // namespace

}  // namespace berberis
