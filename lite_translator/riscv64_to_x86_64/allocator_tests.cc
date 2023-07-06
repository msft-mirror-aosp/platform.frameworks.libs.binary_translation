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

#include <optional>

#include "allocator.h"

namespace berberis {

namespace {

TEST(AllocatorTest, Allocator) {
  Allocator<x86_64::Assembler::Register> allocator;
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::rbx);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::rsi);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::rdi);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r8);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r9);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r10);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r11);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r12);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r13);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r14);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r15);
  EXPECT_EQ(allocator.Alloc(), std::nullopt);
}

TEST(AllocatorTest, AllocTemp) {
  Allocator<x86_64::Assembler::Register> allocator;
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::rbx);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::rsi);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::rdi);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r8);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r9);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r15);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r14);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r13);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r12);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r11);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r10);
  EXPECT_EQ(allocator.AllocTemp(), std::nullopt);
  allocator.FreeTemps();
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r15);
}

TEST(AllocatorTest, SeparateMappedRegsAndTempRegs) {
  Allocator<x86_64::Assembler::Register> allocator;
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::rbx);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::rsi);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::rdi);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r8);
  EXPECT_EQ(allocator.Alloc().value(), x86_64::Assembler::r9);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r15);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r14);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r13);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r12);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r11);
  EXPECT_EQ(allocator.AllocTemp().value(), x86_64::Assembler::r10);
  allocator.FreeTemps();
  EXPECT_EQ(allocator.Alloc(), std::nullopt);
}

}  // namespace

}  // namespace berberis
