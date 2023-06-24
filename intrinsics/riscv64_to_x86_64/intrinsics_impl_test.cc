/*
 * Copyright (C) 2014 The Android Open Source Project
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

#include "berberis/intrinsics/intrinsics.h"

namespace berberis::intrinsics {

namespace {

TEST(Intrinsics, FSgnjS) {
  EXPECT_EQ(std::tuple{Float32{1.0f}}, FSgnjS(Float32{1.0f}, Float32{2.0f}));
  EXPECT_EQ(std::tuple{Float32{1.0f}}, FSgnjS(Float32{-1.0f}, Float32{2.0f}));
  EXPECT_EQ(std::tuple{Float32{-1.0f}}, FSgnjS(Float32{1.0f}, Float32{-2.0f}));
  EXPECT_EQ(std::tuple{Float32{-1.0f}}, FSgnjS(Float32{-1.0f}, Float32{-2.0f}));
}

TEST(Intrinsics, FSgnjD) {
  EXPECT_EQ(std::tuple{Float64{1.0}}, FSgnjD(Float64{1.0}, Float64{2.0}));
  EXPECT_EQ(std::tuple{Float64{1.0}}, FSgnjD(Float64{-1.0}, Float64{2.0}));
  EXPECT_EQ(std::tuple{Float64{-1.0}}, FSgnjD(Float64{1.0}, Float64{-2.0}));
  EXPECT_EQ(std::tuple{Float64{-1.0}}, FSgnjD(Float64{-1.0}, Float64{-2.0}));
}

TEST(Intrinsics, FSgnjnS) {
  EXPECT_EQ(std::tuple{Float32{-1.0f}}, FSgnjnS(Float32{1.0f}, Float32{2.0f}));
  EXPECT_EQ(std::tuple{Float32{-1.0f}}, FSgnjnS(Float32{-1.0f}, Float32{2.0f}));
  EXPECT_EQ(std::tuple{Float32{1.0f}}, FSgnjnS(Float32{1.0f}, Float32{-2.0f}));
  EXPECT_EQ(std::tuple{Float32{1.0f}}, FSgnjnS(Float32{-1.0f}, Float32{-2.0f}));
}

TEST(Intrinsics, FSgnjnD) {
  EXPECT_EQ(std::tuple{Float64{-1.0}}, FSgnjnD(Float64{1.0}, Float64{2.0}));
  EXPECT_EQ(std::tuple{Float64{-1.0}}, FSgnjnD(Float64{-1.0}, Float64{2.0}));
  EXPECT_EQ(std::tuple{Float64{1.0}}, FSgnjnD(Float64{1.0}, Float64{-2.0}));
  EXPECT_EQ(std::tuple{Float64{1.0}}, FSgnjnD(Float64{-1.0}, Float64{-2.0}));
}
TEST(Intrinsics, FSgnjxS) {
  EXPECT_EQ(std::tuple{Float32{1.0f}}, FSgnjxS(Float32{1.0f}, Float32{2.0f}));
  EXPECT_EQ(std::tuple{Float32{-1.0f}}, FSgnjxS(Float32{-1.0f}, Float32{2.0f}));
  EXPECT_EQ(std::tuple{Float32{-1.0f}}, FSgnjxS(Float32{1.0f}, Float32{-2.0f}));
  EXPECT_EQ(std::tuple{Float32{1.0f}}, FSgnjxS(Float32{-1.0f}, Float32{-2.0f}));
}

TEST(Intrinsics, FSgnjxD) {
  EXPECT_EQ(std::tuple{Float64{1.0}}, FSgnjxD(Float64{1.0}, Float64{2.0}));
  EXPECT_EQ(std::tuple{Float64{-1.0}}, FSgnjxD(Float64{-1.0}, Float64{2.0}));
  EXPECT_EQ(std::tuple{Float64{-1.0}}, FSgnjxD(Float64{1.0}, Float64{-2.0}));
  EXPECT_EQ(std::tuple{Float64{1.0}}, FSgnjxD(Float64{-1.0}, Float64{-2.0}));
}

}  // namespace

}  // namespace berberis::intrinsics