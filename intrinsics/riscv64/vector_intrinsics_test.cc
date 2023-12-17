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

#include "xmmintrin.h"

#include <array>
#include <cstdint>
#include <tuple>

#include "berberis/intrinsics/vector_intrinsics.h"

namespace berberis::intrinsics {

namespace {

using UInt8 = uint8_t;
using UInt16 = uint16_t;
using UInt32 = uint32_t;
using UInt64 = uint64_t;

// Easily recognizable bit pattern for target register.
constexpr __m128i kUndisturbedResult = {0x5555'5555'5555'5555, 0x5555'5555'5555'5555};

TEST(VectorIntrinsics, Vaddvv) {
  auto Verify = [](auto Vaddvv, auto Vaddvvm, SIMD128Register arg2, auto result_to_check) {
    ASSERT_EQ(Vaddvv(0, 16, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check});
    ASSERT_EQ(Vaddvvm(0, 16, 0xffff, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check});
  };
  Verify(Vaddvv<UInt8, TailProcessing::kAgnostic>,
         Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0});
  Verify(Vaddvv<UInt8, TailProcessing::kAgnostic>,
         Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255});
  Verify(Vaddvv<UInt16, TailProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{0, 1, 0, 1, 0, 1, 0, 1},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000});
  Verify(Vaddvv<UInt16, TailProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{1, 0, 1, 0, 1, 0, 1, 0},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff});
  Verify(Vaddvv<UInt32, TailProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{0, 1, 0, 1},
         __v4su{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0x0000'0000});
  Verify(Vaddvv<UInt32, TailProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{1, 0, 1, 0},
         __v4su{0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff});
  Verify(Vaddvv<UInt64, TailProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{0, 1},
         __v2du{0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000});
  Verify(Vaddvv<UInt64, TailProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{1, 0},
         __v2du{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff});
}

TEST(VectorIntrinsics, Vaddvx) {
  auto Verify = [](auto Vaddvx, auto Vaddvxm, SIMD128Register arg1, auto result_to_check) {
    ASSERT_EQ(Vaddvx(0, 16, kUndisturbedResult, arg1, UInt8{1}), std::tuple{result_to_check});
    ASSERT_EQ(Vaddvxm(0, 16, 0xffff, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check});
  };
  Verify(Vaddvx<UInt8, TailProcessing::kAgnostic>,
         Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0});
  Verify(Vaddvx<UInt8, TailProcessing::kAgnostic>,
         Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255});
  Verify(Vaddvx<UInt16, TailProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000});
  Verify(Vaddvx<UInt16, TailProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff});
  Verify(Vaddvx<UInt32, TailProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{0xffff'fffe, 0xffff'ffff, 0xffff'fffe, 0xffff'ffff},
         __v4su{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0x0000'0000});
  Verify(Vaddvx<UInt32, TailProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{0xffff'ffff, 0xffff'fffe, 0xffff'ffff, 0xffff'fffe},
         __v4su{0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff});
  Verify(Vaddvx<UInt64, TailProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{0xffff'ffff'ffff'fffe, 0xffff'ffff'ffff'ffff},
         __v2du{0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000});
  Verify(Vaddvx<UInt64, TailProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'fffe},
         __v2du{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff});
}

TEST(VectorIntrinsics, VlArgForVv) {
  auto Verify = [](auto Vaddvva,
                   auto Vaddvvu,
                   auto Vaddvvmaa,
                   auto Vaddvvmau,
                   auto Vaddvvmua,
                   auto Vaddvvmuu,
                   SIMD128Register arg2,
                   auto result_to_check_agnostic,
                   auto result_to_check_undisturbed) {
    static_assert(
        std::is_same_v<decltype(result_to_check_agnostic), decltype(result_to_check_undisturbed)>);
    constexpr size_t kHalfLen =
        sizeof(result_to_check_agnostic) / sizeof(result_to_check_agnostic[0]) / 2;
    ASSERT_EQ(Vaddvva(0, kHalfLen, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check_agnostic});
    ASSERT_EQ(Vaddvvu(0, kHalfLen, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check_undisturbed});
    ASSERT_EQ(Vaddvvmaa(0, kHalfLen, 0xffff, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check_agnostic});
    ASSERT_EQ(Vaddvvmau(0, kHalfLen, 0xffff, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check_agnostic});
    ASSERT_EQ(Vaddvvmua(0, kHalfLen, 0xffff, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check_undisturbed});
    ASSERT_EQ(Vaddvvmuu(0, kHalfLen, 0xffff, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check_undisturbed});
  };
  Verify(Vaddvv<UInt8, TailProcessing::kAgnostic>,
         Vaddvv<UInt8, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v16qu{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55});
  Verify(Vaddvv<UInt8, TailProcessing::kAgnostic>,
         Vaddvv<UInt8, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v16qu{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55});
  Verify(Vaddvv<UInt16, TailProcessing::kAgnostic>,
         Vaddvv<UInt16, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{0, 1, 0, 1, 0, 1, 0, 1},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0x5555, 0x5555, 0x5555, 0x5555});
  Verify(Vaddvv<UInt16, TailProcessing::kAgnostic>,
         Vaddvv<UInt16, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{1, 0, 1, 0, 1, 0, 1, 0},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0x5555, 0x5555, 0x5555, 0x5555});
  Verify(Vaddvv<UInt32, TailProcessing::kAgnostic>,
         Vaddvv<UInt32, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{0, 1, 0, 1},
         __v4su{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
         __v4su{0xffff'ffff, 0x0000'0000, 0x5555'5555, 0x5555'5555});
  Verify(Vaddvv<UInt32, TailProcessing::kAgnostic>,
         Vaddvv<UInt32, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{1, 0, 1, 0},
         __v4su{0x0000'0000, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
         __v4su{0x0000'0000, 0xffff'ffff, 0x5555'5555, 0x5555'5555});
  Verify(Vaddvv<UInt64, TailProcessing::kAgnostic>,
         Vaddvv<UInt64, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{0, 1},
         __v2du{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
         __v2du{0xffff'ffff'ffff'ffff, 0x5555'5555'5555'5555});
  Verify(Vaddvv<UInt64, TailProcessing::kAgnostic>,
         Vaddvv<UInt64, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{1, 0},
         __v2du{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff},
         __v2du{0x0000'0000'0000'0000, 0x5555'5555'5555'5555});
}

TEST(VectorIntrinsics, VlArgForVx) {
  auto Verify = [](auto Vaddvxa,
                   auto Vaddvxu,
                   auto Vaddvxmaa,
                   auto Vaddvxmau,
                   auto Vaddvxmua,
                   auto Vaddvxmuu,
                   SIMD128Register arg1,
                   auto result_to_check_agnostic,
                   auto result_to_check_undisturbed) {
    static_assert(
        std::is_same_v<decltype(result_to_check_agnostic), decltype(result_to_check_undisturbed)>);
    constexpr size_t kHalfLen =
        sizeof(result_to_check_agnostic) / sizeof(result_to_check_agnostic[0]) / 2;
    ASSERT_EQ(Vaddvxa(0, kHalfLen, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check_agnostic});
    ASSERT_EQ(Vaddvxu(0, kHalfLen, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check_undisturbed});
    ASSERT_EQ(Vaddvxmaa(0, kHalfLen, 0xffff, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check_agnostic});
    ASSERT_EQ(Vaddvxmau(0, kHalfLen, 0xffff, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check_agnostic});
    ASSERT_EQ(Vaddvxmua(0, kHalfLen, 0xffff, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check_undisturbed});
    ASSERT_EQ(Vaddvxmuu(0, kHalfLen, 0xffff, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check_undisturbed});
  };
  Verify(Vaddvx<UInt8, TailProcessing::kAgnostic>,
         Vaddvx<UInt8, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v16qu{254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55});
  Verify(Vaddvx<UInt8, TailProcessing::kAgnostic>,
         Vaddvx<UInt8, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v16qu{255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55});
  Verify(Vaddvx<UInt16, TailProcessing::kAgnostic>,
         Vaddvx<UInt16, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0x5555, 0x5555, 0x5555, 0x5555});
  Verify(Vaddvx<UInt16, TailProcessing::kAgnostic>,
         Vaddvx<UInt16, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0x5555, 0x5555, 0x5555, 0x5555});
  Verify(Vaddvx<UInt32, TailProcessing::kAgnostic>,
         Vaddvx<UInt32, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{0xffff'fffe, 0xffff'ffff, 0xffff'fffe, 0xffff'ffff},
         __v4su{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
         __v4su{0xffff'ffff, 0x0000'0000, 0x5555'5555, 0x5555'5555});
  Verify(Vaddvx<UInt32, TailProcessing::kAgnostic>,
         Vaddvx<UInt32, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{0xffff'ffff, 0xffff'fffe, 0xffff'ffff, 0xffff'fffe},
         __v4su{0x0000'0000, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
         __v4su{0x0000'0000, 0xffff'ffff, 0x5555'5555, 0x5555'5555});
  Verify(Vaddvx<UInt64, TailProcessing::kAgnostic>,
         Vaddvx<UInt64, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{0xffff'ffff'ffff'fffe, 0xffff'ffff'ffff'ffff},
         __v2du{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
         __v2du{0xffff'ffff'ffff'ffff, 0x5555'5555'5555'5555});
  Verify(Vaddvx<UInt64, TailProcessing::kAgnostic>,
         Vaddvx<UInt64, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'fffe},
         __v2du{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff},
         __v2du{0x0000'0000'0000'0000, 0x5555'5555'5555'5555});
}


TEST(VectorIntrinsics, VmaskArgForVvvv) {
  auto Verify = [](auto Vaddvvmaa,
                   auto Vaddvvmau,
                   auto Vaddvvmua,
                   auto Vaddvvmuu,
                   SIMD128Register arg2,
                   auto result_to_check_agnostic_agnostic,
                   auto result_to_check_agnostic_undisturbed,
                   auto result_to_check_undisturbed_agnostic,
                   auto result_to_check_undisturbed_undisturbed) {
    static_assert(std::is_same_v<decltype(result_to_check_agnostic_agnostic),
                                 decltype(result_to_check_agnostic_undisturbed)>);
    static_assert(std::is_same_v<decltype(result_to_check_agnostic_agnostic),
                                 decltype(result_to_check_undisturbed_agnostic)>);
    static_assert(std::is_same_v<decltype(result_to_check_agnostic_agnostic),
                                 decltype(result_to_check_undisturbed_undisturbed)>);
    constexpr size_t kHalfLen = sizeof(result_to_check_agnostic_agnostic) /
                                sizeof(result_to_check_agnostic_agnostic[0]) / 2;
    ASSERT_EQ(Vaddvvmaa(0, kHalfLen, 0xfdda, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check_agnostic_agnostic});
    ASSERT_EQ(Vaddvvmau(0, kHalfLen, 0xfdda, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check_agnostic_undisturbed});
    ASSERT_EQ(Vaddvvmua(0, kHalfLen, 0xfdda, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check_undisturbed_agnostic});
    ASSERT_EQ(Vaddvvmuu(0, kHalfLen, 0xfdda, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check_undisturbed_undisturbed});
  };
  Verify(
      Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
      Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
      Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
      Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
      __v16qu{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
      __v16qu{255, 0, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255},
      __v16qu{0x55, 0, 0x55, 0, 255, 0x55, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255},
      __v16qu{255, 0, 255, 0, 255, 255, 255, 0, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55},
      __v16qu{0x55, 0, 0x55, 0, 255, 0x55, 255, 0, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55});
  Verify(
      Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
      Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
      Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
      Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
      __v16qu{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
      __v16qu{255, 255, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      __v16qu{0x55, 255, 0x55, 255, 0, 0x55, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      __v16qu{255, 255, 255, 255, 0, 255, 0, 255, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55},
      __v16qu{
          0x55, 255, 0x55, 255, 0, 0x55, 0, 255, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55});
  Verify(Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{0, 1, 0, 1, 0, 1, 0, 1},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0x5555, 0x0000, 0x5555, 0x0000, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0x5555, 0x5555, 0x5555, 0x5555},
         __v8hu{0x5555, 0x0000, 0x5555, 0x0000, 0x5555, 0x5555, 0x5555, 0x5555});
  Verify(Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{1, 0, 1, 0, 1, 0, 1, 0},
         __v8hu{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0x5555, 0xffff, 0x5555, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0xffff, 0xffff, 0xffff, 0xffff, 0x5555, 0x5555, 0x5555, 0x5555},
         __v8hu{0x5555, 0xffff, 0x5555, 0xffff, 0x5555, 0x5555, 0x5555, 0x5555});
  Verify(Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{0, 1, 0, 1},
         __v4su{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
         __v4su{0x5555'5555, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
         __v4su{0xffff'ffff, 0x0000'0000, 0x5555'5555, 0x5555'5555},
         __v4su{0x5555'5555, 0x0000'0000, 0x5555'5555, 0x5555'5555});
  Verify(Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{1, 0, 1, 0},
         __v4su{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
         __v4su{0x5555'5555, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
         __v4su{0xffff'ffff, 0xffff'ffff, 0x5555'5555, 0x5555'5555},
         __v4su{0x5555'5555, 0xffff'ffff, 0x5555'5555, 0x5555'5555});
  Verify(Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{0, 1},
         __v2du{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
         __v2du{0x5555'5555'5555'5555, 0xffff'ffff'ffff'ffff},
         __v2du{0xffff'ffff'ffff'ffff, 0x5555'5555'5555'5555},
         __v2du{0x5555'5555'5555'5555, 0x5555'5555'5555'5555});
  Verify(Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{1, 0},
         __v2du{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
         __v2du{0x5555'5555'5555'5555, 0xffff'ffff'ffff'ffff},
         __v2du{0xffff'ffff'ffff'ffff, 0x5555'5555'5555'5555},
         __v2du{0x5555'5555'5555'5555, 0x5555'5555'5555'5555});
}

TEST(VectorIntrinsics, VmaskArgForVvvx) {
  auto Verify = [](auto Vaddvxmaa,
                   auto Vaddvxmau,
                   auto Vaddvxmua,
                   auto Vaddvxmuu,
                   SIMD128Register arg1,
                   auto result_to_check_agnostic_agnostic,
                   auto result_to_check_agnostic_undisturbed,
                   auto result_to_check_undisturbed_agnostic,
                   auto result_to_check_undisturbed_undisturbed) {
    static_assert(std::is_same_v<decltype(result_to_check_agnostic_agnostic),
                                 decltype(result_to_check_agnostic_undisturbed)>);
    static_assert(std::is_same_v<decltype(result_to_check_agnostic_agnostic),
                                 decltype(result_to_check_undisturbed_agnostic)>);
    static_assert(std::is_same_v<decltype(result_to_check_agnostic_agnostic),
                                 decltype(result_to_check_undisturbed_undisturbed)>);
    constexpr size_t kHalfLen = sizeof(result_to_check_agnostic_agnostic) /
                                sizeof(result_to_check_agnostic_agnostic[0]) / 2;
    ASSERT_EQ(Vaddvxmaa(0, kHalfLen, 0xfdda, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check_agnostic_agnostic});
    ASSERT_EQ(Vaddvxmau(0, kHalfLen, 0xfdda, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check_agnostic_undisturbed});
    ASSERT_EQ(Vaddvxmua(0, kHalfLen, 0xfdda, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check_undisturbed_agnostic});
    ASSERT_EQ(Vaddvxmuu(0, kHalfLen, 0xfdda, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check_undisturbed_undisturbed});
  };
  Verify(
      Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
      Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
      Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
      Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
      __v16qu{254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255},
      __v16qu{255, 0, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255},
      __v16qu{0x55, 0, 0x55, 0, 255, 0x55, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255},
      __v16qu{255, 0, 255, 0, 255, 255, 255, 0, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55},
      __v16qu{0x55, 0, 0x55, 0, 255, 0x55, 255, 0, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55});
  Verify(
      Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
      Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
      Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
      Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
      __v16qu{255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254},
      __v16qu{255, 255, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      __v16qu{0x55, 255, 0x55, 255, 0, 0x55, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255},
      __v16qu{255, 255, 255, 255, 0, 255, 0, 255, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55},
      __v16qu{
          0x55, 255, 0x55, 255, 0, 0x55, 0, 255, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55});
  Verify(Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0x5555, 0x0000, 0x5555, 0x0000, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0x5555, 0x5555, 0x5555, 0x5555},
         __v8hu{0x5555, 0x0000, 0x5555, 0x0000, 0x5555, 0x5555, 0x5555, 0x5555});
  Verify(Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe},
         __v8hu{0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0x5555, 0xffff, 0x5555, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff},
         __v8hu{0xffff, 0xffff, 0xffff, 0xffff, 0x5555, 0x5555, 0x5555, 0x5555},
         __v8hu{0x5555, 0xffff, 0x5555, 0xffff, 0x5555, 0x5555, 0x5555, 0x5555});
  Verify(Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{0xffff'fffe, 0xffff'ffff, 0xffff'fffe, 0xffff'ffff},
         __v4su{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
         __v4su{0x5555'5555, 0x0000'0000, 0xffff'ffff, 0xffff'ffff},
         __v4su{0xffff'ffff, 0x0000'0000, 0x5555'5555, 0x5555'5555},
         __v4su{0x5555'5555, 0x0000'0000, 0x5555'5555, 0x5555'5555});
  Verify(Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{0xffff'ffff, 0xffff'fffe, 0xffff'ffff, 0xffff'fffe},
         __v4su{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
         __v4su{0x5555'5555, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
         __v4su{0xffff'ffff, 0xffff'ffff, 0x5555'5555, 0x5555'5555},
         __v4su{0x5555'5555, 0xffff'ffff, 0x5555'5555, 0x5555'5555});
  Verify(Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{0xffff'ffff'ffff'fffe, 0xffff'ffff'ffff'ffff},
         __v2du{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
         __v2du{0x5555'5555'5555'5555, 0xffff'ffff'ffff'ffff},
         __v2du{0xffff'ffff'ffff'ffff, 0x5555'5555'5555'5555},
         __v2du{0x5555'5555'5555'5555, 0x5555'5555'5555'5555});
  Verify(Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'fffe},
         __v2du{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'ffff},
         __v2du{0x5555'5555'5555'5555, 0xffff'ffff'ffff'ffff},
         __v2du{0xffff'ffff'ffff'ffff, 0x5555'5555'5555'5555},
         __v2du{0x5555'5555'5555'5555, 0x5555'5555'5555'5555});
}

TEST(VectorIntrinsics, VstartArgVv) {
  auto Verify = [](auto Vaddvva,
                   auto Vaddvvu,
                   auto Vaddvvmaa,
                   auto Vaddvvmau,
                   auto Vaddvvmua,
                   auto Vaddvvmuu,
                   SIMD128Register arg2,
                   auto result_to_check) {
    ASSERT_EQ(Vaddvva(1, 16, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check});
    ASSERT_EQ(Vaddvvu(1, 16, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check});
    ASSERT_EQ(Vaddvvmaa(1, 16, 0xffff, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check});
    ASSERT_EQ(Vaddvvmau(1, 16, 0xffff, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check});
    ASSERT_EQ(Vaddvvmua(1, 16, 0xffff, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check});
    ASSERT_EQ(Vaddvvmuu(1, 16, 0xffff, kUndisturbedResult, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check});
  };
  Verify(Vaddvv<UInt8, TailProcessing::kAgnostic>,
         Vaddvv<UInt8, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v16qu{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
         __v16qu{0x55, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0});
  Verify(Vaddvv<UInt8, TailProcessing::kAgnostic>,
         Vaddvv<UInt8, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v16qu{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
         __v16qu{0x55, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255});
  Verify(Vaddvv<UInt16, TailProcessing::kAgnostic>,
         Vaddvv<UInt16, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{0, 1, 0, 1, 0, 1, 0, 1},
         __v8hu{0x5555, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000});
  Verify(Vaddvv<UInt16, TailProcessing::kAgnostic>,
         Vaddvv<UInt16, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{1, 0, 1, 0, 1, 0, 1, 0},
         __v8hu{0x5555, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff});
  Verify(Vaddvv<UInt32, TailProcessing::kAgnostic>,
         Vaddvv<UInt32, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{0, 1, 0, 1},
         __v4su{0x5555'5555, 0x0000'0000, 0xffff'ffff, 0x0000'0000});
  Verify(Vaddvv<UInt32, TailProcessing::kAgnostic>,
         Vaddvv<UInt32, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{1, 0, 1, 0},
         __v4su{0x5555'5555, 0xffff'ffff, 0x0000'0000, 0xffff'ffff});
  Verify(Vaddvv<UInt64, TailProcessing::kAgnostic>,
         Vaddvv<UInt64, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{0, 1},
         __v2du{0x5555'5555'5555'5555, 0x0000'0000'0000'0000});
  Verify(Vaddvv<UInt64, TailProcessing::kAgnostic>,
         Vaddvv<UInt64, TailProcessing::kUndisturbed>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvvm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{1, 0},
         __v2du{0x5555'5555'5555'5555, 0xffff'ffff'ffff'ffff});
}

TEST(VectorIntrinsics, VstartArgVx) {
  auto Verify = [](auto Vaddvxa,
                   auto Vaddvxu,
                   auto Vaddvxmaa,
                   auto Vaddvxmau,
                   auto Vaddvxmua,
                   auto Vaddvxmuu,
                   SIMD128Register arg1,
                   auto result_to_check) {
    ASSERT_EQ(Vaddvxa(1, 16, kUndisturbedResult, arg1, UInt8{1}), std::tuple{result_to_check});
    ASSERT_EQ(Vaddvxu(1, 16, kUndisturbedResult, arg1, UInt8{1}), std::tuple{result_to_check});
    ASSERT_EQ(Vaddvxmaa(1, 16, 0xffff, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check});
    ASSERT_EQ(Vaddvxmau(1, 16, 0xffff, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check});
    ASSERT_EQ(Vaddvxmua(1, 16, 0xffff, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check});
    ASSERT_EQ(Vaddvxmuu(1, 16, 0xffff, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check});
  };
  Verify(Vaddvx<UInt8, TailProcessing::kAgnostic>,
         Vaddvx<UInt8, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v16qu{254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255},
         __v16qu{0x55, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0});
  Verify(Vaddvx<UInt8, TailProcessing::kAgnostic>,
         Vaddvx<UInt8, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt8, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v16qu{255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254},
         __v16qu{0x55, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255});
  Verify(Vaddvx<UInt16, TailProcessing::kAgnostic>,
         Vaddvx<UInt16, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff},
         __v8hu{0x5555, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000});
  Verify(Vaddvx<UInt16, TailProcessing::kAgnostic>,
         Vaddvx<UInt16, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt16, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v8hu{0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe},
         __v8hu{0x5555, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff});
  Verify(Vaddvx<UInt32, TailProcessing::kAgnostic>,
         Vaddvx<UInt32, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{0xffff'fffe, 0xffff'ffff, 0xffff'fffe, 0xffff'ffff},
         __v4su{0x5555'5555, 0x0000'0000, 0xffff'ffff, 0x0000'0000});
  Verify(Vaddvx<UInt32, TailProcessing::kAgnostic>,
         Vaddvx<UInt32, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt32, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v4su{0xffff'ffff, 0xffff'fffe, 0xffff'ffff, 0xffff'fffe},
         __v4su{0x5555'5555, 0xffff'ffff, 0x0000'0000, 0xffff'ffff});
  Verify(Vaddvx<UInt64, TailProcessing::kAgnostic>,
         Vaddvx<UInt64, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{0xffff'ffff'ffff'fffe, 0xffff'ffff'ffff'ffff},
         __v2du{0x5555'5555'5555'5555, 0x0000'0000'0000'0000});
  Verify(Vaddvx<UInt64, TailProcessing::kAgnostic>,
         Vaddvx<UInt64, TailProcessing::kUndisturbed>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kUndisturbed>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kAgnostic>,
         Vaddvxm<UInt64, TailProcessing::kUndisturbed, InactiveProcessing::kUndisturbed>,
         __v2du{0xffff'ffff'ffff'ffff, 0xffff'ffff'ffff'fffe},
         __v2du{0x5555'5555'5555'5555, 0xffff'ffff'ffff'ffff});
}

TEST(VectorIntrinsics, Vsubvv) {
  auto Verify = [](auto Vsubvv, auto Vsubvvm, SIMD128Register arg2, auto result_to_check) {
    ASSERT_EQ(Vsubvv(0, 16, kUndisturbedResult, __m128i{0, 0}, arg2), std::tuple{result_to_check});
    ASSERT_EQ(Vsubvvm(0, 16, 0xffff, kUndisturbedResult, __m128i{0, 0}, arg2),
              std::tuple{result_to_check});
  };
  Verify(Vsubvv<UInt8, TailProcessing::kAgnostic>,
         Vsubvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255});
  Verify(Vsubvv<UInt8, TailProcessing::kAgnostic>,
         Vsubvvm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0});
  Verify(Vsubvv<UInt16, TailProcessing::kAgnostic>,
         Vsubvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{0, 1, 0, 1, 0, 1, 0, 1},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff});
  Verify(Vsubvv<UInt16, TailProcessing::kAgnostic>,
         Vsubvvm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{1, 0, 1, 0, 1, 0, 1, 0},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000});
  Verify(Vsubvv<UInt32, TailProcessing::kAgnostic>,
         Vsubvvm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{0, 1, 0, 1},
         __v4su{0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff});
  Verify(Vsubvv<UInt64, TailProcessing::kAgnostic>,
         Vsubvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{0, 1},
         __v2du{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff});
  Verify(Vsubvv<UInt64, TailProcessing::kAgnostic>,
         Vsubvvm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{1, 0},
         __v2du{0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000});
}

TEST(VectorIntrinsics, Vsubvx) {
  auto Verify = [](auto Vsubvx, auto Vsubvxm, SIMD128Register arg1, auto result_to_check) {
    ASSERT_EQ(Vsubvx(0, 16, kUndisturbedResult, arg1, UInt8{1}), std::tuple{result_to_check});
    ASSERT_EQ(Vsubvxm(0, 16, 0xffff, kUndisturbedResult, arg1, UInt8{1}),
              std::tuple{result_to_check});
  };
  Verify(Vsubvx<UInt8, TailProcessing::kAgnostic>,
         Vsubvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255});
  Verify(Vsubvx<UInt8, TailProcessing::kAgnostic>,
         Vsubvxm<UInt8, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0});
  Verify(Vsubvx<UInt16, TailProcessing::kAgnostic>,
         Vsubvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{1, 0, 1, 0, 1, 0, 1, 0},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff});
  Verify(Vsubvx<UInt16, TailProcessing::kAgnostic>,
         Vsubvxm<UInt16, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{0, 1, 0, 1, 0, 1, 0, 1},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000});
  Verify(Vsubvx<UInt32, TailProcessing::kAgnostic>,
         Vsubvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{1, 0, 1, 0},
         __v4su{0x0000'0000, 0xffff'ffff, 0x0000'0000, 0xffff'ffff});
  Verify(Vsubvx<UInt32, TailProcessing::kAgnostic>,
         Vsubvxm<UInt32, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{0, 1, 0, 1},
         __v4su{0xffff'ffff, 0x0000'0000, 0xffff'ffff, 0x0000'0000});
  Verify(Vsubvx<UInt64, TailProcessing::kAgnostic>,
         Vsubvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{1, 0},
         __v2du{0x0000'0000'0000'0000, 0xffff'ffff'ffff'ffff});
  Verify(Vsubvx<UInt64, TailProcessing::kAgnostic>,
         Vsubvxm<UInt64, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{0, 1},
         __v2du{0xffff'ffff'ffff'ffff, 0x0000'0000'0000'0000});
}

}  // namespace

}  // namespace berberis::intrinsics
