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

TEST(VectorIntrinsics, Vaddvv) {
  auto Verify = [](auto Vaddvv, auto Vaddvvm, SIMD128Register arg2, auto result_to_check) {
    ASSERT_EQ(Vaddvv(0, 16, __m128i{0x55555555, 0x55555555}, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check});
    ASSERT_EQ(Vaddvvm(0, 16, 0xffff, __m128i{0x55555555, 0x55555555}, __m128i{-1, -1}, arg2),
              std::tuple{result_to_check});
  };
  Verify(Vaddvv<uint8_t, TailProcessing::kAgnostic>,
         Vaddvvm<uint8_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0});
  Verify(Vaddvv<uint8_t, TailProcessing::kAgnostic>,
         Vaddvvm<uint8_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255});
  Verify(Vaddvv<uint16_t, TailProcessing::kAgnostic>,
         Vaddvvm<uint16_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{0, 1, 0, 1, 0, 1, 0, 1},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000});
  Verify(Vaddvv<uint16_t, TailProcessing::kAgnostic>,
         Vaddvvm<uint16_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{1, 0, 1, 0, 1, 0, 1, 0},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff});
  Verify(Vaddvv<uint32_t, TailProcessing::kAgnostic>,
         Vaddvvm<uint32_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{0, 1, 0, 1},
         __v4su{0xffffffff, 0x00000000, 0xffffffff, 0x00000000});
  Verify(Vaddvv<uint32_t, TailProcessing::kAgnostic>,
         Vaddvvm<uint32_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{1, 0, 1, 0},
         __v4su{0x000000000, 0xffffffff, 0x00000000, 0xffffffff});
  Verify(Vaddvv<uint64_t, TailProcessing::kAgnostic>,
         Vaddvvm<uint64_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{0, 1},
         __v2du{0xffffffffffffffff, 0x0000000000000000});
  Verify(Vaddvv<uint64_t, TailProcessing::kAgnostic>,
         Vaddvvm<uint64_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{1, 0},
         __v2du{0x0000000000000000, 0xffffffffffffffff});
}

TEST(VectorIntrinsics, Vaddvx) {
  auto Verify = [](auto Vaddvx, auto Vaddvxm, SIMD128Register arg1, auto result_to_check) {
    ASSERT_EQ(Vaddvx(0, 16, __m128i{0x55555555, 0x55555555}, arg1, 1), std::tuple{result_to_check});
    ASSERT_EQ(Vaddvxm(0, 16, 0xffff, __m128i{0x55555555, 0x55555555}, arg1, 1),
              std::tuple{result_to_check});
  };
  Verify(Vaddvx<uint8_t, TailProcessing::kAgnostic>,
         Vaddvxm<uint8_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0});
  Verify(Vaddvx<uint8_t, TailProcessing::kAgnostic>,
         Vaddvxm<uint8_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255});
  Verify(Vaddvx<uint16_t, TailProcessing::kAgnostic>,
         Vaddvxm<uint16_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000});
  Verify(Vaddvx<uint16_t, TailProcessing::kAgnostic>,
         Vaddvxm<uint16_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe, 0xffff, 0xfffe},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff});
  Verify(Vaddvx<uint32_t, TailProcessing::kAgnostic>,
         Vaddvxm<uint32_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{0xfffffffe, 0xffffffff, 0xfffffffe, 0xffffffff},
         __v4su{0xffffffff, 0x00000000, 0xffffffff, 0x00000000});
  Verify(Vaddvx<uint32_t, TailProcessing::kAgnostic>,
         Vaddvxm<uint32_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{0xffffffff, 0xfffffffe, 0xffffffff, 0xfffffffe},
         __v4su{0x00000000, 0xffffffff, 0x00000000, 0xffffffff});
  Verify(Vaddvx<uint64_t, TailProcessing::kAgnostic>,
         Vaddvxm<uint64_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{0xfffffffffffffffe, 0xffffffffffffffff},
         __v2du{0xffffffffffffffff, 0x0000000000000000});
  Verify(Vaddvx<uint64_t, TailProcessing::kAgnostic>,
         Vaddvxm<uint64_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{0xffffffffffffffff, 0xfffffffffffffffe},
         __v2du{0x0000000000000000, 0xffffffffffffffff});
}

TEST(VectorIntrinsics, Vsubvv) {
  auto Verify = [](auto Vsubvv, auto Vsubvvm, SIMD128Register arg2, auto result_to_check) {
    ASSERT_EQ(Vsubvv(0, 16, __m128i{0x55555555, 0x55555555}, __m128i{0, 0}, arg2),
              std::tuple{result_to_check});
    ASSERT_EQ(Vsubvvm(0, 16, 0xffff, __m128i{0x55555555, 0x55555555}, __m128i{0, 0}, arg2),
              std::tuple{result_to_check});
  };
  Verify(Vsubvv<uint8_t, TailProcessing::kAgnostic>,
         Vsubvvm<uint8_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255});
  Verify(Vsubvv<uint8_t, TailProcessing::kAgnostic>,
         Vsubvvm<uint8_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0});
  Verify(Vsubvv<uint16_t, TailProcessing::kAgnostic>,
         Vsubvvm<uint16_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{0, 1, 0, 1, 0, 1, 0, 1},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff});
  Verify(Vsubvv<uint16_t, TailProcessing::kAgnostic>,
         Vsubvvm<uint16_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{1, 0, 1, 0, 1, 0, 1, 0},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000});
  Verify(Vsubvv<uint32_t, TailProcessing::kAgnostic>,
         Vsubvvm<uint32_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{0, 1, 0, 1},
         __v4su{0x00000000, 0xffffffff, 0x00000000, 0xffffffff});
  Verify(Vsubvv<uint64_t, TailProcessing::kAgnostic>,
         Vsubvvm<uint64_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{0, 1},
         __v2du{0x0000000000000000, 0xffffffffffffffff});
  Verify(Vsubvv<uint64_t, TailProcessing::kAgnostic>,
         Vsubvvm<uint64_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{1, 0},
         __v2du{0xffffffffffffffff, 0x0000000000000000});
}

TEST(VectorIntrinsics, Vsubvx) {
  auto Verify = [](auto Vsubvx, auto Vsubvxm, SIMD128Register arg1, auto result_to_check) {
    ASSERT_EQ(Vsubvx(0, 16, __m128i{0x55555555, 0x55555555}, arg1, 1), std::tuple{result_to_check});
    ASSERT_EQ(Vsubvxm(0, 16, 0xffff, __m128i{0x55555555, 0x55555555}, arg1, 1),
              std::tuple{result_to_check});
  };
  Verify(Vsubvx<uint8_t, TailProcessing::kAgnostic>,
         Vsubvxm<uint8_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
         __v16qu{0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255});
  Verify(Vsubvx<uint8_t, TailProcessing::kAgnostic>,
         Vsubvxm<uint8_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v16qu{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
         __v16qu{255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0});
  Verify(Vsubvx<uint16_t, TailProcessing::kAgnostic>,
         Vsubvxm<uint16_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{1, 0, 1, 0, 1, 0, 1, 0},
         __v8hu{0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff});
  Verify(Vsubvx<uint16_t, TailProcessing::kAgnostic>,
         Vsubvxm<uint16_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v8hu{0, 1, 0, 1, 0, 1, 0, 1},
         __v8hu{0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000});
  Verify(Vsubvx<uint32_t, TailProcessing::kAgnostic>,
         Vsubvxm<uint32_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{1, 0, 1, 0},
         __v4su{0x00000000, 0xffffffff, 0x00000000, 0xffffffff});
  Verify(Vsubvx<uint32_t, TailProcessing::kAgnostic>,
         Vsubvxm<uint32_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v4su{0, 1, 0, 1},
         __v4su{0xffffffff, 0x00000000, 0xffffffff, 0x00000000});
  Verify(Vsubvx<uint64_t, TailProcessing::kAgnostic>,
         Vsubvxm<uint64_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{1, 0},
         __v2du{0x0000000000000000, 0xffffffffffffffff});
  Verify(Vsubvx<uint64_t, TailProcessing::kAgnostic>,
         Vsubvxm<uint64_t, TailProcessing::kAgnostic, InactiveProcessing::kAgnostic>,
         __v2du{0, 1},
         __v2du{0xffffffffffffffff, 0x0000000000000000});
}

}  // namespace

}  // namespace berberis::intrinsics
