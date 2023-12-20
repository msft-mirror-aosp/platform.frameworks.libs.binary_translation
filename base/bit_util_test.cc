/*
 * Copyright (C) 2021 The Android Open Source Project
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

#include "berberis/base/bit_util.h"

namespace berberis {

namespace {

static_assert(IsPowerOf2(sizeof(void*)));
static_assert(!IsPowerOf2(sizeof(void*) + 1));

static_assert(BitUtilLog2(1) == 0);
static_assert(BitUtilLog2(16) == 4);
static_assert(BitUtilLog2(sizeof(void*)) > 0);

static_assert(SatInt8{127} + SatInt8{1} == SatInt8{127});
static_assert(Int8{127} + Int8{1} == Int8{-128});

static_assert(SatUInt8{255} + SatUInt8{1} == SatUInt8{255});
static_assert(UInt8{255} + UInt8{1} == UInt8{0});

static_assert(SatInt8{-128} - SatInt8{1} == SatInt8{-128});
static_assert(Int8{-128} - Int8{1} == Int8{127});

static_assert(SatUInt8{0} - SatUInt8{1} == SatUInt8{0});
static_assert(UInt8{0} - UInt8{1} == UInt8{255});

static_assert(SatInt8{-128} * SatInt8{-128} == SatInt8{127});
static_assert(SatInt8{-128} * SatInt8{127} == SatInt8{-128});
static_assert(SatInt8{127} * SatInt8{-128} == SatInt8{-128});
static_assert(SatInt8{127} * SatInt8{127} == SatInt8{127});
static_assert(Int8{-128} * Int8{-128} == Int8{0});
static_assert(Int8{-128} * Int8{127} == Int8{-128});
static_assert(Int8{127} * Int8{-128} == Int8{-128});
static_assert(Int8{127} * Int8{127} == Int8{1});

static_assert(SatUInt8{255} * SatUInt8{255} == SatUInt8{255});
static_assert(UInt8{255} * UInt8{255} == UInt8{1});

static_assert(SatInt8{-128} / SatInt8{-1} == SatInt8{127});
static_assert(Int8{-128} / Int8{-1} == Int8{-128});

static_assert(SatUInt8{255} / SatUInt8{1} == SatUInt8{255});
static_assert(UInt8{255} / UInt8{1} == UInt8{255});

static_assert((Int8{123} << Int8{8}) == Int8{123});
static_assert((Int8{123} << Int8{65}) == Int8{-10});

static_assert((UInt8{123} << UInt8{8}) == UInt8{123});
static_assert((UInt8{123} << UInt8{65}) == UInt8{246});

static_assert((Int8{123} >> Int8{8}) == Int8{123});
static_assert((Int8{123} >> Int8{65}) == Int8{61});

static_assert((UInt8{123} >> UInt8{8}) == UInt8{123});
static_assert((UInt8{123} >> UInt8{65}) == UInt8{61});

static_assert(SatInt8{1} == SatInt8{Int8{1}});

// Verify that types are correctly expaned when needed.
// Note: attept to use signed and unsigned types in the same expression
// or mix saturating types and wrapping types trigger a compile-time error.
static_assert(SatInt16{1} + SatInt8{1} == SatInt16{2});
static_assert(Int16{1} + Int8{1} == Int16{2});

static_assert(SatInt8{1} + SatInt32{1} == SatInt32{2});
static_assert(Int8{1} + Int32{1} == Int32{2});

// Note: shifts use type of first operand to determine the result type.
// Wrapping also depends on the size of left operand only.
static_assert((Int16{1} << Int8{8}) == Int16{256});
static_assert((Int8{1} << Int16{8}) == Int8{1});

static_assert(MaybeTruncateTo<SatInt8>(SatInt8{127}) == SatInt8{127});
static_assert(MaybeTruncateTo<SatInt8>(SatInt16{32767}) == SatInt8{-1});
static_assert(TruncateTo<SatInt8>(SatInt16{32767}) == SatInt8{-1});
static_assert(MaybeTruncateTo<SatInt8>(SatInt8{-128}) == SatInt8{-128});
static_assert(MaybeTruncateTo<SatInt8>(SatInt16{-32768}) == SatInt8{0});
static_assert(TruncateTo<SatInt8>(SatInt16{-32768}) == SatInt8{0});

static_assert(MaybeTruncateTo<Int8>(Int8{127}) == Int8{127});
static_assert(MaybeTruncateTo<Int8>(Int16{32767}) == Int8{-1});
static_assert(TruncateTo<Int8>(Int16{32767}) == Int8{-1});
static_assert(MaybeTruncateTo<Int8>(Int8{-128}) == Int8{-128});
static_assert(MaybeTruncateTo<Int8>(Int16{-32768}) == Int8{0});
static_assert(TruncateTo<Int8>(Int16{-32768}) == Int8{0});

// BitCastToSigned and BitCastToUnigned don't change the bits of integer, they just treat them differently.
static_assert(BitCastToSigned(SatUInt8{128}) == SatInt8{-128});
static_assert(BitCastToSigned(UInt8{128}) == Int8{-128});
static_assert(BitCastToUnigned(SatInt8{-128}) == SatUInt8{128});
static_assert(BitCastToUnigned(Int8{-128}) == UInt8{128});

static_assert(std::is_same_v<Int16, Int16::SignedType>);
static_assert(std::is_same_v<Int16, UInt16::SignedType>);
static_assert(std::is_same_v<UInt16, Int16::UnsignedType>);
static_assert(std::is_same_v<UInt16, UInt16::UnsignedType>);

static_assert(std::is_same_v<SatInt16, SatInt16::SignedType>);
static_assert(std::is_same_v<SatInt16, SatUInt16::SignedType>);
static_assert(std::is_same_v<SatUInt16, SatInt16::UnsignedType>);
static_assert(std::is_same_v<SatUInt16, SatUInt16::UnsignedType>);

}  // namespace

}  // namespace berberis
