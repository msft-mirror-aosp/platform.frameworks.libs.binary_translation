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

#include <tuple>

namespace berberis {

namespace {

static_assert(IsPowerOf2(sizeof(void*)));
static_assert(!IsPowerOf2(sizeof(void*) + 1));

static_assert(IsPowerOf2(RawInt8(4)));
static_assert(IsPowerOf2(SatInt8(4)));
static_assert(IsPowerOf2(Int8(4)));

static_assert(AlignUp(6, 4) == 8);
static_assert(AlignUp<4>(6) == 8);
static_assert(AlignUp<4>(RawInt8(6)) == RawInt8(8));
static_assert(AlignUp<4>(SatInt8(6)) == SatInt8(8));
static_assert(AlignUp<4>(Int8(6)) == Int8(8));

static_assert(AlignDown(6, 4) == 4);
static_assert(AlignDown<4>(6) == 4);
static_assert(AlignDown<4>(RawInt8(6)) == RawInt8(4));
static_assert(AlignDown<4>(SatInt8(6)) == SatInt8(4));
static_assert(AlignDown<4>(Int8(6)) == Int8(4));

static_assert(IsAligned(6, 2));
static_assert(!IsAligned(6, 4));
static_assert(IsAligned<2>(6));
static_assert(!IsAligned<4>(6));
static_assert(IsAligned<2>(RawInt8(6)));
static_assert(!IsAligned<4>(RawInt8(6)));
static_assert(IsAligned<2>(SatInt8(6)));
static_assert(!IsAligned<4>(SatInt8(6)));
static_assert(IsAligned<2>(Int8(6)));
static_assert(!IsAligned<4>(Int8(6)));

static_assert(BitUtilLog2(1) == 0);
static_assert(BitUtilLog2(16) == 4);
static_assert(BitUtilLog2(sizeof(void*)) > 0);

static_assert(CountRZero(~uint32_t{1}) == 1);
static_assert(CountRZero(RawInt32{~UInt32{1}}) == RawInt32{1});
static_assert(CountRZero(SatUInt32{~Int32{1}}) == SatUInt32{1});
static_assert(CountRZero(~UInt32{1}) == UInt32{1});
static_assert(CountRZero(~uint64_t{1}) == 1);
static_assert(CountRZero(RawInt64{~UInt64{1}}) == RawInt64{1});
static_assert(CountRZero(SatUInt64{~Int64{1}}) == SatUInt64{1});
static_assert(CountRZero(~UInt64{1}) == UInt64{1});
#if defined(__x86_64__)
static_assert(CountRZero(~static_cast<unsigned __int128>(1) << 64) == 65);
static_assert(CountRZero(RawInt128{~UInt128{1}}) == RawInt128{1});
static_assert(CountRZero(SatUInt128{~Int128{1}}) == SatUInt128{1});
static_assert(CountRZero(~UInt128{1} << UInt128{64}) == UInt128{65});
#endif

static_assert(Popcount(~uint32_t{1}) == 31);
static_assert(Popcount(RawInt32{~UInt32{1}}) == RawInt32{31});
static_assert(Popcount(SatUInt32{~Int32{1}}) == SatUInt32{31});
static_assert(Popcount(~UInt32{1}) == UInt32{31});
static_assert(Popcount(~uint64_t{1}) == 63);
static_assert(Popcount(RawInt64{~UInt64{1}}) == RawInt64{63});
static_assert(Popcount(SatUInt64{~Int64{1}}) == SatUInt64{63});
static_assert(Popcount(~UInt64{1}) == UInt64{63});
#if defined(__x86_64__)
static_assert(Popcount(~static_cast<unsigned __int128>(1)) == 127);
static_assert(Popcount(RawInt128{~UInt128{1}}) == RawInt128{127});
static_assert(Popcount(SatUInt128{~Int128{1}}) == SatUInt128{127});
static_assert(Popcount(~UInt128{1}) == UInt128{127});
#endif

static_assert(Add(SatInt8{126}, SatInt8{1}) == std::tuple{SatInt8{127}, false});
static_assert(Add(SatInt8{127}, SatInt8{1}) == std::tuple{SatInt8{127}, true});
static_assert(SatInt8{127} + SatInt8{1} == SatInt8{127});
static_assert(Int8{127} + Int8{1} == Int8{-128});

static_assert(Add(SatUInt8{254}, SatUInt8{1}) == std::tuple{SatUInt8{255}, false});
static_assert(Add(SatUInt8{255}, SatUInt8{1}) == std::tuple{SatUInt8{255}, true});
static_assert(SatUInt8{255} + SatUInt8{1} == SatUInt8{255});
static_assert(UInt8{255} + UInt8{1} == UInt8{0});

static_assert(Sub(SatInt8{-127}, SatInt8{1}) == std::tuple{SatInt8{-128}, false});
static_assert(Sub(SatInt8{-128}, SatInt8{1}) == std::tuple{SatInt8{-128}, true});
static_assert(SatInt8{-128} - SatInt8{1} == SatInt8{-128});
static_assert(Int8{-128} - Int8{1} == Int8{127});

static_assert(Sub(SatUInt8{1}, SatUInt8{1}) == std::tuple{SatUInt8{0}, false});
static_assert(Sub(SatUInt8{0}, SatUInt8{1}) == std::tuple{SatUInt8{0}, true});
static_assert(SatUInt8{0} - SatUInt8{1} == SatUInt8{0});
static_assert(UInt8{0} - UInt8{1} == UInt8{255});

static_assert(Mul(SatInt8{127}, SatInt8{1}) == std::tuple{SatInt8{127}, false});
static_assert(Mul(SatInt8{-128}, SatInt8{1}) == std::tuple{SatInt8{-128}, false});
static_assert(Mul(SatInt8{1}, SatInt8{-128}) == std::tuple{SatInt8{-128}, false});
static_assert(Mul(SatInt8{1}, SatInt8{127}) == std::tuple{SatInt8{127}, false});
static_assert(Mul(SatInt8{-128}, SatInt8{-128}) == std::tuple{SatInt8{127}, true});
static_assert(Mul(SatInt8{-128}, SatInt8{127}) == std::tuple{SatInt8{-128}, true});
static_assert(Mul(SatInt8{127}, SatInt8{-128}) == std::tuple{SatInt8{-128}, true});
static_assert(Mul(SatInt8{127}, SatInt8{127}) == std::tuple{SatInt8{127}, true});
static_assert(SatInt8{-128} * SatInt8{-128} == SatInt8{127});
static_assert(SatInt8{-128} * SatInt8{127} == SatInt8{-128});
static_assert(SatInt8{127} * SatInt8{-128} == SatInt8{-128});
static_assert(SatInt8{127} * SatInt8{127} == SatInt8{127});
static_assert(Int8{-128} * Int8{-128} == Int8{0});
static_assert(Int8{-128} * Int8{127} == Int8{-128});
static_assert(Int8{127} * Int8{-128} == Int8{-128});
static_assert(Int8{127} * Int8{127} == Int8{1});

static_assert(Mul(SatUInt8{255}, SatUInt8{1}) == std::tuple{SatUInt8{255}, false});
static_assert(Mul(SatUInt8{255}, SatUInt8{255}) == std::tuple{SatUInt8{255}, true});
static_assert(SatUInt8{255} * SatUInt8{255} == SatUInt8{255});
static_assert(UInt8{255} * UInt8{255} == UInt8{1});

static_assert(Div(SatInt8{127}, SatInt8{1}) == std::tuple{SatInt8{127}, false});
static_assert(Div(SatInt8{-128}, SatInt8{-1}) == std::tuple{SatInt8{127}, true});
static_assert(SatInt8{-128} / SatInt8{-1} == SatInt8{127});
static_assert(Int8{-128} / Int8{-1} == Int8{-128});

// Note: division couldn't overflow with SatUInt8 (but could with SatUnt8, see above).
static_assert(Div(SatUInt8{255}, SatUInt8{1}) == std::tuple{SatUInt8{255}, false});
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

// BitCastToSigned and BitCastToUnsigned don't change the bits of integer, they just treat them
// differently.
static_assert(BitCastToSigned(SatUInt8{128}) == SatInt8{-128});
static_assert(BitCastToSigned(UInt8{128}) == Int8{-128});
static_assert(BitCastToUnsigned(SatInt8{-128}) == SatUInt8{128});
static_assert(BitCastToUnsigned(Int8{-128}) == UInt8{128});

static_assert(std::is_same_v<Int16, Int16::SignedType>);
static_assert(std::is_same_v<Int16, UInt16::SignedType>);
static_assert(std::is_same_v<UInt16, Int16::UnsignedType>);
static_assert(std::is_same_v<UInt16, UInt16::UnsignedType>);

static_assert(std::is_same_v<Int16, SignedType<RawInt16>>);
static_assert(std::is_same_v<Int16, SignedType<Int16>>);
static_assert(std::is_same_v<Int16, SignedType<UInt16>>);
static_assert(std::is_same_v<UInt16, UnsignedType<RawInt16>>);
static_assert(std::is_same_v<UInt16, UnsignedType<Int16>>);
static_assert(std::is_same_v<UInt16, UnsignedType<UInt16>>);

static_assert(std::is_same_v<Int16, WrappingType<Int16>>);
static_assert(std::is_same_v<UInt16, WrappingType<UInt16>>);
static_assert(std::is_same_v<Int16, WrappingType<SatInt16>>);
static_assert(std::is_same_v<UInt16, WrappingType<SatUInt16>>);

static_assert(std::is_same_v<SatInt16, SatInt16::SignedType>);
static_assert(std::is_same_v<SatInt16, SatUInt16::SignedType>);
static_assert(std::is_same_v<SatUInt16, SatInt16::UnsignedType>);
static_assert(std::is_same_v<SatUInt16, SatUInt16::UnsignedType>);

static_assert(std::is_same_v<SatInt16, SignedType<SatInt16>>);
static_assert(std::is_same_v<SatInt16, SignedType<SatUInt16>>);
static_assert(std::is_same_v<SatUInt16, UnsignedType<SatInt16>>);
static_assert(std::is_same_v<SatUInt16, UnsignedType<SatUInt16>>);

static_assert(std::is_same_v<SatInt16, SaturatingType<Int16>>);
static_assert(std::is_same_v<SatUInt16, SaturatingType<UInt16>>);
static_assert(std::is_same_v<SatInt16, SaturatingType<SatInt16>>);
static_assert(std::is_same_v<SatUInt16, SaturatingType<SatUInt16>>);

static_assert(std::is_same_v<SatInt16, SignedType<SatInt16>>);
static_assert(std::is_same_v<SatInt16, SignedType<SatUInt16>>);
static_assert(std::is_same_v<SatUInt16, UnsignedType<SatInt16>>);
static_assert(std::is_same_v<SatUInt16, UnsignedType<SatUInt16>>);

static_assert(std::is_same_v<RawInt16, RawType<RawInt16>>);
static_assert(std::is_same_v<RawInt16, RawType<Int16>>);
static_assert(std::is_same_v<RawInt16, RawType<UInt16>>);
static_assert(std::is_same_v<RawInt16, RawType<SatInt16>>);
static_assert(std::is_same_v<RawInt16, RawType<SatUInt16>>);

}  // namespace

}  // namespace berberis
