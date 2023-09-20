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

#ifndef BERBERIS_GUEST_ABI_GUEST_ABI_ARCH_H_
#define BERBERIS_GUEST_ABI_GUEST_ABI_ARCH_H_

#include "berberis/calling_conventions/calling_conventions_riscv64.h"  // IWYU pragma: export.
#include "berberis/guest_abi/guest_type.h"

namespace berberis {

class GuestAbi {
 public:
  enum CallingConventionsVariant {
    kLp64,   // Soft float.
    kLp64d,  // Hardware float and double.
    kDefaultAbi = kLp64
  };

 protected:
  enum class ArgumentClass { kInteger, kFp, kLargeStruct };

  template <typename Type, CallingConventionsVariant = kDefaultAbi, typename = void>
  struct GuestArgumentInfo;

  template <typename IntegerType, CallingConventionsVariant kCallingConventionsVariant>
  struct GuestArgumentInfo<IntegerType,
                           kCallingConventionsVariant,
                           std::enable_if_t<std::is_integral_v<IntegerType>>> {
    // Integers wider than 8 bytes are not supported.  They do not appear in the public Android API.
    // TODO: Remove this if 16-byte parameters are encountered.
    static_assert(sizeof(IntegerType) <= 8);
    constexpr static ArgumentClass kArgumentClass = ArgumentClass::kInteger;
    constexpr static unsigned kSize = sizeof(IntegerType);
    // Use sizeof, not alignof for kAlignment because all integer types are naturally aligned on
    // RISC-V, which is not guaranteed to be true for the host.
    constexpr static unsigned kAlignment = sizeof(IntegerType);
    using GuestType = GuestType<IntegerType>;
    using HostType = IntegerType;
  };

  template <typename EnumType, CallingConventionsVariant kCallingConventionsVariant>
  struct GuestArgumentInfo<EnumType,
                           kCallingConventionsVariant,
                           std::enable_if_t<std::is_enum_v<EnumType>>>
      : GuestArgumentInfo<std::underlying_type_t<EnumType>> {
    using GuestType = GuestType<EnumType>;
    using HostType = EnumType;
  };

  template <typename PointeeType, CallingConventionsVariant kCallingConventionsVariant>
  struct GuestArgumentInfo<PointeeType*, kCallingConventionsVariant> {
    constexpr static ArgumentClass kArgumentClass = ArgumentClass::kInteger;
    constexpr static unsigned kSize = 8;
    constexpr static unsigned kAlignment = 8;
    using GuestType = GuestType<PointeeType*>;
    using HostType = PointeeType*;
  };

  template <typename ResultType,
            typename... ArgumentType,
            CallingConventionsVariant kCallingConventionsVariant>
  struct GuestArgumentInfo<ResultType (*)(ArgumentType...), kCallingConventionsVariant> {
    constexpr static ArgumentClass kArgumentClass = ArgumentClass::kInteger;
    constexpr static unsigned kSize = 8;
    constexpr static unsigned kAlignment = 8;
    using GuestType = GuestType<ResultType (*)(ArgumentType...)>;
    using HostType = ResultType (*)(ArgumentType...);
  };

  template <>
  struct GuestArgumentInfo<float, kLp64> {
    constexpr static ArgumentClass kArgumentClass = ArgumentClass::kInteger;
    constexpr static unsigned kSize = 4;
    constexpr static unsigned kAlignment = 4;
    using GuestType = GuestType<float>;
    using HostType = float;
  };

  template <>
  struct GuestArgumentInfo<float, kLp64d> : GuestArgumentInfo<float, kLp64> {
    constexpr static ArgumentClass kArgumentClass = ArgumentClass::kFp;
  };

  template <>
  struct GuestArgumentInfo<double, kLp64> {
    constexpr static ArgumentClass kArgumentClass = ArgumentClass::kInteger;
    constexpr static unsigned kSize = 8;
    constexpr static unsigned kAlignment = 8;
    using GuestType = GuestType<double>;
    using HostType = double;
  };

  template <>
  struct GuestArgumentInfo<double, kLp64d> : GuestArgumentInfo<double, kLp64> {
    constexpr static ArgumentClass kArgumentClass = ArgumentClass::kFp;
  };

  // Structures larger than 16 bytes are passed by reference.
  template <typename LargeStructType, CallingConventionsVariant kCallingConventionsVariant>
  struct GuestArgumentInfo<
      LargeStructType,
      kCallingConventionsVariant,
      std::enable_if_t<std::is_class_v<LargeStructType> && (sizeof(LargeStructType) > 16)>> {
    constexpr static ArgumentClass kArgumentClass = ArgumentClass::kLargeStruct;
    constexpr static unsigned kSize = 8;
    constexpr static unsigned kAlignment = 8;
    // Although the structure is passed by reference, keep the type of the underlying structure
    // here.  This is for passing the structure as a function argument.  It is easier to pass
    // constant structures using this declaration than adding const to the pointee type of a
    // pointer.
    using GuestType = GuestType<LargeStructType>;
    using HostType = LargeStructType;
  };
};

}  // namespace berberis

#endif  // BERBERIS_GUEST_ABI_GUEST_ABI_ARCH_H_
