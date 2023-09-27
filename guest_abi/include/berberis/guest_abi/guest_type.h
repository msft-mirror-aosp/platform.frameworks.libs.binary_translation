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

#ifndef BERBERIS_GUEST_ABI_GUEST_TYPE_H_
#define BERBERIS_GUEST_ABI_GUEST_TYPE_H_

#include <type_traits>

#include "berberis/base/bit_util.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

// Host representation of GuestType.  Could be automatically converted to host type, currently.
template <typename Type, typename = void>
class GuestType;

template <typename Type>
GuestType(Type) -> GuestType<Type>;

template <typename Type>
inline constexpr bool kIsGuestType = false;

template <typename Type1, typename Type2>
inline constexpr bool kIsGuestType<GuestType<Type1, Type2>> = true;

template <typename StructType>
class GuestType<
    StructType,
    std::enable_if_t<(std::is_class_v<StructType> ||
                      std::is_union_v<StructType>)&&std::is_standard_layout_v<StructType> &&
                     std::is_trivially_copyable_v<StructType>>> {
 public:
  using Type = StructType;
  GuestType(const StructType& value) : value_(value) {}
  GuestType(StructType&& value) : value_(value) {}
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = default;
  GuestType& operator=(const GuestType& data) = default;
  GuestType& operator=(GuestType&& data) = default;
  ~GuestType() = default;
  operator StructType() const { return value_; }

 private:
  StructType value_ = {};
};

template <typename IntegerType>
class alignas(
    sizeof(IntegerType)) GuestType<IntegerType, std::enable_if_t<std::is_integral_v<IntegerType>>> {
 public:
  using Type = IntegerType;
  GuestType(const IntegerType& value) : value_(value) {}
  GuestType(IntegerType&& value) : value_(value) {}
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = default;
  GuestType& operator=(const GuestType& data) = default;
  GuestType& operator=(GuestType&& data) = default;
  ~GuestType() = default;
  operator IntegerType() const { return value_; }
#define DEFINE_ARITHMETIC_ASSIGNMENT_OPERATOR(x) \
  GuestType& operator x(const GuestType& data) { \
    value_ x data.value_;                        \
    return *this;                                \
  }                                              \
  GuestType& operator x(GuestType&& data) {      \
    value_ x data.value_;                        \
    return *this;                                \
  }
  DEFINE_ARITHMETIC_ASSIGNMENT_OPERATOR(+=)
  DEFINE_ARITHMETIC_ASSIGNMENT_OPERATOR(-=)
  DEFINE_ARITHMETIC_ASSIGNMENT_OPERATOR(*=)
  DEFINE_ARITHMETIC_ASSIGNMENT_OPERATOR(/=)
  DEFINE_ARITHMETIC_ASSIGNMENT_OPERATOR(%=)
  DEFINE_ARITHMETIC_ASSIGNMENT_OPERATOR(^=)
  DEFINE_ARITHMETIC_ASSIGNMENT_OPERATOR(&=)
  DEFINE_ARITHMETIC_ASSIGNMENT_OPERATOR(|=)
#undef DEFINE_ARITHMETIC_ASSIGNMENT_OPERATOR

 private:
  IntegerType value_ = 0;
};

template <typename EnumType>
class GuestType<EnumType, std::enable_if_t<std::is_enum_v<EnumType>>> {
 public:
  using Type = EnumType;
  GuestType(const EnumType& value) : value_(value) {}
  GuestType(EnumType&& value) : value_(value) {}
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = default;
  GuestType& operator=(const GuestType& data) = default;
  GuestType& operator=(GuestType&& data) = default;
  ~GuestType() = default;
  operator EnumType() const { return value_; }

 private:
  EnumType value_ = EnumType(0);
};

template <typename FloatingPointType>
class alignas(sizeof(FloatingPointType))
    GuestType<FloatingPointType, std::enable_if_t<std::is_floating_point_v<FloatingPointType>>> {
 public:
  using Type = FloatingPointType;
  GuestType(const FloatingPointType& value) : value_(value) {}
  GuestType(FloatingPointType&& value) : value_(value) {}
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = default;
  GuestType& operator=(const GuestType& data) = default;
  GuestType& operator=(GuestType&& data) = default;
  ~GuestType() = default;
  operator FloatingPointType() const { return value_; }

 private:
  FloatingPointType value_ = 0;
};

template <>
class GuestType<void*> {
 public:
  using Type = void*;
  GuestType(void* const& value) : value_(ToGuestAddr(value)) {}
  GuestType(void*&& value) : value_(ToGuestAddr(value)) {}
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = default;
  GuestType& operator=(const GuestType& data) = default;
  GuestType& operator=(GuestType&& data) = default;
  ~GuestType() = default;
  friend GuestAddr ToGuestAddr(const GuestType& guest_type) { return guest_type.value_; }
  friend void* ToHostAddr(const GuestType& guest_type) {
    return ToHostAddr<void>(guest_type.value_);
  }
  operator void*() const { return ToHostAddr<void>(value_); }
  template <typename Type>
  explicit operator Type() const {
    return ToHostAddr<std::remove_pointer_t<Type>>(value_);
  }

 private:
  GuestAddr value_ = 0;
};

template <>
class GuestType<void* const> {
 public:
  using Type = void* const;
  GuestType(void* const& value) : value_(ToGuestAddr(value)) {}
  GuestType(void*&& value) = delete;
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = delete;
  GuestType& operator=(const GuestType& data) = delete;
  GuestType& operator=(GuestType&& data) = delete;
  ~GuestType() = default;
  friend GuestAddr ToGuestAddr(const GuestType& guest_type) { return guest_type.value_; }
  friend void* ToHostAddr(const GuestType& guest_type) {
    return ToHostAddr<void>(guest_type.value_);
  }
  operator void*() const { return ToHostAddr<void>(value_); }
  template <typename Type>
  explicit operator Type() const {
    return ToHostAddr<std::remove_pointer_t<Type>>(value_);
  }

 private:
  const GuestAddr value_ = 0;
};

template <>
class GuestType<const void*> {
 public:
  using Type = const void*;
  GuestType(const void* const& value) : value_(ToGuestAddr(value)) {}
  GuestType(const void*&& value) : value_(ToGuestAddr(value)) {}
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = default;
  GuestType& operator=(const GuestType& data) = default;
  GuestType& operator=(GuestType&& data) = default;
  ~GuestType() = default;
  friend GuestAddr ToGuestAddr(const GuestType& guest_type) { return guest_type.value_; }
  friend const void* ToHostAddr(const GuestType& guest_type) {
    return ToHostAddr<const void>(guest_type.value_);
  }
  operator const void*() const { return ToHostAddr<void>(value_); }
  template <typename Type>
  explicit operator const Type() const {
    return static_cast<const Type>(value_);
  }

 private:
  GuestAddr value_ = 0;
};

template <>
class GuestType<const void* const> {
 public:
  using Type = const void* const;
  GuestType(const void* const& value) : value_(ToGuestAddr(value)) {}
  GuestType(const void*&& value) = delete;
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = delete;
  GuestType& operator=(const GuestType& data) = delete;
  GuestType& operator=(GuestType&& data) = delete;
  ~GuestType() = default;
  friend GuestAddr ToGuestAddr(const GuestType& guest_type) { return guest_type.value_; }
  friend const void* ToHostAddr(const GuestType& guest_type) {
    return ToHostAddr<const void>(guest_type.value_);
  }
  operator const void*() const { return ToHostAddr<void>(value_); }
  template <typename Type>
  explicit operator const Type() const {
    return static_cast<const Type>(value_);
  }

 private:
  const GuestAddr value_ = 0;
};

template <typename PointeeType>
class GuestType<PointeeType*> {
 public:
  using Type = PointeeType*;
  GuestType(Type const& value) : value_(ToGuestAddr(value)) {}
  GuestType(Type&& value) : value_(ToGuestAddr(value)) {}
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = default;
  GuestType& operator=(const GuestType& data) = default;
  GuestType& operator=(GuestType&& data) = default;
  ~GuestType() = default;
  friend GuestAddr ToGuestAddr(const GuestType& guest_type) { return guest_type.value_; }
  friend Type ToHostAddr(const GuestType& guest_type) {
    return ToHostAddr<PointeeType>(guest_type.value_);
  }
  operator Type() const { return ToHostAddr<PointeeType>(value_); }
  const Type operator->() const { return ToHostAddr<PointeeType>(value_); }
  Type operator->() { return ToHostAddr<PointeeType>(value_); }

 private:
  GuestAddr value_ = 0;
};

template <typename PointeeType>
class GuestType<PointeeType* const> {
 public:
  using Type = PointeeType*;
  GuestType(Type const& value) : value_(ToGuestAddr(value)) {}
  GuestType(Type&& value) = delete;
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = delete;
  GuestType& operator=(const GuestType& data) = delete;
  GuestType& operator=(GuestType&& data) = delete;
  ~GuestType() = default;
  friend GuestAddr ToGuestAddr(const GuestType& guest_type) { return guest_type.value_; }
  friend Type ToHostAddr(const GuestType& guest_type) {
    return ToHostAddr<PointeeType>(guest_type.value_);
  }
  operator Type() const { return ToHostAddr<PointeeType>(value_); }
  const Type operator->() const { return ToHostAddr<PointeeType>(value_); }
  Type operator->() { return ToHostAddr<PointeeType>(value_); }

 private:
  const GuestAddr value_ = 0;
};

// Pointer to function could be implicitly converted to GuestAddr.  It couldn't be called without
// doing explicit wrapping and that procedure is not cheap, so better to do it explicitly.
template <typename ResultType, typename... ArgumentType>
class GuestType<ResultType (*)(ArgumentType...)> {
 public:
  using Type = ResultType (*)(ArgumentType...);
  explicit GuestType(GuestAddr value) : value_(value) {}
  GuestType(ResultType (*const& value)(ArgumentType...))
      : value_(ToGuestAddr(reinterpret_cast<void*>(value))) {}
  GuestType(ResultType (*&&value)(ArgumentType...))
      : value_(ToGuestAddr(reinterpret_cast<void*>(value))) {}
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = default;
  GuestType& operator=(const GuestType& data) = default;
  GuestType& operator=(GuestType&& data) = default;
  ~GuestType() = default;
  friend GuestAddr ToGuestAddr(const GuestType& guest_type) { return guest_type.value_; }
  friend HostCode ToHostCode(const GuestType& guest_type) {
    return ToHostAddr<HostCode>(guest_type.value_);
  }

 private:
  GuestAddr value_ = 0;
};

template <typename ResultType, typename... ArgumentType>
class GuestType<ResultType (*const)(ArgumentType...)> {
 public:
  using Type = ResultType (*const)(ArgumentType...);
  explicit GuestType(GuestAddr value) : value_(value) {}
  GuestType(ResultType (*const& value)(ArgumentType...))
      : value_(ToGuestAddr(reinterpret_cast<void*>(value))) {}
  GuestType(ResultType (*&&value)(ArgumentType...)) = delete;
  GuestType() = default;
  GuestType(const GuestType&) = default;
  GuestType(GuestType&&) = delete;
  GuestType& operator=(const GuestType& data) = delete;
  GuestType& operator=(GuestType&& data) = delete;
  ~GuestType() = default;
  friend GuestAddr ToGuestAddr(const GuestType& guest_type) { return guest_type.value_; }
  friend HostCode ToHostCode(const GuestType& guest_type) {
    return ToHostAddr<HostCode>(guest_type.value_);
  }

 private:
  const GuestAddr value_ = 0;
};

// Const cast conversion routine for most GuestTypes. If a certain type is not compatible for some
// reason then deleted specialization should be provided.
template <typename TypeOut, typename TypeIn1, typename TypeIn2>
inline auto ConstCast(GuestType<TypeIn1, TypeIn2> value)
    -> std::enable_if_t<kIsGuestType<TypeOut> &&
                            sizeof(const_cast<typename TypeOut::Type>(
                                std::declval<typename GuestType<TypeIn1, TypeIn2>::Type>())),
                        TypeOut> {
  return bit_cast<TypeOut>(value);
}

// Static cast conversion routine for most GuestTypes. If a certain type is not compatible for some
// reason then deleted specialization should be provided.
template <typename TypeOut, typename TypeIn1, typename TypeIn2>
inline auto StaticCast(GuestType<TypeIn1, TypeIn2> value)
    -> std::enable_if_t<kIsGuestType<TypeOut> &&
                            sizeof(static_cast<typename TypeOut::Type>(
                                std::declval<typename GuestType<TypeIn1, TypeIn2>::Type>())),
                        TypeOut> {
  return bit_cast<TypeOut>(value);
}

}  // namespace berberis

#endif  // BERBERIS_GUEST_ABI_GUEST_TYPE_H_
