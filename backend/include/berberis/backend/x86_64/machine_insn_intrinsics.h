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

#ifndef BERBERIS_BACKEND_X86_64_MACHINE_INSN_INTRINSICS_H_
#define BERBERIS_BACKEND_X86_64_MACHINE_INSN_INTRINSICS_H_

#include <string>
#include <tuple>
#include <type_traits>
#include <variant>

#include "berberis/backend/code_emitter.h"
#include "berberis/backend/common/machine_ir.h"
#include "berberis/backend/x86_64/code_debug.h"
#include "berberis/backend/x86_64/code_emit.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/base/dependent_false.h"
#include "berberis/base/stringprintf.h"
#include "berberis/intrinsics/intrinsics_args.h"
#include "berberis/intrinsics/intrinsics_bindings.h"

namespace berberis::x86_64 {

// tuple_cat for types, to help remove filtered out types below.
template <typename... Ts>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Ts>()...));

// Predicate to determine whether type T has a Usage alias.
template <class, class = void>
struct has_usage_impl : std::false_type {};
template <class T>
struct has_usage_impl<T, std::void_t<typename T::Usage>> : std::true_type {};
template <typename T>
using has_usage_t = has_usage_impl<T>;

// Filter out types from Ts... that do not satisfy the predicate, collect them
// into a tuple.
template <template <typename> typename Predicate, typename... Ts>
using filter_t =
    tuple_cat_t<std::conditional_t<Predicate<Ts>::value, std::tuple<Ts>, std::tuple<>>...>;

// Convert Binding into constructor argument(s).
template <typename T, typename = void>
struct ConstructorArg;

// Immediates expand into their class type.
template <typename T>
struct ConstructorArg<ArgTraits<T>, std::enable_if_t<ArgTraits<T>::Class::kIsImmediate, void>> {
  using type = std::tuple<typename ArgTraits<T>::Class::Type>;
};

// Mem ops expand into base register and disp.
template <typename T>
struct ConstructorArg<ArgTraits<T>,
                      std::enable_if_t<!ArgTraits<T>::Class::kIsImmediate &&
                                           ArgTraits<T>::RegisterClass::kAsRegister == 'm',
                                       void>> {
  static_assert(
      std::is_same_v<typename ArgTraits<T>::Usage, intrinsics::bindings::DefEarlyClobber>);
  // Need to emit base register AND disp.
  using type = std::tuple<MachineReg, int32_t>;
};

// Everything else expands into a MachineReg.
template <typename T>
struct ConstructorArg<ArgTraits<T>,
                      std::enable_if_t<!ArgTraits<T>::Class::kIsImmediate &&
                                           ArgTraits<T>::RegisterClass::kAsRegister != 'm',
                                       void>> {
  using type = std::tuple<MachineReg>;
};

template <typename T>
using constructor_one_arg_t = typename ConstructorArg<ArgTraits<T>>::type;

// Use this alias to generate constructor Args from bindings via the AsmCallInfo::MachineInsn
// alias. The tuple args will be extracted by the tuple specialization on MachineInsn below.
template <typename... T>
using constructor_args_t = tuple_cat_t<constructor_one_arg_t<T>...>;

// Predicate to determine whether type T is a memory access arg.
template <class, class = void>
struct is_mem_impl : std::false_type {};
template <class T>
struct is_mem_impl<
    T,
    std::enable_if_t<!T::Class::kIsImmediate && T::RegisterClass::kAsRegister == 'm', void>>
    : std::true_type {};
template <typename T>
using is_mem_t = is_mem_impl<T>;

template <typename... Bindings>
constexpr size_t mem_count_v = std::tuple_size_v<filter_t<is_mem_t, ArgTraits<Bindings>...>>;

template <size_t N, typename... Bindings>
constexpr bool has_n_mem_v = mem_count_v<Bindings...> > (N - 1);

template <typename AsmCallInfo, auto kMnemo, auto kOpcode, typename... Bindings>
class MachineInsn final : public MachineInsnX86_64 {
 private:
  template <typename>
  struct GenMachineInsnInfoT;
  // We want to filter out any bindings that are not used for Registers.
  using RegBindings = filter_t<has_usage_t, ArgTraits<Bindings>...>;

 public:
  // This static simplifies constructing this MachineInsn in intrinsic implementations.
  static constexpr MachineInsn* (MachineIRBuilder::*kGenFunc)(
      typename ArgTraits<Bindings>::template BuilderArg<berberis::MachineReg>...) =
      &MachineIRBuilder::template Gen<MachineInsn>;

 public:
  explicit MachineInsn(
      typename ArgTraits<Bindings>::template BuilderArg<berberis::MachineReg>... args)
      : MachineInsnX86_64(&kInfo) {
    ProcessArgs<0 /* reg_idx */, Bindings...>(args...);
  }

  static constexpr MachineInsnInfo kInfo = GenMachineInsnInfoT<RegBindings>::value;

  static constexpr int NumRegOperands() { return kInfo.num_reg_operands; }
  static constexpr const MachineRegKind& RegKindAt(int i) { return kInfo.reg_kinds[i]; }

  std::string GetDebugString() const override {
    std::string s(kMnemo);
    ProcessDebugString<Bindings...>(&s);
    return s;
  }

  void Emit(CodeEmitter* as) const override {
    std::apply(
        AsmCallInfo::kMacroInstruction,
        std::tuple_cat(std::tuple<CodeEmitter&>{*as}, EmitArgs<0 /* reg_idx */, Bindings...>()));
  }

 private:
  // TODO(b/260725458): Use inline template lambda instead after C++20 becomes available.
  template <std::size_t, typename...>
  void ProcessArgs() {}

  template <std::size_t reg_idx, typename B, typename... BindingsRest, typename T, typename... Args>
  void ProcessArgs(T arg, Args... args) {
    if constexpr (std::is_same_v<MachineReg, T>) {
      this->SetRegAt(reg_idx, arg);
      ProcessArgs<reg_idx + 1, BindingsRest..., Args...>(args...);
    } else if constexpr (ArgTraits<B>::Class::kIsImmediate) {
      this->set_imm(arg);
      ProcessArgs<reg_idx, BindingsRest..., Args...>(args...);
    } else {
      static_assert(kDependentTypeFalse<T>);
    }
  }

  static constexpr auto GetInsnKind() {
    if constexpr (AsmCallInfo::kSideEffects) {
      return kMachineInsnSideEffects;
    } else {
      return kMachineInsnDefault;
    }
  }

  template <typename... T>
  struct GenMachineInsnInfoT<std::tuple<T...>> {
    static constexpr MachineInsnInfo value = MachineInsnInfo(
        {kOpcode,
         sizeof...(T),
         {{&T::RegisterClass::template kRegClass<MachineInsnX86_64>,
           intrinsics::bindings::kRegKind<typename T::Usage, berberis::MachineRegKind>}...},
         GetInsnKind()});
  };

  template <typename... Args>
  void ProcessDebugString(std::string* s) const {
    *s += " " + ProcessDebugStringArgs<0 /* arg_idx */, 0 /* reg_idx */, Args...>();
    if (this->recovery_pc()) {
      *s += StringPrintf(" <0x%" PRIxPTR ">", this->recovery_pc());
    }
  }

  // TODO(b/260725458): Use inline template lambda instead after C++20 becomes available.
  template <>
  void ProcessDebugString<>(std::string*) const {}

  template <std::size_t arg_idx, std::size_t reg_idx, typename T, typename... Args>
  std::string ProcessDebugStringArgs() const {
    std::string prefix;
    if constexpr (arg_idx > 0) {
      prefix = ", ";
    }
    if constexpr (ArgTraits<T>::Class::kIsImmediate) {
      return prefix + GetImmOperandDebugString(this) +
             ProcessDebugStringArgs<arg_idx + 1, reg_idx, Args...>();
    } else if constexpr (ArgTraits<T>::RegisterClass::kIsImplicitReg) {
      return prefix + GetImplicitRegOperandDebugString(this, reg_idx) +
             ProcessDebugStringArgs<arg_idx + 1, reg_idx + 1, Args...>();
    } else {
      return prefix + GetRegOperandDebugString(this, reg_idx) +
             ProcessDebugStringArgs<arg_idx + 1, reg_idx + 1, Args...>();
    }
  }

  template <std::size_t, std::size_t>
  std::string ProcessDebugStringArgs() const {
    return "";
  }

  // TODO(b/260725458): Use inline template lambda instead after C++20 becomes available.
  template <std::size_t>
  auto EmitArgs() const {
    return std::tuple{};
  }

  template <std::size_t reg_idx, typename T, typename... Args>
  auto EmitArgs() const {
    if constexpr (ArgTraits<T>::Class::kIsImmediate) {
      return std::tuple_cat(
          std::tuple{static_cast<typename ArgTraits<T>::template BuilderArg<berberis::MachineReg>>(
              MachineInsnX86_64::imm())},
          EmitArgs<reg_idx, Args...>());
    } else if constexpr (ArgTraits<T>::RegisterClass::kAsRegister == 'x') {
      return std::tuple_cat(std::tuple{GetXReg(this->RegAt(reg_idx))},
                            EmitArgs<reg_idx + 1, Args...>());
    } else if constexpr (ArgTraits<T>::RegisterClass::kAsRegister == 'r' ||
                         ArgTraits<T>::RegisterClass::kAsRegister == 'q') {
      return std::tuple_cat(std::tuple{GetGReg(this->RegAt(reg_idx))},
                            EmitArgs<reg_idx + 1, Args...>());
    } else if constexpr (ArgTraits<T>::RegisterClass::kIsImplicitReg) {
      return EmitArgs<reg_idx, Args...>();
    } else {
      static_assert(kDependentTypeFalse<T>);
    }
  }
};

}  // namespace berberis::x86_64

#endif  // BERBERIS_BACKEND_X86_64_MACHINE_INSN_INTRINSICS_H_
