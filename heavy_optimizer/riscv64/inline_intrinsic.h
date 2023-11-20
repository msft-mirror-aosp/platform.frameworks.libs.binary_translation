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

#ifndef BERBERIS_HEAVY_OPTIMIZER_RISCV64_INLINE_INTRINSIC_H_
#define BERBERIS_HEAVY_OPTIMIZER_RISCV64_INLINE_INTRINSIC_H_

#include <cfenv>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <variant>

#include "berberis/assembler/x86_64.h"
#include "berberis/backend/x86_64/machine_insn_intrinsics.h"
#include "berberis/backend/x86_64/machine_ir.h"
#include "berberis/backend/x86_64/machine_ir_builder.h"
#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/intrinsics.h"
#include "berberis/intrinsics/intrinsics_process_bindings.h"
#include "berberis/intrinsics/macro_assembler.h"
#include "berberis/runtime_primitives/platform.h"

namespace berberis {

template <auto kFunc>
class InlineIntrinsic {
 public:
  template <typename ResType, typename FlagRegister, typename... ArgType>
  static bool TryInline(x86_64::MachineIRBuilder* /* builder */,
                        ResType /* result */,
                        FlagRegister /* flag_register */,
                        ArgType... /* args */) {
    // TODO(b/232598137) Implement intrinsics
    return false;
  }

  template <typename FlagRegister, typename... ArgType>
  static bool TryInline(x86_64::MachineIRBuilder* /* builder */,
                        FlagRegister /* flag_register */,
                        ArgType... /* args */) {
    // TODO(b/232598137) Implement intrinsics
    return false;
  }

 private:
  // Comparison of pointers which point to different functions is generally not a
  // constexpr since such functions can be merged in object code (comparing
  // pointers to the same function is constexpr). This helper compares them using
  // templates explicitly telling that we are not worried about such subtleties here.
  template <auto kFunction>
  class FunctionCompareTag;

  // Note, if we define it as a variable clang doesn't consider it a constexpr in TryInline().
  template <auto kOtherFunction>
  static constexpr bool IsTagEq() {
    return std::is_same_v<FunctionCompareTag<kFunc>, FunctionCompareTag<kOtherFunction>>;
  }
};

template <std::size_t size, typename DestType, typename SrcType>
auto GenPseudoCopy(x86_64::MachineIRBuilder* builder, DestType dest, SrcType src)
    -> decltype(std::declval<x86_64::MachineIRBuilder*>()->Gen<PseudoCopy>(
        std::declval<DestType>(),
        std::declval<SrcType>(),
        std::declval<std::size_t>())) {
  return builder->Gen<PseudoCopy>(dest, src, size);
}

template <auto kFunction, typename ResType, typename FlagRegister, typename... ArgType>
class TryBindingBasedInlineIntrinsicForHeavyOptimizer {
  template <auto kFunctionForFriend,
            typename ResTypeForFriend,
            typename FlagRegisterForFriend,
            typename... ArgTypeForFriend>
  friend bool TryInlineIntrinsicForHeavyOptimizer(x86_64::MachineIRBuilder* builder,
                                                  ResTypeForFriend result,
                                                  FlagRegisterForFriend flag_register,
                                                  ArgTypeForFriend... args);
  template <auto kFunctionForFriend>
  friend bool TryInlineIntrinsicForHeavyOptimizer(x86_64::MachineIRBuilder* builder);

  template <auto kFunc,
            typename Assembler_common_x86,
            typename Assembler_x86_64,
            typename MacroAssembler,
            typename Result,
            typename Callback,
            typename... Args>
  friend Result intrinsics::bindings::ProcessBindings(Callback callback,
                                                      Result def_result,
                                                      Args&&... args);

  template <
      auto kIntrinsicTemplateName,
      auto kMacroInstructionTemplateName,
      auto kMnemo,
      typename GetOpcode,
      intrinsics::bindings::CPUIDRestriction kCPUIDRestrictionTemplateValue,
      intrinsics::bindings::PreciseNanOperationsHandling kPreciseNanOperationsHandlingTemplateValue,
      bool kSideEffectsTemplateValue,
      typename... Types>
  friend class intrinsics::bindings::AsmCallInfo;

  TryBindingBasedInlineIntrinsicForHeavyOptimizer() = delete;
  TryBindingBasedInlineIntrinsicForHeavyOptimizer(
      const TryBindingBasedInlineIntrinsicForHeavyOptimizer&) = delete;
  TryBindingBasedInlineIntrinsicForHeavyOptimizer(
      TryBindingBasedInlineIntrinsicForHeavyOptimizer&&) = delete;
  TryBindingBasedInlineIntrinsicForHeavyOptimizer& operator=(
      const TryBindingBasedInlineIntrinsicForHeavyOptimizer&) = delete;
  TryBindingBasedInlineIntrinsicForHeavyOptimizer& operator=(
      TryBindingBasedInlineIntrinsicForHeavyOptimizer&&) = delete;

  TryBindingBasedInlineIntrinsicForHeavyOptimizer(x86_64::MachineIRBuilder* builder,
                                                  ResType result,
                                                  FlagRegister flag_register,
                                                  ArgType... args)
      : builder_(builder),
        result_{result},
        flag_register_{flag_register},
        input_args_(std::tuple{args...}),
        success_(
            intrinsics::bindings::ProcessBindings<kFunction,
                                                  AssemblerX86<x86_64::Assembler>,
                                                  x86_64::Assembler,
                                                  std::tuple<MacroAssembler<x86_64::Assembler>>,
                                                  bool,
                                                  TryBindingBasedInlineIntrinsicForHeavyOptimizer&>(
                *this,
                false)) {}

  operator bool() { return success_; }

  // TODO(b/232598137) The MachineIR bindings for some macros can't be instantiated yet. This should
  // be removed once they're supported.
  template <typename AsmCallInfo,
            std::enable_if_t<AsmCallInfo::template kOpcode<MachineOpcode> ==
                                 MachineOpcode::kMachineOpUndefined,
                             bool> = true>
  std::optional<bool> /*ProcessBindingsClient*/ operator()(AsmCallInfo /* asm_call_info */) {
    return false;
  }

  template <typename AsmCallInfo,
            std::enable_if_t<AsmCallInfo::template kOpcode<MachineOpcode> !=
                                 MachineOpcode::kMachineOpUndefined,
                             bool> = true>
  std::optional<bool> /*ProcessBindingsClient*/ operator()(AsmCallInfo asm_call_info) {
    static_assert(std::is_same_v<decltype(kFunction), typename AsmCallInfo::IntrinsicType>);
    if constexpr (AsmCallInfo::kPreciseNanOperationsHandling !=
                  intrinsics::bindings::kNoNansOperation) {
      return false;
    }

    if constexpr (AsmCallInfo::kCPUIDRestriction == intrinsics::bindings::kHasAVX) {
      if (!host_platform::kHasAVX) {
        return false;
      }
    } else if constexpr (AsmCallInfo::kCPUIDRestriction == intrinsics::bindings::kHasBMI) {
      if (!host_platform::kHasBMI) {
        return false;
      }
    } else if constexpr (AsmCallInfo::kCPUIDRestriction == intrinsics::bindings::kHasLZCNT) {
      if (!host_platform::kHasLZCNT) {
        return false;
      }
    } else if constexpr (AsmCallInfo::kCPUIDRestriction == intrinsics::bindings::kHasPOPCNT) {
      if (!host_platform::kHasPOPCNT) {
        return false;
      }
    } else if constexpr (AsmCallInfo::kCPUIDRestriction ==
                         intrinsics::bindings::kNoCPUIDRestriction) {
      // No restrictions. Do nothing.
    } else {
      static_assert(berberis::kDependentValueFalse<AsmCallInfo::kCPUIDRestriction>);
    }

    using MachineInsn =
        typename AsmCallInfo::template MachineInsn<berberis::x86_64::MachineInsn, MachineOpcode>;
    std::apply(MachineInsn::kGenFunc,
               std::tuple_cat(
                   std::tuple<x86_64::MachineIRBuilder&>{*builder_},
                   AsmCallInfo::template MakeTuplefromBindings<
                       TryBindingBasedInlineIntrinsicForHeavyOptimizer&>(*this, asm_call_info)));
    return true;
  }

  template <typename ArgBinding, typename AsmCallInfo>
  auto /*MakeTuplefromBindingsClient*/ operator()(ArgTraits<ArgBinding>, AsmCallInfo) {
    static constexpr const auto& arg_info = ArgTraits<ArgBinding>::arg_info;
    if constexpr (arg_info.arg_type == ArgInfo::IMM_ARG) {
      auto imm = std::get<arg_info.from>(input_args_);
      return std::tuple{imm};
    } else {
      return ProcessArgInput<ArgBinding, AsmCallInfo>();
    }
  }

  template <typename ArgBinding, typename AsmCallInfo>
  auto ProcessArgInput() {
    static constexpr const auto& arg_info = ArgTraits<ArgBinding>::arg_info;
    using RegisterClass = typename ArgTraits<ArgBinding>::RegisterClass;
    using Usage = typename ArgTraits<ArgBinding>::Usage;
    static constexpr const auto kNumOut = std::tuple_size_v<typename AsmCallInfo::OutputArguments>;

    if constexpr (arg_info.arg_type == ArgInfo::IN_ARG) {
      static_assert(std::is_same_v<Usage, intrinsics::bindings::Use>);
      static_assert(!RegisterClass::kIsImplicitReg);
      return std::tuple{std::get<arg_info.from>(input_args_)};
    } else if constexpr (arg_info.arg_type == ArgInfo::IN_OUT_ARG) {
      static_assert(!std::is_same_v<ResType, std::monostate>);
      static_assert(std::is_same_v<Usage, intrinsics::bindings::UseDef>);
      static_assert(!RegisterClass::kIsImplicitReg);
      if constexpr (RegisterClass::kAsRegister == 'x') {
        if constexpr (kNumOut > 1) {
          auto res = std::get<arg_info.to>(result_);
          GenPseudoCopy<16>(builder_, res, std::get<arg_info.from>(input_args_));
          return std::tuple{res};
        } else {
          GenPseudoCopy<16>(builder_, result_, std::get<arg_info.from>(input_args_));
          return std::tuple{result_};
        }
      } else if constexpr (kNumOut > 1) {
        auto res = std::get<arg_info.to>(result_);
        GenPseudoCopy<sizeof(typename RegisterClass::Type)>(
            builder_, res, std::get<arg_info.from>(input_args_));
        return std::tuple{res};
      } else {
        GenPseudoCopy<sizeof(typename RegisterClass::Type)>(
            builder_, result_, std::get<arg_info.from>(input_args_));
        return std::tuple{result_};
      }
    } else if constexpr (arg_info.arg_type == ArgInfo::IN_TMP_ARG) {
      static_assert(std::is_same_v<Usage, intrinsics::bindings::UseDef>);
      static_assert(!RegisterClass::kIsImplicitReg);
      return std::tuple{std::get<arg_info.from>(input_args_)};
    } else if constexpr (arg_info.arg_type == ArgInfo::OUT_ARG) {
      static_assert(!std::is_same_v<ResType, std::monostate>);
      static_assert(std::is_same_v<Usage, intrinsics::bindings::Def> ||
                    std::is_same_v<Usage, intrinsics::bindings::DefEarlyClobber>);
      static_assert(!RegisterClass::kIsImplicitReg);
      if constexpr (kNumOut > 1) {
        return std::tuple{std::get<arg_info.to>(result_)};
      } else {
        return std::tuple{result_};
      }
    } else if constexpr (arg_info.arg_type == ArgInfo::TMP_ARG) {
      static_assert(std::is_same_v<Usage, intrinsics::bindings::Def> ||
                    std::is_same_v<Usage, intrinsics::bindings::DefEarlyClobber>);
      if constexpr (RegisterClass::kAsRegister == 'm') {
        static_assert(kDependentTypeFalse<RegisterClass>);
      } else if constexpr (RegisterClass::kIsImplicitReg) {
        if constexpr (RegisterClass::kAsRegister == 0) {
          return std::tuple{flag_register_};
        } else {
          return std::tuple{};
        }
      } else {
        auto reg = builder_->ir()->AllocVReg();
        return std::tuple{reg};
      }
    } else {
      static_assert(berberis::kDependentValueFalse<arg_info.arg_type>);
    }
  }

 private:
  x86_64::MachineIRBuilder* builder_;
  ResType result_;
  FlagRegister flag_register_;
  std::tuple<ArgType...> input_args_;
  bool success_;
};

template <auto kFunction, typename ResType, typename FlagRegister, typename... ArgType>
bool TryInlineIntrinsicForHeavyOptimizer(x86_64::MachineIRBuilder* builder,
                                         ResType result,
                                         FlagRegister flag_register,
                                         ArgType... args) {
  if (InlineIntrinsic<kFunction>::TryInline(builder, result, flag_register, args...)) {
    return true;
  }

  return TryBindingBasedInlineIntrinsicForHeavyOptimizer<kFunction,
                                                         ResType,
                                                         FlagRegister,
                                                         ArgType...>(
      builder, result, flag_register, args...);
}

template <auto kFunction, typename FlagRegister, typename... ArgType>
bool TryInlineIntrinsicForHeavyOptimizer(x86_64::MachineIRBuilder* builder,
                                         FlagRegister flag_register,
                                         ArgType... args) {
  if (InlineIntrinsic<kFunction>::TryInline(builder, flag_register, args...)) {
    return true;
  }

  return TryBindingBasedInlineIntrinsicForHeavyOptimizer<kFunction, std::monostate, std::monostate>(
      builder, std::monostate{}, flag_register, args...);
}

}  // namespace berberis

#endif  // BERBERIS_HEAVY_OPTIMIZER_RISCV64_INLINE_INTRINSIC_H_
