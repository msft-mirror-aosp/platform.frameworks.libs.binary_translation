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

#ifndef BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_INLINE_INTRINSIC_H_
#define BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_INLINE_INTRINSIC_H_

#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>

#include "berberis/assembler/x86_64.h"
#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/guest_fp_flags.h"
#include "berberis/intrinsics/intrinsics_process_bindings.h"
#include "berberis/intrinsics/macro_assembler.h"
#include "berberis/runtime_primitives/platform.h"

namespace berberis::inline_intrinsic {

template <auto kFunction,
          typename RegAlloc,
          typename SIMDRegAlloc,
          typename AssemblerResType,
          typename... AssemblerArgType>
bool TryInlineIntrinsic(MacroAssembler<x86_64::Assembler>& as,
                        RegAlloc&& reg_alloc,
                        SIMDRegAlloc&& simd_reg_alloc,
                        AssemblerResType result,
                        AssemblerArgType... args);

template <auto kFunc>
class InlineIntrinsic {
 public:
  template <typename RegAlloc, typename SIMDRegAlloc, typename ResType, typename... ArgType>
  static bool TryInlineWithHostRounding(MacroAssembler<x86_64::Assembler>& as,
                                        RegAlloc&& reg_alloc,
                                        SIMDRegAlloc&& simd_reg_alloc,
                                        ResType result,
                                        ArgType... args) {
    std::tuple args_tuple = std::make_tuple(args...);
    if constexpr (IsTagEq<&intrinsics::FMul<intrinsics::Float64>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FMulHostRounding<intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FMul<intrinsics::Float32>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FMulHostRounding<intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FAdd<intrinsics::Float64>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FAddHostRounding<intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FAdd<intrinsics::Float32>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FAddHostRounding<intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FSub<intrinsics::Float64>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FSubHostRounding<intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FSub<intrinsics::Float32>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FSubHostRounding<intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FDiv<intrinsics::Float64>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FDivHostRounding<intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FDiv<intrinsics::Float32>>) {
      auto [rm, frm, src1, src2] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<&intrinsics::FDivHostRounding<intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src1, src2);
    } else if constexpr (IsTagEq<&intrinsics::FCvtFloatToInteger<int64_t, intrinsics::Float64>>) {
      auto [rm, frm, src] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<
          &intrinsics::FCvtFloatToIntegerHostRounding<int64_t, intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src);
    } else if constexpr (IsTagEq<&intrinsics::FCvtFloatToInteger<int64_t, intrinsics::Float32>>) {
      auto [rm, frm, src] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<
          &intrinsics::FCvtFloatToIntegerHostRounding<int64_t, intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src);
    } else if constexpr (IsTagEq<&intrinsics::FCvtFloatToInteger<int32_t, intrinsics::Float64>>) {
      auto [rm, frm, src] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<
          &intrinsics::FCvtFloatToIntegerHostRounding<int32_t, intrinsics::Float64>>(
          as, reg_alloc, simd_reg_alloc, result, src);
    } else if constexpr (IsTagEq<&intrinsics::FCvtFloatToInteger<int32_t, intrinsics::Float32>>) {
      auto [rm, frm, src] = args_tuple;
      if (rm != FPFlags::DYN) {
        return false;
      }
      return TryInlineIntrinsic<
          &intrinsics::FCvtFloatToIntegerHostRounding<int32_t, intrinsics::Float32>>(
          as, reg_alloc, simd_reg_alloc, result, src);
    }
    return false;
  }

 private:
  template <auto kFunction>
  class FunctionCompareTag;

  template <auto kOtherFunction>
  static constexpr bool IsTagEq =
      std::is_same_v<FunctionCompareTag<kFunc>, FunctionCompareTag<kOtherFunction>>;
};

template <typename format, typename DestType, typename SrcType>
auto Mov(MacroAssembler<x86_64::Assembler>& as, DestType dest, SrcType src)
    -> decltype(std::declval<MacroAssembler<x86_64::Assembler>>()
                    .Mov<format>(std::declval<DestType>(), std::declval<SrcType>())) {
  if constexpr (std::is_integral_v<format>) {
    return as.template Mov<format>(dest, src);
  } else if (host_platform::kHasAVX) {
    return as.template Vmov<format>(dest, src);
  } else {
    return as.template Mov<format>(dest, src);
  }
}

template <typename format, typename DestType, typename SrcType>
auto Mov(MacroAssembler<x86_64::Assembler>& as, DestType dest, SrcType src)
    -> decltype(std::declval<MacroAssembler<x86_64::Assembler>>()
                    .Movs<format>(std::declval<DestType>(), std::declval<SrcType>())) {
  if (host_platform::kHasAVX) {
    if constexpr (std::is_same_v<DestType, MacroAssembler<x86_64::Assembler>::XMMRegister> &&
                  std::is_same_v<SrcType, MacroAssembler<x86_64::Assembler>::XMMRegister>) {
      return as.template Vmovs<format>(dest, dest, src);
    } else {
      return as.template Vmovs<format>(dest, src);
    }
  } else {
    return as.template Movs<format>(dest, src);
  }
}

template <auto kFunction,
          typename RegAlloc,
          typename SIMDRegAlloc,
          typename AssemblerResType,
          typename... AssemblerArgType>
bool TryInlineIntrinsic(MacroAssembler<x86_64::Assembler>& as,
                        RegAlloc&& reg_alloc,
                        SIMDRegAlloc&& simd_reg_alloc,
                        AssemblerResType result,
                        AssemblerArgType... args);

template <auto kFunction,
          typename RegAlloc,
          typename SIMDRegAlloc,
          typename AssemblerResType,
          typename... AssemblerArgType>
class TryBindingBasedInlineIntrinsic {

  template <auto kFunctionForFriend,
            typename RegAllocForFriend,
            typename SIMDRegAllocForFriend,
            typename AssemblerResTypeForFriend,
            typename... AssemblerArgTypeForFriend>
  friend bool TryInlineIntrinsic(MacroAssembler<x86_64::Assembler>& as,
                                 RegAllocForFriend&& reg_alloc,
                                 SIMDRegAllocForFriend&& simd_reg_alloc,
                                 AssemblerResTypeForFriend result,
                                 AssemblerArgTypeForFriend... args);
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
      intrinsics::bindings::CPUIDRestriction kCPUIDRestrictionTemplateValue,
      intrinsics::bindings::PreciseNanOperationsHandling kPreciseNanOperationsHandlingTemplateValue,
      bool kSideEffectsTemplateValue,
      typename... Types>
  friend class intrinsics::bindings::AsmCallInfo;

  TryBindingBasedInlineIntrinsic() = delete;
  TryBindingBasedInlineIntrinsic(const TryBindingBasedInlineIntrinsic&) = delete;
  TryBindingBasedInlineIntrinsic(TryBindingBasedInlineIntrinsic&&) = default;
  TryBindingBasedInlineIntrinsic& operator=(const TryBindingBasedInlineIntrinsic&) = delete;
  TryBindingBasedInlineIntrinsic& operator=(TryBindingBasedInlineIntrinsic&&) = default;

  TryBindingBasedInlineIntrinsic(MacroAssembler<x86_64::Assembler>& as,
                                 RegAlloc& reg_alloc,
                                 SIMDRegAlloc& simd_reg_alloc,
                                 AssemblerResType result,
                                 AssemblerArgType... args)
      : as_(as),
        reg_alloc_(reg_alloc),
        simd_reg_alloc_(simd_reg_alloc),
        result_(result),
        input_args_(std::tuple{args...}),
        success_(
            intrinsics::bindings::ProcessBindings<kFunction,
                                                  AssemblerX86<x86_64::Assembler>,
                                                  x86_64::Assembler,
                                                  MacroAssembler<x86_64::Assembler>,
                                                  bool,
                                                  TryBindingBasedInlineIntrinsic&>(*this, false)) {}
  operator bool() { return success_; }

  template <typename AsmCallInfo>
  std::optional<bool> /*ProcessBindingsClient*/ operator()(AsmCallInfo asm_call_info) {
    static_assert(std::is_same_v<decltype(kFunction), typename AsmCallInfo::IntrinsicType>);
    static_assert(AsmCallInfo::kPreciseNanOperationsHandling ==
                  intrinsics::bindings::kNoNansOperation);
    if constexpr (AsmCallInfo::kCPUIDRestriction == intrinsics::bindings::kNoCPUIDRestriction) {
      // No restrictions. Do nothing.
    } else if constexpr (AsmCallInfo::kCPUIDRestriction == intrinsics::bindings::kHasLZCNT) {
      if (!host_platform::kHasLZCNT) {
        return false;
      }
    } else if constexpr (AsmCallInfo::kCPUIDRestriction == intrinsics::bindings::kHasBMI) {
      if (!host_platform::kHasBMI) {
        return false;
      }
    } else if constexpr (AsmCallInfo::kCPUIDRestriction == intrinsics::bindings::kHasAVX) {
      if (!host_platform::kHasAVX) {
        return false;
      }
    } else {
      static_assert(kDependentValueFalse<AsmCallInfo::kCPUIDRestriction>);
    }
    std::apply(
        AsmCallInfo::kMacroInstruction,
        std::tuple_cat(std::tuple<MacroAssembler<x86_64::Assembler>&>{as_},
                       AsmCallInfo::template MakeTuplefromBindings<TryBindingBasedInlineIntrinsic&>(
                           *this, asm_call_info)));
    static_assert(std::tuple_size_v<typename AsmCallInfo::OutputArguments> == 1);
    using ReturnType = std::tuple_element_t<0, typename AsmCallInfo::OutputArguments>;
    if constexpr (std::is_integral_v<ReturnType> && sizeof(ReturnType) < sizeof(std::int32_t)) {
      // Don't handle these types just yet. We are not sure how to expand them and there
      // are no examples.
      static_assert(kDependentTypeFalse<ReturnType>);
    }
    if constexpr (std::is_same_v<ReturnType, int32_t> || std::is_same_v<ReturnType, uint32_t>) {
      // Expans 32 bit values as signed. Even if actual results are processed as unsigned!
      as_.Expand<int64_t, std::make_signed_t<ReturnType>>(result_, result_);
    } else if constexpr (std::is_integral_v<ReturnType> &&
                         sizeof(ReturnType) == sizeof(std::int64_t)) {
      // Do nothing, we have already produced expanded value.
    } else if constexpr (std::is_same_v<ReturnType, intrinsics::Float32> ||
                         std::is_same_v<ReturnType, intrinsics::Float64>) {
      // Do nothing, NaN boxing is handled by semantics player.
    } else {
      static_assert(kDependentTypeFalse<ReturnType>);
    }
    return {true};
  }

  template <typename ArgBinding, typename AsmCallInfo>
  auto /*MakeTuplefromBindingsClient*/ operator()(ArgTraits<ArgBinding>, AsmCallInfo) {
    using RegisterClass = typename ArgTraits<ArgBinding>::RegisterClass;
    if constexpr (RegisterClass::kAsRegister == 'x') {
      return ProcessArgInput<ArgBinding, AsmCallInfo>(simd_reg_alloc_);
    } else {
      return ProcessArgInput<ArgBinding, AsmCallInfo>(reg_alloc_);
    }
  }

  template <typename ArgBinding, typename AsmCallInfo, typename RegAllocForArg>
  auto ProcessArgInput(RegAllocForArg&& reg_alloc) {
    static constexpr const auto& arg_info = ArgTraits<ArgBinding>::arg_info;
    using RegisterClass = typename ArgTraits<ArgBinding>::RegisterClass;
    using Usage = typename ArgTraits<ArgBinding>::Usage;
    if constexpr (arg_info.arg_type == ArgInfo::IN_ARG) {
      static_assert(std::is_same_v<Usage, intrinsics::bindings::Use>);
      static_assert(!RegisterClass::kIsImplicitReg);
      return std::tuple{std::get<arg_info.from>(input_args_)};
    } else if constexpr (arg_info.arg_type == ArgInfo::IN_OUT_ARG) {
      static_assert(std::is_same_v<Usage, intrinsics::bindings::UseDef>);
      static_assert(!RegisterClass::kIsImplicitReg);
      Mov<std::tuple_element_t<arg_info.from, typename AsmCallInfo::InputArguments>>(
          as_, result_, std::get<arg_info.from>(input_args_));
      return std::tuple{result_};
    } else if constexpr (arg_info.arg_type == ArgInfo::IN_TMP_ARG) {
      if constexpr (RegisterClass::kAsRegister == 'c') {
        Mov<std::tuple_element_t<arg_info.from, typename AsmCallInfo::InputArguments>>(
            as_, as_.rcx, std::get<arg_info.from>(input_args_));
        return std::tuple{};
      } else {
        static_assert(std::is_same_v<Usage, intrinsics::bindings::UseDef>);
        static_assert(!RegisterClass::kIsImplicitReg);
        auto reg = reg_alloc();
        Mov<std::tuple_element_t<arg_info.from, typename AsmCallInfo::InputArguments>>(
            as_, reg, std::get<arg_info.from>(input_args_));
        return std::tuple{reg};
      }

    } else if constexpr (arg_info.arg_type == ArgInfo::OUT_ARG) {
      static_assert(std::is_same_v<Usage, intrinsics::bindings::Def> ||
                    std::is_same_v<Usage, intrinsics::bindings::DefEarlyClobber>);
      static_assert(!RegisterClass::kIsImplicitReg);
      return std::tuple{result_};
    } else if constexpr (arg_info.arg_type == ArgInfo::TMP_ARG) {
      static_assert(std::is_same_v<Usage, intrinsics::bindings::Def> ||
                    std::is_same_v<Usage, intrinsics::bindings::DefEarlyClobber>);
      if constexpr (RegisterClass::kIsImplicitReg) {
        return std::tuple{};
      } else {
        return std::tuple{reg_alloc()};
      }
    } else {
      static_assert(kDependentValueFalse<arg_info.arg_type>);
    }
  }

 private:
  MacroAssembler<x86_64::Assembler>& as_;
  RegAlloc& reg_alloc_;
  SIMDRegAlloc& simd_reg_alloc_;
  AssemblerResType result_;
  std::tuple<AssemblerArgType...> input_args_;
  bool success_;
};

template <auto kFunction,
          typename RegAlloc,
          typename SIMDRegAlloc,
          typename AssemblerResType,
          typename... AssemblerArgType>
bool TryInlineIntrinsic(MacroAssembler<x86_64::Assembler>& as,
                        RegAlloc&& reg_alloc,
                        SIMDRegAlloc&& simd_reg_alloc,
                        AssemblerResType result,
                        AssemblerArgType... args) {
  if (InlineIntrinsic<kFunction>::TryInlineWithHostRounding(
          as, reg_alloc, simd_reg_alloc, result, args...)) {
    return true;
  }

  return TryBindingBasedInlineIntrinsic<kFunction,
                                        RegAlloc,
                                        SIMDRegAlloc,
                                        AssemblerResType,
                                        AssemblerArgType...>(
      as, reg_alloc, simd_reg_alloc, result, args...);
}

}  // namespace berberis::inline_intrinsic

#endif  // BERBERIS_LITE_TRANSLATOR_RISCV64_TO_X86_64_CALL_INTRINSIC_H_
