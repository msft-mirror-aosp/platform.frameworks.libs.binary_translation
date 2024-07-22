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

#ifndef BERBERIS_INTRINSICS_COMMON_TO_X86_INTRINSICS_BINDINGS_H_
#define BERBERIS_INTRINSICS_COMMON_TO_X86_INTRINSICS_BINDINGS_H_

#include <xmmintrin.h>

#include <cstdint>

#include "berberis/base/dependent_false.h"
#include "berberis/intrinsics/intrinsics_args.h"
#include "berberis/intrinsics/type_traits.h"

namespace berberis::intrinsics::bindings {

class Imm2 {
 public:
  using Type = int8_t;
  static constexpr bool kIsImmediate = true;
};

class Imm8 {
 public:
  using Type = int8_t;
  static constexpr bool kIsImmediate = true;
};

class Imm16 {
 public:
  using Type = int16_t;
  static constexpr bool kIsImmediate = true;
};

class Imm32 {
 public:
  using Type = int32_t;
  static constexpr bool kIsImmediate = true;
};

class Imm64 {
 public:
  using Type = int64_t;
  static constexpr bool kIsImmediate = true;
};

class AL {
 public:
  using Type = uint8_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'a';
};

class AX {
 public:
  using Type = uint16_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'a';
};

class EAX {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'a';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kEAX;
};

class RAX {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'a';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kRAX;
};

class CL {
 public:
  using Type = uint8_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'c';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kCL;
};

class CX {
 public:
  using Type = uint16_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'c';
};

class ECX {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'c';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kECX;
};

class RCX {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'c';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kRCX;
};

class DL {
 public:
  using Type = uint8_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'd';
};

class DX {
 public:
  using Type = uint16_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'd';
};

class EDX {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'd';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kEDX;
};

class RDX {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'd';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kRDX;
};

class GeneralReg8 {
 public:
  using Type = uint8_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'q';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kGeneralReg8;
};

class GeneralReg16 {
 public:
  using Type = uint16_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'r';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kGeneralReg16;
};

class GeneralReg32 {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'r';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kGeneralReg32;
};

class GeneralReg64 {
 public:
  using Type = uint64_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'r';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kGeneralReg64;
};

class FLAGS {
 public:
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 0;
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kFLAGS;
};

class FpReg32 {
 public:
  using Type = __m128;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'x';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kFpReg32;
};

class FpReg64 {
 public:
  using Type = __m128;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'x';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kFpReg64;
};

class VecReg128 {
 public:
  using Type = __m128;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'x';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kVecReg128;
};

class XmmReg {
 public:
  using Type = __m128;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'x';
  template <typename MachineInsnArch>
  static constexpr auto kRegClass = MachineInsnArch::kXmmReg;
};

class Mem8 {
 public:
  using Type = uint8_t;
  static constexpr bool kIsImmediate = false;
  static constexpr char kAsRegister = 'm';
};

class Mem16 {
 public:
  using Type = uint16_t;
  static constexpr bool kIsImmediate = false;
  static constexpr char kAsRegister = 'm';
};

class Mem32 {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr char kAsRegister = 'm';
};

class Mem64 {
 public:
  using Type = uint64_t;
  static constexpr bool kIsImmediate = false;
  static constexpr char kAsRegister = 'm';
};

class MemX87 {
 public:
  static constexpr bool kIsImmediate = false;
  static constexpr char kAsRegister = 'm';
};

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class Def;
class DefEarlyClobber;
class Use;
class UseDef;

template <typename Tag, typename MachineRegKind>
constexpr auto ToRegKind() {
  if constexpr (std::is_same_v<Tag, Def>) {
    return MachineRegKind::kDef;
  } else if constexpr (std::is_same_v<Tag, DefEarlyClobber>) {
    return MachineRegKind::kDefEarlyClobber;
  } else if constexpr (std::is_same_v<Tag, Use>) {
    return MachineRegKind::kUse;
  } else if constexpr (std::is_same_v<Tag, UseDef>) {
    return MachineRegKind::kUseDef;
  } else {
    static_assert(kDependentTypeFalse<Tag>);
  }
}

template <typename Tag, typename MachineRegKind>
inline constexpr auto kRegKind = ToRegKind<Tag, MachineRegKind>();

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class NoCPUIDRestriction;
class Has3DNOW;
class Has3DNOWP;
class HasADX;
class HasAES;
class HasAESAVX;
class HasAMXBF16;
class HasAMXFP16;
class HasAMXINT8;
class HasAMXTILE;
class HasAVX;
class HasAVX2;
class HasAVX5124FMAPS;
class HasAVX5124VNNIW;
class HasAVX512BF16;
class HasAVX512BITALG;
class HasAVX512BW;
class HasAVX512CD;
class HasAVX512DQ;
class HasAVX512ER;
class HasAVX512F;
class HasAVX512FP16;
class HasAVX512IFMA;
class HasAVX512PF;
class HasAVX512VBMI;
class HasAVX512VBMI2;
class HasAVX512VL;
class HasAVX512VNNI;
class HasAVX512VPOPCNTDQ;
class HasBMI;
class HasBMI2;
class HasCLMUL;
class HasCMOV;
class HasCMPXCHG16B;
class HasCMPXCHG8B;
class HasF16C;
class HasFMA;
class HasFMA4;
class HasFXSAVE;
class HasLZCNT;
// BMI2 is set and PDEP/PEXT are ok to use. See more here:
//   https://twitter.com/instlatx64/status/1322503571288559617
class HashPDEP;
class HasPOPCNT;
class HasRDSEED;
class HasSERIALIZE;
class HasSHA;
class HasSSE;
class HasSSE2;
class HasSSE3;
class HasSSE4_1;
class HasSSE4_2;
class HasSSE4a;
class HasSSSE3;
class HasTBM;
class HasVAES;
class HasX87;
class IsAuthenticAMD;

// Tag classes. They are never instantioned, only used as tags to pass information about
// bindings.
class NoNansOperation;
class PreciseNanOperationsHandling;
class ImpreciseNanOperationsHandling;

template <auto kIntrinsicTemplateName,
          auto kMacroInstructionTemplateName,
          auto kMnemo,
          typename GetOpcode,
          typename CPUIDRestrictionTemplateValue,
          typename PreciseNanOperationsHandlingTemplateValue,
          bool kSideEffectsTemplateValue,
          typename... Types>
class AsmCallInfo;

template <auto kIntrinsicTemplateName,
          auto kMacroInstructionTemplateName,
          auto kMnemo,
          typename GetOpcode,
          typename CPUIDRestrictionTemplateValue,
          typename PreciseNanOperationsHandlingTemplateValue,
          bool kSideEffectsTemplateValue,
          typename... InputArgumentsTypes,
          typename... OutputArgumentsTypes,
          typename... BindingsTypes>
class AsmCallInfo<kIntrinsicTemplateName,
                  kMacroInstructionTemplateName,
                  kMnemo,
                  GetOpcode,
                  CPUIDRestrictionTemplateValue,
                  PreciseNanOperationsHandlingTemplateValue,
                  kSideEffectsTemplateValue,
                  std::tuple<InputArgumentsTypes...>,
                  std::tuple<OutputArgumentsTypes...>,
                  BindingsTypes...>
    final {
 public:
  static constexpr auto kIntrinsic = kIntrinsicTemplateName;
  static constexpr auto kMacroInstruction = kMacroInstructionTemplateName;
  // TODO(b/260725458): Use lambda template argument after C++20 becomes available.
  template <typename Opcode>
  static constexpr auto kOpcode = GetOpcode{}.template operator()<Opcode>();
  using CPUIDRestriction = CPUIDRestrictionTemplateValue;
  using PreciseNanOperationsHandling = PreciseNanOperationsHandlingTemplateValue;
  static constexpr bool kSideEffects = kSideEffectsTemplateValue;
  static constexpr const char* InputArgumentsTypeNames[] = {
      TypeTraits<InputArgumentsTypes>::kName...};
  static constexpr const char* OutputArgumentsTypeNames[] = {
      TypeTraits<OutputArgumentsTypes>::kName...};
  template <typename Callback, typename... Args>
  constexpr static void ProcessBindings(Callback&& callback, Args&&... args) {
    (callback(ArgTraits<BindingsTypes>(), std::forward<Args>(args)...), ...);
  }
  template <typename Callback, typename... Args>
  constexpr static auto MakeTuplefromBindings(Callback&& callback, Args&&... args) {
    return std::tuple_cat(callback(ArgTraits<BindingsTypes>(), std::forward<Args>(args)...)...);
  }
  using InputArguments = std::tuple<InputArgumentsTypes...>;
  using OutputArguments = std::tuple<OutputArgumentsTypes...>;
  using Bindings = std::tuple<BindingsTypes...>;
  using IntrinsicType = std::conditional_t<std::tuple_size_v<OutputArguments> == 0,
                                           void (*)(InputArgumentsTypes...),
                                           OutputArguments (*)(InputArgumentsTypes...)>;
  template <template <typename, auto, auto, typename...> typename MachineInsnType,
            template <typename...>
            typename ConstructorArgs,
            typename Opcode>
  using MachineInsn = MachineInsnType<AsmCallInfo,
                                      kMnemo,
                                      kOpcode<Opcode>,
                                      ConstructorArgs<BindingsTypes...>,
                                      BindingsTypes...>;
};

}  // namespace berberis::intrinsics::bindings

#endif  // BERBERIS_INTRINSICS_COMMON_TO_X86_INTRINSICS_BINDINGS_H_
