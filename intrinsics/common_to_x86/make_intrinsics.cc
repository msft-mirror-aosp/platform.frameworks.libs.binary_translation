/*
 * Copyright (C) 2019 The Android Open Source Project
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

#include <stdio.h>
#include <xmmintrin.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "berberis/base/checks.h"
#include "berberis/intrinsics/intrinsics_args.h"
#include "berberis/intrinsics/intrinsics_float.h"
#include "berberis/intrinsics/macro_assembler.h"
#include "berberis/intrinsics/simd_register.h"
#include "berberis/intrinsics/type_traits.h"

#include "text_assembler.h"

namespace berberis {

namespace constants_pool {

// Note: kBerberisMacroAssemblerConstantsRelocated is the same as original,
// unrelocated version in 32-bit world.  But in 64-bit world it's copy on the first 2GiB.
//
// Our builder could be built as 64-bit binary thus we must not mix them.
//
// Note: we have CHECK_*_LAYOUT tests in macro_assembler_common_x86.cc to make sure
// offsets produced by 64-bit builder are usable in 32-bit libberberis.so

extern const int32_t kBerberisMacroAssemblerConstantsRelocated;

int32_t GetOffset(int32_t address) {
  return address - constants_pool::kBerberisMacroAssemblerConstantsRelocated;
}

}  // namespace constants_pool

namespace x86 {

namespace {

// "Addendum to" global TypeTraits - we only need that information here.
template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<uint8_t> {
  [[maybe_unused]] static constexpr char kName[] = "uint8_t";
};

template <>
struct TypeTraits<uint16_t> {
  [[maybe_unused]] static constexpr char kName[] = "uint16_t";
};

template <>
struct TypeTraits<uint32_t> {
  using XMMType = float;
  [[maybe_unused]] static constexpr char kName[] = "uint32_t";
};

template <>
struct TypeTraits<uint64_t> {
  using XMMType = double;
  [[maybe_unused]] static constexpr char kName[] = "uint64_t";
};

template <>
struct TypeTraits<int8_t> {
  [[maybe_unused]] static constexpr char kName[] = "int8_t";
};

template <>
struct TypeTraits<int16_t> {
  [[maybe_unused]] static constexpr char kName[] = "int16_t";
};

template <>
struct TypeTraits<int32_t> {
  using XMMType = float;
  [[maybe_unused]] static constexpr char kName[] = "int32_t";
};

template <>
struct TypeTraits<int64_t> {
  using XMMType = double;
  [[maybe_unused]] static constexpr char kName[] = "int64_t";
};

template <>
struct TypeTraits<intrinsics::Float32> {
  using XMMType = float;
  [[maybe_unused]] static constexpr char kName[] = "Float32";
};

template <>
struct TypeTraits<intrinsics::Float64> {
  using XMMType = double;
  [[maybe_unused]] static constexpr char kName[] = "Float64";
};

template <>
struct TypeTraits<SIMD128Register> {
  using XMMType = __m128;
  [[maybe_unused]] static constexpr char kName[] = "SIMD128Register";
};

template <>
struct TypeTraits<float> {
  [[maybe_unused]] static constexpr char kName[] = "float";
};

template <>
struct TypeTraits<double> {
  [[maybe_unused]] static constexpr char kName[] = "double";
};

template <>
struct TypeTraits<__m128> {
  [[maybe_unused]] static constexpr char kName[] = "__m128";
};

class OperandClass {
 public:
  class CL {
   public:
    using Type = uint8_t;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = true;
    [[maybe_unused]] static constexpr char kAsRegister = 'c';
  };

  class FLAGS {
   public:
    [[maybe_unused]] static constexpr bool kIsImplicitReg = true;
    [[maybe_unused]] static constexpr char kAsRegister = 0;
  };

  class EAX {
   public:
    using Type = uint32_t;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = true;
    [[maybe_unused]] static constexpr char kAsRegister = 'a';
  };

  class RAX {
   public:
    using Type = uint64_t;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = true;
    [[maybe_unused]] static constexpr char kAsRegister = 'a';
  };

  class ECX {
   public:
    using Type = uint32_t;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = true;
    [[maybe_unused]] static constexpr char kAsRegister = 'c';
  };

  class EDX {
   public:
    using Type = uint32_t;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = true;
    [[maybe_unused]] static constexpr char kAsRegister = 'd';
  };

  class FpReg32 {
   public:
    using Type = __m128;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = false;
    [[maybe_unused]] static constexpr char kAsRegister = 'x';
  };

  class FpReg64 {
   public:
    using Type = __m128;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = false;
    [[maybe_unused]] static constexpr char kAsRegister = 'x';
  };

  class GeneralReg8 {
   public:
    using Type = uint8_t;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = false;
    [[maybe_unused]] static constexpr char kAsRegister = 'q';
  };

  class GeneralReg32 {
   public:
    using Type = uint32_t;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = false;
    [[maybe_unused]] static constexpr char kAsRegister = 'r';
  };

  class GeneralReg64 {
   public:
    using Type = uint64_t;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = false;
    [[maybe_unused]] static constexpr char kAsRegister = 'r';
  };

  class VecReg128 {
   public:
    using Type = __m128;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = false;
    [[maybe_unused]] static constexpr char kAsRegister = 'x';
  };

  class XmmReg {
   public:
    using Type = __m128;
    [[maybe_unused]] static constexpr bool kIsImplicitReg = false;
    [[maybe_unused]] static constexpr char kAsRegister = 'x';
  };

  class Def;
  class DefEarlyClobber;
  class Use;
  class UseDef;
};

}  // namespace

}  // namespace x86

namespace intrinsics::bindings {

namespace {

enum SSERestrictionEnum : int {
  kNoCPUIDRestriction = 0,
  kHasLZCNT,
  kHasSSE3,
  kHasSSSE3,
  kHasSSE4_1,
  kHasSSE4_2,
  kHasAVX,
  kHasFMA,
  kHasFMA4,
  kIsAuthenticAMD
};
enum PreciseNanOperationsHandlingEnum : int {
  kNoNansOperation = 0,
  kPreciseNanOperationsHandling,
  kImpreciseNanOperationsHandling
};

}  // namespace

}  // namespace intrinsics::bindings

template <bool kSideEffects, typename... Types>
class GenerateAsmCall;

template <bool kSideEffects,
          typename... InputArguments,
          typename... OutputArguments,
          typename... Bindings>
class GenerateAsmCall<kSideEffects,
                      std::tuple<InputArguments...>,
                      std::tuple<OutputArguments...>,
                      Bindings...>
    final {
 private:
  // Note: we couldn't just make constructor which accepts random function because there are
  // certain functions which have many prototypes (e.g. Movd, naturally, could move from an
  // XMM register to a GP register or from GP register to XMM register).
  //
  // Bindings argument pack have all the required information needed to calculate EmitFunctionType,
  // but we need this dummy function to circumvent C++ restriction "lambda expressions are not
  // allowed in an unevaluated operand".
  //
  // EmitFunctionTypeHelper calculates and returns empty std::optional<EmitFunctionType> - where
  // EmitFunctionType is correct type of member function used to generate intrinsics with a given
  // Bindings...
  constexpr auto static EmitFunctionTypeHelper() {
    return std::apply(
        [](auto... args) {
          using EmitFunctionType = void (MacroAssembler<TextAssembler>::*)(decltype(args)...);
          return std::optional<EmitFunctionType>{};
        },
        std::tuple_cat([](auto arg) {
          using RegisterClass = typename decltype(arg)::RegisterClass;
          if constexpr (std::is_same_v<RegisterClass, x86::OperandClass::FLAGS> ||
                        RegisterClass::kIsImplicitReg) {
            return std::tuple{};
          } else if constexpr (RegisterClass::kAsRegister == 'x') {
            return std::tuple{TextAssembler::XMMRegister{}};
          } else {
            return std::tuple{TextAssembler::Register{}};
          }
        }(ArgTraits<Bindings>())...));
  }

  using EmitFunctionType = typename decltype(EmitFunctionTypeHelper())::value_type;

 public:
  GenerateAsmCall(
      EmitFunctionType emit,
      const char* name_,
      intrinsics::bindings::SSERestrictionEnum cpuid_restriction_,
      intrinsics::bindings::PreciseNanOperationsHandlingEnum precise_nan_operations_handling_)
      : cpuid_restriction(cpuid_restriction_),
        precise_nan_operations_handling(precise_nan_operations_handling_),
        name(name_),
        emit_{emit} {}
  const intrinsics::bindings::SSERestrictionEnum cpuid_restriction;
  const intrinsics::bindings::PreciseNanOperationsHandlingEnum precise_nan_operations_handling;
  const char* name;
  size_t GetArgumentsCount() { return sizeof...(InputArguments); }
  void GenerateFunctionHeader(FILE* out, int indent) {
    if (strchr(name, '<')) {
      fprintf(out, "template <>\n");
    }
    std::string prefix;
    if constexpr (sizeof...(InputArguments) == 0) {
      prefix = "inline void " + std::string(name) + "(";
    } else {
      const char* prefix_of_prefix = "inline std::tuple<";
      ((prefix += prefix_of_prefix + std::string(x86::TypeTraits<OutputArguments>::kName),
        prefix_of_prefix = ", "),
       ...);
      prefix += "> " + std::string(name) + "(";
    }
    std::size_t id = 0;
    std::vector<std::string> ins;
    (ins.push_back(std::string(x86::TypeTraits<InputArguments>::kName) + " in" +
                   std::to_string(id++)),
     ...);
    GenerateElementsList(out, indent, prefix, ") {", ins);
  }
  void GenerateFunctionBody(FILE* out, int indent) {
    // Declare out variables.
    GenerateOutputVariables(out, indent);
    // Declare temporary variables.
    GenerateTemporaries(out, indent);
    // We need "shadow variables" for ins of types: Float32, Float64 and SIMD128Register.
    // This is because assembler does not accept these arguments for XMMRegisters and
    // we couldn't use "float"/"double" function arguments because if ABI issues.
    GenerateInShadows(out, indent);
    // Even if we don't pass any registers we need to allocate at least one element.
    int register_numbers[sizeof...(Bindings) == 0 ? 1 : sizeof...(Bindings)];
    // Assign numbers to registers - we need to pass them to assembler and then, later,
    // to Generator of Input Variable line.
    AssignRegisterNumbers(register_numbers);
    // Print opening line for asm call.
    if constexpr (kSideEffects) {
      fprintf(out, "%*s__asm__ __volatile__(\n", indent, "");
    } else {
      fprintf(out, "%*s__asm__(\n", indent, "");
    }
    // Call text assembler to produce the body of an asm call.
    auto [need_gpr_macroassembler_mxcsr_scratch, need_gpr_macroassembler_constants] =
        CallTextAssembler(out, indent, register_numbers);
    // Assembler instruction outs.
    GenerateAssemblerOuts(out, indent);
    // Assembler instruction ins.
    GenerateAssemblerIns(out,
                         indent,
                         register_numbers,
                         need_gpr_macroassembler_mxcsr_scratch,
                         need_gpr_macroassembler_constants);
    // Close asm call.
    fprintf(out, "%*s);\n", indent, "");
    // Generate copies from shadows to outputs.
    GenerateOutShadows(out, indent);
    // Return value from function.
    if constexpr (sizeof...(OutputArguments) > 0) {
      std::vector<std::string> outs;
      for (std::size_t id = 0; id < sizeof...(OutputArguments); ++id) {
        outs.push_back("out" + std::to_string(id));
      }
      GenerateElementsList(out, indent, "return {", "};", outs);
    }
  }
  ~GenerateAsmCall() {}

 private:
  void GenerateOutputVariables(FILE* out, int indent) {
    std::size_t id = 0;
    (fprintf(out, "%*s%s out%zd;\n", indent, "", x86::TypeTraits<OutputArguments>::kName, id++),
     ...);
  }
  void GenerateTemporaries(FILE* out, int indent) {
    std::size_t id = 0;
    (
        [out, &id, indent](auto arg) {
          using RegisterClass = typename decltype(arg)::RegisterClass;
          if constexpr (!std::is_same_v<RegisterClass, x86::OperandClass::FLAGS>) {
            if constexpr (!HaveInput(arg.arg_info) && !HaveOutput(arg.arg_info)) {
              static_assert(std::is_same_v<typename decltype(arg)::Usage, x86::OperandClass::Def> ||
                            std::is_same_v<typename decltype(arg)::Usage,
                                           x86::OperandClass::DefEarlyClobber>);
              fprintf(out,
                      "%*s%s tmp%zd;\n",
                      indent,
                      "",
                      x86::TypeTraits<typename RegisterClass::Type>::kName,
                      id++);
            }
          }
        }(ArgTraits<Bindings>()),
        ...);
  }
  void GenerateInShadows(FILE* out, int indent) {
    (
        [out, indent](auto arg) {
          using RegisterClass = typename decltype(arg)::RegisterClass;
          if constexpr (RegisterClass::kAsRegister == 'r') {
            // TODO(b/138439904): remove when clang handling of 'r' constraint would be fixed.
            if constexpr (NeedInputShadow(arg)) {
              fprintf(
                  out, "%2$*1$suint32_t in%3$d_shadow = in%3$d;\n", indent, "", arg.arg_info.from);
            }
            if constexpr (NeedOutputShadow(arg)) {
              fprintf(out, "%*suint32_t out%d_shadow;\n", indent, "", arg.arg_info.to);
            }
          } else if constexpr (RegisterClass::kAsRegister == 'x') {
            if constexpr (HaveInput(arg.arg_info)) {
              using Type = std::tuple_element_t<arg.arg_info.from, std::tuple<InputArguments...>>;
              const char* type_name;
              const char* xmm_type_name;
              const char* expanded = "";
              if constexpr (std::is_same_v<Type, uint8_t>) {
                fprintf(out,
                        "%2$*1$suint64_t in%3$d_expanded = in%3$d;\n",
                        indent,
                        "",
                        arg.arg_info.from);
                type_name = x86::TypeTraits<uint64_t>::kName;
                xmm_type_name = x86::TypeTraits<typename x86::TypeTraits<uint64_t>::XMMType>::kName;
                expanded = "_expanded";
              } else {
                type_name = x86::TypeTraits<Type>::kName;
                xmm_type_name = x86::TypeTraits<typename x86::TypeTraits<Type>::XMMType>::kName;
              }
              fprintf(out, "%*s%s in%d_shadow;\n", indent, "", xmm_type_name, arg.arg_info.from);
              fprintf(out,
                      "%*sstatic_assert(sizeof(%s) == sizeof(%s));\n",
                      indent,
                      "",
                      type_name,
                      xmm_type_name);
              // Note: it's not safe to use bit_cast here till we have std::bit_cast from C++20.
              // If optimizer wouldn't be enabled (e.g. if code is compiled with -O0) then bit_cast
              // would use %st on 32-bit platform which destroys NaNs.
              fprintf(out,
                      "%2$*1$smemcpy(&in%3$d_shadow, &in%3$d%4$s, sizeof(%5$s));\n",
                      indent,
                      "",
                      arg.arg_info.from,
                      expanded,
                      xmm_type_name);
            }
            if constexpr (HaveOutput(arg.arg_info)) {
              using Type = std::tuple_element_t<arg.arg_info.to, std::tuple<OutputArguments...>>;
              const char* xmm_type_name =
                  x86::TypeTraits<typename x86::TypeTraits<Type>::XMMType>::kName;
              fprintf(out, "%*s%s out%d_shadow;\n", indent, "", xmm_type_name, arg.arg_info.to);
            }
          }
        }(ArgTraits<Bindings>()),
        ...);
  }
  static void AssignRegisterNumbers(int* register_numbers) {
    // Assign number for output (and temporary) arguments.
    std::size_t id = 0;
    int arg_counter = 0;
    (
        [&id, &arg_counter, &register_numbers](auto arg) {
          using RegisterClass = typename decltype(arg)::RegisterClass;
          if constexpr (!std::is_same_v<RegisterClass, x86::OperandClass::FLAGS>) {
            if constexpr (!std::is_same_v<typename decltype(arg)::Usage, x86::OperandClass::Use>) {
              register_numbers[arg_counter] = id++;
            }
            ++arg_counter;
          }
        }(ArgTraits<Bindings>()),
        ...);
    // Assign numbers for input arguments.
    arg_counter = 0;
    (
        [&id, &arg_counter, &register_numbers](auto arg) {
          using RegisterClass = typename decltype(arg)::RegisterClass;
          if constexpr (!std::is_same_v<RegisterClass, x86::OperandClass::FLAGS>) {
            if constexpr (std::is_same_v<typename decltype(arg)::Usage, x86::OperandClass::Use>) {
              register_numbers[arg_counter] = id++;
            }
            ++arg_counter;
          }
        }(ArgTraits<Bindings>()),
        ...);
  }
  auto CallTextAssembler(FILE* out, int indent, int* register_numbers) {
    MacroAssembler<TextAssembler> as(indent, out);
    int arg_counter = 0;
    (
        [&arg_counter, &as, register_numbers](auto arg) {
          using RegisterClass = typename decltype(arg)::RegisterClass;
          if constexpr (!std::is_same_v<RegisterClass, x86::OperandClass::FLAGS>) {
            if constexpr (RegisterClass::kIsImplicitReg) {
              if constexpr (RegisterClass::kAsRegister == 'a') {
                as.gpr_a = TextAssembler::Register(register_numbers[arg_counter]);
              } else if constexpr (RegisterClass::kAsRegister == 'c') {
                as.gpr_c = TextAssembler::Register(register_numbers[arg_counter]);
              } else {
                static_assert(RegisterClass::kAsRegister == 'd');
                as.gpr_d = TextAssembler::Register(register_numbers[arg_counter]);
              }
            }
            ++arg_counter;
          }
        }(ArgTraits<Bindings>()),
        ...);
    as.gpr_macroassembler_constants = TextAssembler::Register(arg_counter);
    arg_counter = 0;
    std::apply(
        emit_,
        std::tuple_cat(std::tuple<MacroAssembler<TextAssembler>&>(as),
                       [&arg_counter, register_numbers](auto arg) {
                         using RegisterClass = typename decltype(arg)::RegisterClass;
                         if constexpr (!std::is_same_v<RegisterClass, x86::OperandClass::FLAGS>) {
                           if constexpr (RegisterClass::kIsImplicitReg) {
                             ++arg_counter;
                             return std::tuple{};
                           } else {
                             return std::tuple{register_numbers[arg_counter++]};
                           }
                         } else {
                           return std::tuple{};
                         }
                       }(ArgTraits<Bindings>())...));
    // Verify CPU vendor and SSE restrictions.
    bool expect_lzcnt = false;
    bool expect_sse3 = false;
    bool expect_ssse3 = false;
    bool expect_sse4_1 = false;
    bool expect_sse4_2 = false;
    bool expect_avx = false;
    bool expect_fma = false;
    bool expect_fma4 = false;
    switch (cpuid_restriction) {
      case intrinsics::bindings::kHasLZCNT:
        expect_lzcnt = true;
        break;
      case intrinsics::bindings::kHasFMA:
      case intrinsics::bindings::kHasFMA4:
        if (cpuid_restriction == intrinsics::bindings::kHasFMA) {
          expect_fma = true;
        } else {
          expect_fma4 = true;
        }
        [[fallthrough]];
      case intrinsics::bindings::kHasAVX:
        expect_avx = true;
        [[fallthrough]];
      case intrinsics::bindings::kHasSSE4_2:
        expect_sse4_2 = true;
        [[fallthrough]];
      case intrinsics::bindings::kHasSSE4_1:
        expect_sse4_1 = true;
        [[fallthrough]];
      case intrinsics::bindings::kHasSSSE3:
        expect_ssse3 = true;
        [[fallthrough]];
      case intrinsics::bindings::kHasSSE3:
        expect_sse3 = true;
        [[fallthrough]];
      case intrinsics::bindings::kIsAuthenticAMD:
      case intrinsics::bindings::kNoCPUIDRestriction:; /* Do nothing - make compiler happy */
    }
    CHECK_EQ(expect_lzcnt, as.need_lzcnt);
    CHECK_EQ(expect_sse3, as.need_sse3);
    CHECK_EQ(expect_ssse3, as.need_ssse3);
    CHECK_EQ(expect_sse4_1, as.need_sse4_1);
    CHECK_EQ(expect_sse4_2, as.need_sse4_2);
    CHECK_EQ(expect_avx, as.need_avx);
    CHECK_EQ(expect_fma, as.need_fma);
    CHECK_EQ(expect_fma4, as.need_fma4);
    return std::tuple{as.need_gpr_macroassembler_mxcsr_scratch(),
                      as.need_gpr_macroassembler_constants()};
  }
  void GenerateAssemblerOuts(FILE* out, int indent) {
    std::vector<std::string> outs;
    int tmp_id = 0;
    (
        [&outs, &tmp_id](auto arg) {
          using RegisterClass = typename decltype(arg)::RegisterClass;
          if constexpr (!std::is_same_v<RegisterClass, x86::OperandClass::FLAGS> &&
                        !std::is_same_v<typename decltype(arg)::Usage, x86::OperandClass::Use>) {
            std::string out = "\"=";
            if constexpr (std::is_same_v<typename decltype(arg)::Usage,
                                         x86::OperandClass::DefEarlyClobber>) {
              out += "&";
            }
            out += RegisterClass::kAsRegister;
            if constexpr (HaveOutput(arg.arg_info)) {
              bool need_shadow = NeedOutputShadow(arg);
              out += "\"(out" + std::to_string(arg.arg_info.to) + (need_shadow ? "_shadow)" : ")");
            } else if constexpr (HaveInput(arg.arg_info)) {
              bool need_shadow = NeedInputShadow(arg);
              out += "\"(in" + std::to_string(arg.arg_info.from) + (need_shadow ? "_shadow)" : ")");
            } else {
              out += "\"(tmp" + std::to_string(tmp_id++) + ")";
            }
            outs.push_back(out);
          }
        }(ArgTraits<Bindings>()),
        ...);
    GenerateElementsList(out, indent, "  : ", "", outs);
  }
  void GenerateAssemblerIns(FILE* out,
                            int indent,
                            int* register_numbers,
                            bool need_gpr_macroassembler_mxcsr_scratch,
                            bool need_gpr_macroassembler_constants) {
    std::vector<std::string> ins;
    (
        [&ins](auto arg) {
          using RegisterClass = typename decltype(arg)::RegisterClass;
          if constexpr (!std::is_same_v<RegisterClass, x86::OperandClass::FLAGS> &&
                        std::is_same_v<typename decltype(arg)::Usage, x86::OperandClass::Use>) {
            ins.push_back("\"" + std::string(1, RegisterClass::kAsRegister) + "\"(in" +
                          std::to_string(arg.arg_info.from) +
                          (NeedInputShadow(arg) ? "_shadow)" : ")"));
          }
        }(ArgTraits<Bindings>()),
        ...);
    if (need_gpr_macroassembler_mxcsr_scratch) {
      ins.push_back("\"m\"(*&MxcsrStorage()), \"m\"(*&MxcsrStorage())");
    }
    if (need_gpr_macroassembler_constants) {
      ins.push_back(
          "\"m\"(*reinterpret_cast<const char*>(&constants_pool::kBerberisMacroAssemblerConstants))");
    }
    int arg_counter = 0;
    (
        [&ins, &arg_counter, register_numbers](auto arg) {
          using RegisterClass = typename decltype(arg)::RegisterClass;
          if constexpr (!std::is_same_v<RegisterClass, x86::OperandClass::FLAGS>) {
            if constexpr (HaveInput(arg.arg_info) &&
                          !std::is_same_v<typename decltype(arg)::Usage, x86::OperandClass::Use>) {
              ins.push_back("\"" + std::to_string(register_numbers[arg_counter]) + "\"(in" +
                            std::to_string(arg.arg_info.from) +
                            (NeedInputShadow(arg) ? "_shadow)" : ")"));
            }
            ++arg_counter;
          }
        }(ArgTraits<Bindings>()),
        ...);
    GenerateElementsList(out, indent, "  : ", "", ins);
  }
  void GenerateOutShadows(FILE* out, int indent) {
    (
        [out, indent](auto arg) {
          using RegisterClass = typename decltype(arg)::RegisterClass;
          if constexpr (RegisterClass::kAsRegister == 'r') {
            // TODO(b/138439904): remove when clang handling of 'r' constraint would be fixed.
            if constexpr (HaveOutput(arg.arg_info)) {
              using Type = std::tuple_element_t<arg.arg_info.to, std::tuple<OutputArguments...>>;
              if constexpr (sizeof(Type) == sizeof(uint8_t)) {
                fprintf(out, "%2$*1$sout%3$d = out%3$d_shadow;\n", indent, "", arg.arg_info.to);
              }
            }
          } else if constexpr (RegisterClass::kAsRegister == 'x') {
            if constexpr (HaveOutput(arg.arg_info)) {
              using Type = std::tuple_element_t<arg.arg_info.to, std::tuple<OutputArguments...>>;
              const char* type_name = x86::TypeTraits<Type>::kName;
              const char* xmm_type_name =
                  x86::TypeTraits<typename x86::TypeTraits<Type>::XMMType>::kName;
              fprintf(out,
                      "%*sstatic_assert(sizeof(%s) == sizeof(%s));\n",
                      indent,
                      "",
                      type_name,
                      xmm_type_name);
              // Note: it's not safe to use bit_cast here till we have std::bit_cast from C++20.
              // If optimizer wouldn't be enabled (e.g. if code is compiled with -O0) then bit_cast
              // would use %st on 32-bit platform which destroys NaNs.
              fprintf(out,
                      "%2$*1$smemcpy(&out%3$d, &out%3$d_shadow, sizeof(%4$s));\n",
                      indent,
                      "",
                      arg.arg_info.to,
                      xmm_type_name);
            }
          }
        }(ArgTraits<Bindings>()),
        ...);
  }
  void GenerateElementsList(FILE* out,
                            int indent,
                            const std::string& prefix,
                            const std::string& suffix,
                            const std::vector<std::string>& elements) {
    std::size_t length = prefix.length() + suffix.length();
    if (elements.size() == 0) {
      fprintf(out, "%*s%s%s\n", indent, "", prefix.c_str(), suffix.c_str());
      return;
    }
    for (const auto& element : elements) {
      length += element.length() + 2;
    }
    for (const auto& element : elements) {
      if (&element == &elements[0]) {
        fprintf(out, "%*s%s%s", indent, "", prefix.c_str(), element.c_str());
      } else {
        if (length <= 102) {
          fprintf(out, ", %s", element.c_str());
        } else {
          fprintf(out, ",\n%*s%s", static_cast<int>(prefix.length()) + indent, "", element.c_str());
        }
      }
    }
    fprintf(out, "%s\n", suffix.c_str());
  }
  template <typename Arg>
  static constexpr bool NeedInputShadow(Arg arg) {
    using RegisterClass = typename Arg::RegisterClass;
    // Without shadow clang silently converts 'r' restriction into 'q' restriction which
    // is wrong: if %ah or %bh is picked we would produce incorrect result here.
    // TODO(b/138439904): remove when clang handling of 'r' constraint would be fixed.
    if constexpr (RegisterClass::kAsRegister == 'r' && HaveInput(arg.arg_info)) {
      // Only 8-bit registers are special because each 16-bit registers include two of them
      // (%al/%ah, %cl/%ch, %dl/%dh, %bl/%bh).
      // Mix of 16-bit and 64-bit registers doesn't trigger bug in Clang.
      if constexpr (sizeof(
                        std::tuple_element_t<arg.arg_info.from, std::tuple<InputArguments...>>) ==
                    sizeof(uint8_t)) {
        return true;
      }
    } else if constexpr (RegisterClass::kAsRegister == 'x') {
      return true;
    }
    return false;
  }
  template <typename Arg>
  static constexpr bool NeedOutputShadow(Arg arg) {
    using RegisterClass = typename Arg::RegisterClass;
    // Without shadow clang silently converts 'r' restriction into 'q' restriction which
    // is wrong: if %ah or %bh is picked we would produce incorrect result here.
    // TODO(b/138439904): remove when clang handling of 'r' constraint would be fixed.
    if constexpr (RegisterClass::kAsRegister == 'r' && HaveOutput(arg.arg_info)) {
      // Only 8-bit registers are special because each some 16-bit registers include two of
      // them (%al/%ah, %cl/%ch, %dl/%dh, %bl/%bh).
      // Mix of 16-bit and 64-bit registers don't trigger bug in Clang.
      if constexpr (sizeof(std::tuple_element_t<arg.arg_info.to, std::tuple<OutputArguments...>>) ==
                    sizeof(uint8_t)) {
        return true;
      }
    } else if constexpr (RegisterClass::kAsRegister == 'x') {
      return true;
    }
    return false;
  }

  EmitFunctionType emit_;
};

#define INTRINSIC_FUNCTION_NAME(func, func_name) func_name
#include "make_intrinsics-inl.h"
#undef INTRINSIC_FUNCTION_NAME

void GenerateAsmCalls(FILE* out) {
  intrinsics::bindings::SSERestrictionEnum cpuid_restriction =
      intrinsics::bindings::kNoCPUIDRestriction;
  bool if_opened = false;
  std::string running_name;
  // Note: ProcessBindings is designed for TryInlineIntrinsic thus it processes bindings till
  // callback supplied returns “true” and then ProcessBindings returns success.
  //
  // But we want to unconditionally process all bindings, thus our callback always returns false
  // and thus ProcessBindings also returns false which subsequently ignore.
  ProcessBindings<MacroAssembler<berberis::TextAssembler>,
                  x86::OperandClass>([&running_name, &if_opened, &cpuid_restriction, out](
                                         auto&& asm_call_generator) {
    std::string full_name =
        std::string(asm_call_generator.name, std::strlen(asm_call_generator.name) - 1) +
        ", kUseCppImplementation>";
    if (size_t arguments_count = asm_call_generator.GetArgumentsCount()) {
      full_name += "(in0";
      for (size_t i = 1; i < arguments_count; ++i) {
        full_name += ", in" + std::to_string(i);
      }
      full_name += ")";
    } else {
      full_name += "()";
    }
    if (full_name != running_name) {
      if (if_opened) {
        if (cpuid_restriction != intrinsics::bindings::kNoCPUIDRestriction) {
          fprintf(out, "  } else {\n    return %s;\n", running_name.c_str());
          cpuid_restriction = intrinsics::bindings::kNoCPUIDRestriction;
        }
        if_opened = false;
        fprintf(out, "  }\n");
      }
      // Final line of function.
      fprintf(out, "};\n\n");
      asm_call_generator.GenerateFunctionHeader(out, 0);
      running_name = full_name;
    }
    if (asm_call_generator.cpuid_restriction != cpuid_restriction) {
      if (asm_call_generator.cpuid_restriction == intrinsics::bindings::kNoCPUIDRestriction) {
        fprintf(out, "  } else {\n");
      } else {
        if (if_opened) {
          fprintf(out, "  } else if (");
        } else {
          fprintf(out, "  if (");
          if_opened = true;
        }
        switch (asm_call_generator.cpuid_restriction) {
          case intrinsics::bindings::kIsAuthenticAMD:
            fprintf(out, "host_platform::kIsAuthenticAMD");
            break;
          case intrinsics::bindings::kHasLZCNT:
            fprintf(out, "host_platform::kHasLZCNT");
            break;
          case intrinsics::bindings::kHasSSE3:
            fprintf(out, "host_platform::kHasSSE3");
            break;
          case intrinsics::bindings::kHasSSSE3:
            fprintf(out, "host_platform::kHasSSSE3");
            break;
          case intrinsics::bindings::kHasSSE4_1:
            fprintf(out, "host_platform::kHasSSE4_1");
            break;
          case intrinsics::bindings::kHasSSE4_2:
            fprintf(out, "host_platform::kHasSSE4_2");
            break;
          case intrinsics::bindings::kHasAVX:
            fprintf(out, "host_platform::kHasAVX");
            break;
          case intrinsics::bindings::kHasFMA:
            fprintf(out, "host_platform::kHasFMA");
            break;
          case intrinsics::bindings::kHasFMA4:
            fprintf(out, "host_platform::kHasFMA4");
            break;
          case intrinsics::bindings::kNoCPUIDRestriction:; /* Do nothing - make compiler happy */
        }
        fprintf(out, ") {\n");
      }
      cpuid_restriction = asm_call_generator.cpuid_restriction;
    }
    asm_call_generator.GenerateFunctionBody(out, 2 + 2 * if_opened);
    return false;  // Returning true would mean that we finished processing bindings.
  });
  if (if_opened) {
    fprintf(out, "  }\n");
  }
  // Final line of function.
  fprintf(out, "};\n\n");
}

}  // namespace berberis

int main(int argc, char* argv[]) {
  FILE* out = argc > 1 ? fopen(argv[1], "w") : stdout;
  fprintf(out,
          R"STRING(
// This file automatically generated by make_intrinsics.cc
// DO NOT EDIT!

#ifndef %1$s_%2$s_INTRINSICS_INTRINSICS_H_
#define %1$s_%2$s_INTRINSICS_INTRINSICS_H_

#include <xmmintrin.h>

#include "berberis/runtime_primitives/platform.h"
#include "%2$s/intrinsics/common/intrinsics.h"
#include "%2$s/intrinsics/vector_intrinsics.h"

namespace berberis::constants_pool {

struct MacroAssemblerConstants;

extern const MacroAssemblerConstants kBerberisMacroAssemblerConstants
    __attribute__((visibility("hidden")));

}  // namespace berberis::constants_pool

namespace %2$s {

namespace constants_pool {

%3$s

}  // namespace constants_pool

namespace intrinsics {

class MxcsrStorage {
 public:
  uint32_t* operator&() { return &storage_; }

 private:
  uint32_t storage_;
)STRING",
          berberis::TextAssembler::kArchName,
          berberis::TextAssembler::kNamespaceName,
          strcmp(berberis::TextAssembler::kNamespaceName, "berberis")
              ? "using berberis::constants_pool::kBerberisMacroAssemblerConstants;"
              : "");

  berberis::GenerateAsmCalls(out);
  berberis::MakeExtraGuestFunctions(out);

  fprintf(out,
          R"STRING(
}  // namespace intrinsics

}  // namespace %2$s

#endif /* %1$s_%2$s_INTRINSICS_INTRINSICS_H_ */
)STRING",
          berberis::TextAssembler::kArchName,
          berberis::TextAssembler::kNamespaceName);

  fclose(out);
  return 0;
}
