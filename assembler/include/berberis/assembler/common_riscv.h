/*
 * Copyright (C) 2024 The Android Open Source Project
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

#ifndef BERBERIS_ASSEMBLER_COMMON_RISCV_H_
#define BERBERIS_ASSEMBLER_COMMON_RISCV_H_

#include <cstddef>  // std::size_t
#include <cstdint>
#include <type_traits>  // std::enable_if, std::is_integral

#include "berberis/assembler/common.h"
#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"

namespace berberis {

// AssemblerRiscV includes implementation of most Risc V assembler instructions.
//
// RV32 and RV64 assemblers are nearly identical, but difference lies in handling
// of some instructions: RV32 uses certain encodings differently to handle compressed
// instructions, while RV64 adds some extra instructions to handle 32bit quantities
// (*not* 64bit quantities as the name implies, instead there are width-native instructions
// and extra 32bit ones for RV64).
//
// To handle that difference efficiently AssemblerRiscV is CRTP class: it's parameterized
// by its own descendant and pull certain functions from its implementation.

namespace rv32e {

class Assembler;

}  // namespace rv32e

namespace rv32i {

class Assembler;

}  // namespace rv32i

namespace rv64i {

class Assembler;

}  // namespace rv64i

template <typename Assembler>
class AssemblerRiscV : public AssemblerBase {
 public:
  explicit AssemblerRiscV(MachineCode* code) : AssemblerBase(code) {}

  class Register {
    constexpr bool operator==(const Register& reg) const { return num_ == reg.num_; }
    constexpr bool operator!=(const Register& reg) const { return num_ != reg.num_; }
    constexpr uint8_t GetPhysicalIndex() { return num_; }
    friend constexpr uint8_t ValueForFmtSpec(Register value) { return value.num_; }
    friend class AssemblerRiscV<Assembler>;
    friend class rv32e::Assembler;
    friend class rv32i::Assembler;
    friend class rv64i::Assembler;

   private:
    constexpr Register(uint8_t num) : num_(num) {}
    uint8_t num_;
  };

  // Note: register x0, technically, can be specified in assembler even if it doesn't exist
  // as separate hardware register. It even have alias “zero” even in clang assembler.
  static constexpr Register x0{0};
  static constexpr Register x1{1};
  static constexpr Register x2{2};
  static constexpr Register x3{3};
  static constexpr Register x4{4};
  static constexpr Register x5{5};
  static constexpr Register x6{6};
  static constexpr Register x7{7};
  static constexpr Register x8{8};
  static constexpr Register x9{9};
  static constexpr Register x10{10};
  static constexpr Register x11{11};
  static constexpr Register x12{12};
  static constexpr Register x13{13};
  static constexpr Register x14{14};
  static constexpr Register x15{15};
  static constexpr Register x16{16};
  static constexpr Register x17{17};
  static constexpr Register x18{18};
  static constexpr Register x19{19};
  static constexpr Register x20{20};
  static constexpr Register x21{21};
  static constexpr Register x22{22};
  static constexpr Register x23{23};
  static constexpr Register x24{24};
  static constexpr Register x25{25};
  static constexpr Register x26{26};
  static constexpr Register x27{27};
  static constexpr Register x28{28};
  static constexpr Register x29{29};
  static constexpr Register x30{30};
  static constexpr Register x31{31};

  // Aliases
  static constexpr Register no_register{0x80};
  static constexpr Register zero{0};

  // Immediates are kept in a form ready to be used with emitter.
  class Immediate;

  // Don't use templates here to enable implicit conversions.
  static constexpr std::optional<Immediate> make_immediate(int8_t value);
  static constexpr std::optional<Immediate> make_immediate(uint8_t value);
  static constexpr std::optional<Immediate> make_immediate(int16_t value);
  static constexpr std::optional<Immediate> make_immediate(uint16_t value);
  static constexpr std::optional<Immediate> make_immediate(int32_t value);
  static constexpr std::optional<Immediate> make_immediate(uint32_t value);
  static constexpr std::optional<Immediate> make_immediate(int64_t value);
  static constexpr std::optional<Immediate> make_immediate(uint64_t value);

 private:
  // RawImmediate is used to bypass checks in constructor. It's not supposed to be used directly.
  class RawImmediate {
   private:
    friend class Immediate;
    friend class AssemblerRiscV;

    constexpr RawImmediate(int32_t value) : value_(value) {}
    int32_t value_;
  };

 public:
  class Immediate {
   public:
    static constexpr int32_t kMask = static_cast<int32_t>(0xfff0'0000);
#define BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(IntType) \
  constexpr Immediate(IntType value) : Immediate(MakeRaw(value)) { CHECK(AccetableValue(value)); }
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(int8_t)
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(uint8_t)
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(int16_t)
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(uint16_t)
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(int32_t)
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(uint32_t)
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(int64_t)
    BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR(uint64_t)
#undef BERBERIS_DEFINE_IMMEDIATE_CONSTRUCTOR

    friend class AssemblerRiscV;

   private:
    constexpr Immediate(RawImmediate raw) : value_(raw.value_) {}

    // Return true if value would fit into immediate.
    template <typename IntType>
    static constexpr bool AccetableValue(IntType value);
    // Make RawImmediate from immediate value.
    // Note: value is not checked for correctness here! Public interface is make_immediate factory.
    template <typename IntType>
    static constexpr RawImmediate MakeRaw(IntType value);

    int32_t value_;
  };

  // Macro operations.
  void Finalize() { ResolveJumps(); }

  void ResolveJumps();

// Instructions.
#include "berberis/assembler/gen_assembler_common_riscv-inl.h"  // NOLINT generated file!

  AssemblerRiscV() = delete;
  AssemblerRiscV(const AssemblerRiscV&) = delete;
  AssemblerRiscV(AssemblerRiscV&&) = delete;
  void operator=(const AssemblerRiscV&) = delete;
  void operator=(AssemblerRiscV&&) = delete;
};

#define BERBERIS_DEFINE_MAKE_IMMEDIATE(IntType)                             \
  template <typename Assembler>                                             \
  constexpr std::optional<typename AssemblerRiscV<Assembler>::Immediate>    \
  AssemblerRiscV<Assembler>::make_immediate(IntType value) {                \
    if (!AssemblerRiscV<Assembler>::Immediate::AccetableValue(value)) {     \
      return {};                                                            \
    }                                                                       \
    return Immediate{AssemblerRiscV<Assembler>::Immediate::MakeRaw(value)}; \
  }
BERBERIS_DEFINE_MAKE_IMMEDIATE(int8_t)
BERBERIS_DEFINE_MAKE_IMMEDIATE(uint8_t)
BERBERIS_DEFINE_MAKE_IMMEDIATE(int16_t)
BERBERIS_DEFINE_MAKE_IMMEDIATE(uint16_t)
BERBERIS_DEFINE_MAKE_IMMEDIATE(int32_t)
BERBERIS_DEFINE_MAKE_IMMEDIATE(uint32_t)
BERBERIS_DEFINE_MAKE_IMMEDIATE(int64_t)
BERBERIS_DEFINE_MAKE_IMMEDIATE(uint64_t)
#undef BERBERIS_DEFINE_MAKE_IMMEDIATE

// Return true if value would fit into immediate.
template <typename Assembler>
template <typename IntType>
constexpr bool AssemblerRiscV<Assembler>::Immediate::AccetableValue(IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  // I-immediate accepts 12 bits, but encodes signed values, that's why we only may accept low
  // 11 bits of any unsigned value.
  // Encode mask as the largest accepted value plus one and cut it to IntType size.
  constexpr uint64_t kUnsigned64bitInputMask = 0xffff'ffff'ffff'f800;
  if constexpr (!std::is_signed_v<IntType>) {
    constexpr IntType kUnsignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kUnsignedInputMask) == IntType{0};
  } else {
    // For signed values we accept the same values as for unsigned case, but also accept
    // values that have all bits in kUnsignedInputMask set.
    // I-immediate compresses these into one single sign bit.
    constexpr IntType kSignedInputMask = static_cast<IntType>(kUnsigned64bitInputMask);
    return static_cast<IntType>(value & kSignedInputMask) == IntType{0} ||
           static_cast<IntType>(value & kSignedInputMask) == kSignedInputMask;
  }
}

// Make RawImmediate from immediate value.
// Note: value is not checked for correctness here! Public interface is make_immediate factory.
template <typename Assembler>
template <typename IntType>
constexpr AssemblerRiscV<Assembler>::RawImmediate AssemblerRiscV<Assembler>::Immediate::MakeRaw(
    IntType value) {
  static_assert(std::is_integral_v<IntType>);
  static_assert(sizeof(IntType) <= sizeof(uint64_t));
  return static_cast<int32_t>(value) << 20;
}

}  // namespace berberis

#endif  // BERBERIS_ASSEMBLER_COMMON_X86_H_
