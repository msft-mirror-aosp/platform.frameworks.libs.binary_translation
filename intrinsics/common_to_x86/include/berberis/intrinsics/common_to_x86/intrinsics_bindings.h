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
};

class RAX {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'a';
};

class CL {
 public:
  using Type = uint8_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'c';
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
};

class RCX {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'c';
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
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'd';
};

class RDX {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 'd';
};

class GeneralReg8 {
 public:
  using Type = uint8_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'q';
};

class GeneralReg16 {
 public:
  using Type = uint16_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'r';
};

class GeneralReg32 {
 public:
  using Type = uint32_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'r';
};

class GeneralReg64 {
 public:
  using Type = uint64_t;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'r';
};

class FLAGS {
 public:
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = true;
  static constexpr char kAsRegister = 0;
};

class FpReg32 {
 public:
  using Type = __m128;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'x';
};

class FpReg64 {
 public:
  using Type = __m128;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'x';
};

class VecReg128 {
 public:
  using Type = __m128;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'x';
};

class XmmReg {
 public:
  using Type = __m128;
  static constexpr bool kIsImmediate = false;
  static constexpr bool kIsImplicitReg = false;
  static constexpr char kAsRegister = 'x';
};

// Tag classes. They are never instantioned, only used as tags to pass information about bindings.
class Def;
class DefEarlyClobber;
class Use;
class UseDef;

enum CPUIDRestriction : int {
  kNoCPUIDRestriction = 0,
  kHasX87,
  kHasFXSAVE,
  kHasCMOV,
  kHas3DNOW,
  kHas3DNOWP,
  kHasCMPXCHG8B,
  kHasCMPXCHG16B,
  kHasLZCNT,
  kHasBMI,
  kHasBMI2,
  kHasADX,
  kHasTBM,
  kHasRDSEED,
  kHasSERIALIZE,
  kHasSSE,
  kHasSSE2,
  kHasSSE3,
  kHasSSSE3,
  kHasSSE4a,
  kHasSSE4_1,
  kHasSSE4_2,
  kHasFMA,
  kHasFMA4,
  kHasF16C,
  kHasCLMUL,
  kHasSHA,
  kHasAES,
  kHasAVX,
  kHasAESAVX,
  kHasAVX2,
  kHasVAES,
  kHasAVX5124FMAPS,
  kHasAVX5124VNNIW,
  kHasAVX512BF16,
  kHasAVX512BITALG,
  kHasAVX512BW,
  kHasAVX512CD,
  kHasAVX512DQ,
  kHasAVX512ER,
  kHasAVX512F,
  kHasAVX512FP16,
  kHasAVX512IFMA,
  kHasAVX512PF,
  kHasAVX512VBMI,
  kHasAVX512VBMI2,
  kHasAVX512VL,
  kHasAVX512VNNI,
  kHasAVX512VPOPCNTDQ,
  kHasAMXBF16,
  kHasAMXFP16,
  kHasAMXINT8,
  kHasAMXTILE,
  kIsAuthenticAMD
};

enum PreciseNanOperationsHandling : int {
  kNoNansOperation = 0,
  kPreciseNanOperationsHandling,
  kImpreciseNanOperationsHandling
};

}  // namespace berberis::intrinsics::bindings

#endif  // BERBERIS_INTRINSICS_COMMON_TO_X86_INTRINSICS_BINDINGS_H_
