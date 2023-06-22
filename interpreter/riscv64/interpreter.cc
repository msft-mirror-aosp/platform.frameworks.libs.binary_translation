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

#include "berberis/interpreter/riscv64/interpreter.h"

#include <cfenv>
#include <cstdint>
#include <cstring>

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"
#include "berberis/base/logging.h"
#include "berberis/base/macros.h"
#include "berberis/decoder/riscv64/decoder.h"
#include "berberis/decoder/riscv64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/intrinsics/guest_fp_flags.h"  // ToHostRoundingMode
#include "berberis/intrinsics/intrinsics.h"
#include "berberis/intrinsics/intrinsics_float.h"
#include "berberis/intrinsics/type_traits.h"
#include "berberis/kernel_api/run_guest_syscall.h"

#include "atomics.h"
#include "fp_regs.h"

namespace berberis {

namespace {

class Interpreter {
 public:
  using Decoder = Decoder<SemanticsPlayer<Interpreter>>;
  using Register = uint64_t;
  using FpRegister = uint64_t;
  using Float32 = intrinsics::Float32;
  using Float64 = intrinsics::Float64;

  explicit Interpreter(ThreadState* state) : state_(state), branch_taken_(false) {}

  //
  // Instruction implementations.
  //

  Register Csr(Decoder::CsrOpcode opcode, Register arg, Decoder::CsrRegister csr) {
    Register (*UpdateStatus)(Register arg, Register original_csr_value);
    switch (opcode) {
      case Decoder::CsrOpcode::kCsrrw:
        UpdateStatus = [](Register arg, Register /*original_csr_value*/) { return arg; };
        break;
      case Decoder::CsrOpcode::kCsrrs:
        UpdateStatus = [](Register arg, Register original_csr_value) {
          return arg | original_csr_value;
        };
        break;
      case Decoder::CsrOpcode::kCsrrc:
        UpdateStatus = [](Register arg, Register original_csr_value) {
          return ~arg & original_csr_value;
        };
        break;
      default:
        Unimplemented();
        return {};
    }
    Register result;
    switch (csr) {
      case Decoder::CsrRegister::kFrm:
        result = state_->cpu.frm;
        arg = UpdateStatus(arg, result);
        state_->cpu.frm = arg;
        if (arg <= FPFlags::RM_MAX) {
          std::fesetround(intrinsics::ToHostRoundingMode(arg));
        }
        break;
      default:
        Unimplemented();
        return {};
    }
    return result;
  }

  Register Csr(Decoder::CsrImmOpcode opcode, uint8_t imm, Decoder::CsrRegister csr) {
    return Csr(Decoder::CsrOpcode(opcode), imm, csr);
  }

  // Note: we prefer not to use C11/C++ atomic_thread_fence or even gcc/clang builtin
  // __atomic_thread_fence because all these function rely on the fact that compiler never uses
  // non-temporal loads and stores and only issue “mfence” when sequentially consistent ordering is
  // requested. They never issue “lfence” or “sfence”.
  // Instead we pull the page from Linux's kernel book and map read ordereding to “lfence”, write
  // ordering to “sfence” and read-write ordering to “mfence”.
  // This can be important in the future if we would start using nontemporal moves in manually
  // created assembly code.
  // Ordering affecting I/O devices is not relevant to user-space code thus we just ignore bits
  // related to devices I/O.
  void Fence(Decoder::FenceOpcode /*opcode*/,
             Register /*src*/,
             bool sw,
             bool sr,
             bool /*so*/,
             bool /*si*/,
             bool pw,
             bool pr,
             bool /*po*/,
             bool /*pi*/) {
    bool read_fence = sr | pr;
    bool write_fence = sw | pw;
    // Two types of fences (total store ordering fence and normal fence) are supposed to be
    // processed differently, but only for the “read_fence && write_fence” case (otherwise total
    // store ordering fence becomes normal fence for the “forward compatibility”), yet because x86
    // doesn't distinguish between these two types of fences and since we are supposed to map all
    // not-yet defined fences to normal fence (again, for the “forward compatibility”) it's Ok to
    // just ignore opcode field.
    if (read_fence) {
      if (write_fence) {
        asm volatile("mfence" ::: "memory");
      } else {
        asm volatile("lfence" ::: "memory");
      }
    } else if (write_fence) {
      asm volatile("sfence" ::: "memory");
    }
    return;
  }

  void FenceI(Register /*arg*/, int16_t /*imm*/) {
    // For interpreter-only mode we don't need to do anything here, but when we will have a
    // translator we would need to flush caches here.
  }

  Register Op(Decoder::OpOpcode opcode, Register arg1, Register arg2) {
    using uint128_t = unsigned __int128;
    switch (opcode) {
      case Decoder::OpOpcode::kAdd:
        return arg1 + arg2;
      case Decoder::OpOpcode::kSub:
        return arg1 - arg2;
      case Decoder::OpOpcode::kAnd:
        return arg1 & arg2;
      case Decoder::OpOpcode::kOr:
        return arg1 | arg2;
      case Decoder::OpOpcode::kXor:
        return arg1 ^ arg2;
      case Decoder::OpOpcode::kSll:
        return arg1 << arg2;
      case Decoder::OpOpcode::kSrl:
        return arg1 >> arg2;
      case Decoder::OpOpcode::kSra:
        return bit_cast<int64_t>(arg1) >> arg2;
      case Decoder::OpOpcode::kSlt:
        return bit_cast<int64_t>(arg1) < bit_cast<int64_t>(arg2) ? 1 : 0;
      case Decoder::OpOpcode::kSltu:
        return arg1 < arg2 ? 1 : 0;
      case Decoder::OpOpcode::kMul:
        return arg1 * arg2;
      case Decoder::OpOpcode::kMulh:
        return (__int128{bit_cast<int64_t>(arg1)} * __int128{bit_cast<int64_t>(arg2)}) >> 64;
      case Decoder::OpOpcode::kMulhsu:
        return (__int128{bit_cast<int64_t>(arg1)} * uint128_t{arg2}) >> 64;
      case Decoder::OpOpcode::kMulhu:
        return (uint128_t{arg1} * uint128_t{arg2}) >> 64;
      case Decoder::OpOpcode::kDiv:
        return bit_cast<int64_t>(arg1) / bit_cast<int64_t>(arg2);
      case Decoder::OpOpcode::kDivu:
        return arg1 / arg2;
      case Decoder::OpOpcode::kRem:
        return bit_cast<int64_t>(arg1) % bit_cast<int64_t>(arg2);
      case Decoder::OpOpcode::kRemu:
        return arg1 % arg2;
      default:
        Unimplemented();
        return {};
    }
  }

  Register Op32(Decoder::Op32Opcode opcode, Register arg1, Register arg2) {
    switch (opcode) {
      case Decoder::Op32Opcode::kAddw:
        return int32_t(arg1) + int32_t(arg2);
      case Decoder::Op32Opcode::kSubw:
        return int32_t(arg1) - int32_t(arg2);
      case Decoder::Op32Opcode::kSllw:
        return int32_t(arg1) << int32_t(arg2);
      case Decoder::Op32Opcode::kSrlw:
        return bit_cast<int32_t>(uint32_t(arg1) >> uint32_t(arg2));
      case Decoder::Op32Opcode::kSraw:
        return int32_t(arg1) >> int32_t(arg2);
      case Decoder::Op32Opcode::kMulw:
        return int32_t(arg1) * int32_t(arg2);
      case Decoder::Op32Opcode::kDivw:
        return int32_t(arg1) / int32_t(arg2);
      case Decoder::Op32Opcode::kDivuw:
        return static_cast<int32_t>(uint32_t(arg1) / uint32_t(arg2));
      case Decoder::Op32Opcode::kRemw:
        return int32_t(arg1) % int32_t(arg2);
      case Decoder::Op32Opcode::kRemuw:
        return static_cast<int32_t>(uint32_t(arg1) % uint32_t(arg2));
      default:
        Unimplemented();
        return {};
    }
  }

  Register Amo(Decoder::AmoOpcode opcode, Register arg1, Register arg2, bool aq, bool rl) {
    switch (opcode) {
      // TODO(b/287347834): Implement reservation semantics when it's added to runtime_primitives.
      case Decoder::AmoOpcode::kLrW:
        return Load<int32_t>(ToHostAddr<void>(arg1));
      case Decoder::AmoOpcode::kLrD:
        return Load<uint64_t>(ToHostAddr<void>(arg1));
      case Decoder::AmoOpcode::kScW:
        Store<uint32_t>(ToHostAddr<void>(arg1), arg2);
        return 0;
      case Decoder::AmoOpcode::kScD:
        Store<uint64_t>(ToHostAddr<void>(arg1), arg2);
        return 0;

      case Decoder::AmoOpcode::kAmoswapW:
        return AtomicExchange<int32_t>(arg1, arg2, aq, rl);
      case Decoder::AmoOpcode::kAmoswapD:
        return AtomicExchange<int64_t>(arg1, arg2, aq, rl);

      case Decoder::AmoOpcode::kAmoaddW:
        return AtomicAdd<int32_t>(arg1, arg2, aq, rl);
      case Decoder::AmoOpcode::kAmoaddD:
        return AtomicAdd<int64_t>(arg1, arg2, aq, rl);

      case Decoder::AmoOpcode::kAmoxorW:
        return AtomicXor<int32_t>(arg1, arg2, aq, rl);
      case Decoder::AmoOpcode::kAmoxorD:
        return AtomicXor<int64_t>(arg1, arg2, aq, rl);

      case Decoder::AmoOpcode::kAmoandW:
        return AtomicAnd<int32_t>(arg1, arg2, aq, rl);
      case Decoder::AmoOpcode::kAmoandD:
        return AtomicAnd<int64_t>(arg1, arg2, aq, rl);

      case Decoder::AmoOpcode::kAmoorW:
        return AtomicOr<int32_t>(arg1, arg2, aq, rl);
      case Decoder::AmoOpcode::kAmoorD:
        return AtomicOr<int64_t>(arg1, arg2, aq, rl);

      case Decoder::AmoOpcode::kAmominW:
        return AtomicMin<int32_t>(arg1, arg2, aq, rl);
      case Decoder::AmoOpcode::kAmominD:
        return AtomicMin<int64_t>(arg1, arg2, aq, rl);

      case Decoder::AmoOpcode::kAmomaxW:
        return AtomicMax<int32_t>(arg1, arg2, aq, rl);
      case Decoder::AmoOpcode::kAmomaxD:
        return AtomicMax<int64_t>(arg1, arg2, aq, rl);

      case Decoder::AmoOpcode::kAmominuW:
        return AtomicMinu<uint32_t>(arg1, arg2, aq, rl);
      case Decoder::AmoOpcode::kAmominuD:
        return AtomicMinu<uint64_t>(arg1, arg2, aq, rl);

      case Decoder::AmoOpcode::kAmomaxuW:
        return AtomicMaxu<uint32_t>(arg1, arg2, aq, rl);
      case Decoder::AmoOpcode::kAmomaxuD:
        return AtomicMaxu<uint64_t>(arg1, arg2, aq, rl);

      default:
        Unimplemented();
        return {};
    }
  }

  Register Load(Decoder::LoadOperandType operand_type, Register arg, int16_t offset) {
    void* ptr = ToHostAddr<void>(arg + offset);
    switch (operand_type) {
      case Decoder::LoadOperandType::k8bitUnsigned:
        return Load<uint8_t>(ptr);
      case Decoder::LoadOperandType::k16bitUnsigned:
        return Load<uint16_t>(ptr);
      case Decoder::LoadOperandType::k32bitUnsigned:
        return Load<uint32_t>(ptr);
      case Decoder::LoadOperandType::k64bit:
        return Load<uint64_t>(ptr);
      case Decoder::LoadOperandType::k8bitSigned:
        return Load<int8_t>(ptr);
      case Decoder::LoadOperandType::k16bitSigned:
        return Load<int16_t>(ptr);
      case Decoder::LoadOperandType::k32bitSigned:
        return Load<int32_t>(ptr);
      default:
        Unimplemented();
        return {};
    }
  }

  FpRegister LoadFp(Decoder::FloatOperandType opcode, Register arg, int16_t offset) {
    void* ptr = ToHostAddr<void>(arg + offset);
    switch (opcode) {
      case Decoder::FloatOperandType::kFloat:
        return LoadFp<float>(ptr);
      case Decoder::FloatOperandType::kDouble:
        return LoadFp<double>(ptr);
      default:
        Unimplemented();
        return {};
    }
  }

  template <typename TargetOperandType, typename SourceOperandType>
  auto Fcvt(uint8_t rm, SourceOperandType arg) {
    // TODO(265372622): handle rm properly in integer-to-float and float-to-integer cases.
    if constexpr (std::is_integral_v<TargetOperandType>) {
      TargetOperandType result = static_cast<TargetOperandType>(arg);
      return static_cast<std::make_signed_t<TargetOperandType>>(result);
    } else if constexpr (std::is_integral_v<SourceOperandType>) {
      TargetOperandType result = static_cast<TargetOperandType>(arg);
      return result;
    } else if constexpr (sizeof(TargetOperandType) > sizeof(SourceOperandType)) {
      // Conversion from narrow type to wide one ignores rm because all possible values from narrow
      // type fit in the wide type.
      return TargetOperandType(arg);
    } else {
      return intrinsics::ExecuteFloatOperation<TargetOperandType>(
          rm,
          state_->cpu.frm,
          [](auto x) { return typename TypeTraits<decltype(x)>::Narrow(x); },
          arg);
    }
  }

  FpRegister Fcvt(Decoder::FloatOperandType target_operand_size,
                  Decoder::FloatOperandType source_operand_size,
                  uint8_t rm,
                  FpRegister arg) {
    if (target_operand_size == Decoder::FloatOperandType::kFloat &&
        source_operand_size == Decoder::FloatOperandType::kDouble) {
      return FloatToFPReg(Fcvt<Float32, Float64>(rm, FPRegToFloat<Float64>(arg)));
    }
    if (target_operand_size == Decoder::FloatOperandType::kDouble &&
        source_operand_size == Decoder::FloatOperandType::kFloat) {
      return FloatToFPReg(Fcvt<Float64, Float32>(rm, FPRegToFloat<Float32>(arg)));
    }
    Unimplemented();
    return {};
  }

  Register Fcvt(Decoder::FcvtOperandType target_operand_size,
                Decoder::FloatOperandType source_operand_size,
                uint8_t rm,
                FpRegister arg) {
    switch (source_operand_size) {
      case Decoder::FloatOperandType::kFloat:
        switch (target_operand_size) {
          case Decoder::FcvtOperandType::k32bitSigned:
            return Fcvt<int32_t, Float32>(rm, FPRegToFloat<Float32>(arg));
          case Decoder::FcvtOperandType::k32bitUnsigned:
            return Fcvt<uint32_t, Float32>(rm, FPRegToFloat<Float32>(arg));
          case Decoder::FcvtOperandType::k64bitSigned:
            return Fcvt<int64_t, Float32>(rm, FPRegToFloat<Float32>(arg));
          case Decoder::FcvtOperandType::k64bitUnsigned:
            return Fcvt<uint64_t, Float32>(rm, FPRegToFloat<Float32>(arg));
          default:
            Unimplemented();
            return {};
        }
      case Decoder::FloatOperandType::kDouble:
        switch (target_operand_size) {
          case Decoder::FcvtOperandType::k32bitSigned:
            return Fcvt<int32_t, Float64>(rm, FPRegToFloat<Float64>(arg));
          case Decoder::FcvtOperandType::k32bitUnsigned:
            return Fcvt<uint32_t, Float64>(rm, FPRegToFloat<Float64>(arg));
          case Decoder::FcvtOperandType::k64bitSigned:
            return Fcvt<int64_t, Float64>(rm, FPRegToFloat<Float64>(arg));
          case Decoder::FcvtOperandType::k64bitUnsigned:
            return Fcvt<uint64_t, Float64>(rm, FPRegToFloat<Float64>(arg));
          default:
            Unimplemented();
            return {};
        }
      default:
        Unimplemented();
        return {};
    }
  }

  FpRegister Fcvt(Decoder::FloatOperandType target_operand_size,
                  Decoder::FcvtOperandType source_operand_size,
                  uint8_t rm,
                  Register arg) {
    switch (target_operand_size) {
      case Decoder::FloatOperandType::kFloat:
        switch (source_operand_size) {
          case Decoder::FcvtOperandType::k32bitSigned:
            return FloatToFPReg(Fcvt<Float32, int32_t>(rm, arg));
          case Decoder::FcvtOperandType::k32bitUnsigned:
            return FloatToFPReg(Fcvt<Float32, uint32_t>(rm, arg));
          case Decoder::FcvtOperandType::k64bitSigned:
            return FloatToFPReg(Fcvt<Float32, int64_t>(rm, arg));
          case Decoder::FcvtOperandType::k64bitUnsigned:
            return FloatToFPReg(Fcvt<Float32, uint64_t>(rm, arg));
          default:
            Unimplemented();
            return {};
        }
      case Decoder::FloatOperandType::kDouble:
        switch (source_operand_size) {
          case Decoder::FcvtOperandType::k32bitSigned:
            return FloatToFPReg(Fcvt<Float64, int32_t>(rm, arg));
          case Decoder::FcvtOperandType::k32bitUnsigned:
            return FloatToFPReg(Fcvt<Float64, uint32_t>(rm, arg));
          case Decoder::FcvtOperandType::k64bitSigned:
            return FloatToFPReg(Fcvt<Float64, int64_t>(rm, arg));
          case Decoder::FcvtOperandType::k64bitUnsigned:
            return FloatToFPReg(Fcvt<Float64, uint64_t>(rm, arg));
          default:
            Unimplemented();
            return {};
        }
      default:
        Unimplemented();
        return {};
    }
  }

  FpRegister Fma(Decoder::FmaOpcode opcode,
                 Decoder::FloatOperandType float_size,
                 uint8_t rm,
                 FpRegister arg1,
                 FpRegister arg2,
                 FpRegister arg3) {
    switch (float_size) {
      case Decoder::FloatOperandType::kFloat:
        return FloatToFPReg(Fma<Float32>(opcode,
                                         rm,
                                         FPRegToFloat<Float32>(arg1),
                                         FPRegToFloat<Float32>(arg2),
                                         FPRegToFloat<Float32>(arg3)));
      case Decoder::FloatOperandType::kDouble:
        return FloatToFPReg(Fma<Float64>(opcode,
                                         rm,
                                         FPRegToFloat<Float64>(arg1),
                                         FPRegToFloat<Float64>(arg2),
                                         FPRegToFloat<Float64>(arg3)));
      default:
        Unimplemented();
        return {};
    }
  }

  // TODO(b/278812060): switch to intrinsics when they would become available and stop using
  // ExecuteFloatOperation directly.
  template <typename FloatType>
  FloatType Fma(Decoder::FmaOpcode opcode,
                uint8_t rm,
                FloatType arg1,
                FloatType arg2,
                FloatType arg3) {
    switch (opcode) {
      case Decoder::FmaOpcode::kFmadd:
        return intrinsics::ExecuteFloatOperation<FloatType>(
            rm,
            state_->cpu.frm,
            [](auto x, auto y, auto z) { return intrinsics::MulAdd(x, y, z); },
            arg1,
            arg2,
            arg3);
      case Decoder::FmaOpcode::kFmsub:
        return intrinsics::ExecuteFloatOperation<FloatType>(
            rm,
            state_->cpu.frm,
            [](auto x, auto y, auto z) {
              return intrinsics::MulAdd(x, y, intrinsics::Negative(z));
            },
            arg1,
            arg2,
            arg3);
      case Decoder::FmaOpcode::kFnmsub:
        return intrinsics::ExecuteFloatOperation<FloatType>(
            rm,
            state_->cpu.frm,
            [](auto x, auto y, auto z) {
              return intrinsics::MulAdd(intrinsics::Negative(x), y, z);
            },
            arg1,
            arg2,
            arg3);
      case Decoder::FmaOpcode::kFnmadd:
        return intrinsics::ExecuteFloatOperation<FloatType>(
            rm,
            state_->cpu.frm,
            [](auto x, auto y, auto z) {
              return intrinsics::MulAdd(intrinsics::Negative(x), y, intrinsics::Negative(z));
            },
            arg1,
            arg2,
            arg3);
      default:
        Unimplemented();
        return {};
    }
  }

  Register OpImm(Decoder::OpImmOpcode opcode, Register arg, int16_t imm) {
    switch (opcode) {
      case Decoder::OpImmOpcode::kAddi:
        return arg + int64_t{imm};
      case Decoder::OpImmOpcode::kSlti:
        return bit_cast<int64_t>(arg) < int64_t{imm} ? 1 : 0;
      case Decoder::OpImmOpcode::kSltiu:
        return arg < bit_cast<uint64_t>(int64_t{imm}) ? 1 : 0;
      case Decoder::OpImmOpcode::kXori:
        return arg ^ int64_t { imm };
      case Decoder::OpImmOpcode::kOri:
        return arg | int64_t{imm};
      case Decoder::OpImmOpcode::kAndi:
        return arg & int64_t{imm};
      default:
        Unimplemented();
        return {};
    }
  }

  Register Lui(int32_t imm) { return int64_t{imm}; }

  Register Auipc(int32_t imm) {
    uint64_t pc = state_->cpu.insn_addr;
    return pc + int64_t{imm};
  }

  Register OpImm32(Decoder::OpImm32Opcode opcode, Register arg, int16_t imm) {
    switch (opcode) {
      case Decoder::OpImm32Opcode::kAddiw:
        return int32_t(arg) + int32_t{imm};
      default:
        Unimplemented();
        return {};
    }
  }

  Register Ecall(Register syscall_nr, Register arg0, Register arg1, Register arg2, Register arg3,
                 Register arg4, Register arg5) {
    return RunGuestSyscall(syscall_nr, arg0, arg1, arg2, arg3, arg4, arg5);
  }

  FpRegister OpFp(Decoder::OpFpOpcode opcode,
                  Decoder::FloatOperandType float_size,
                  uint8_t rm,
                  FpRegister arg1,
                  FpRegister arg2) {
    switch (float_size) {
      case Decoder::FloatOperandType::kFloat:
        return FloatToFPReg(
            OpFp<Float32>(opcode, rm, FPRegToFloat<Float32>(arg1), FPRegToFloat<Float32>(arg2)));
      case Decoder::FloatOperandType::kDouble:
        return FloatToFPReg(
            OpFp<Float64>(opcode, rm, FPRegToFloat<Float64>(arg1), FPRegToFloat<Float64>(arg2)));
      default:
        Unimplemented();
        return {};
    }
  }

  FpRegister OpFpNoRounding(Decoder::OpFpNoRoundingOpcode opcode,
                            Decoder::FloatOperandType float_size,
                            FpRegister arg1,
                            FpRegister arg2) {
    switch (float_size) {
      case Decoder::FloatOperandType::kFloat:
        return FloatToFPReg(OpFpNoRounding<Float32>(
            opcode, FPRegToFloat<Float32>(arg1), FPRegToFloat<Float32>(arg2)));
      case Decoder::FloatOperandType::kDouble:
        return FloatToFPReg(OpFpNoRounding<Float64>(
            opcode, FPRegToFloat<Float64>(arg1), FPRegToFloat<Float64>(arg2)));
      default:
        Unimplemented();
        return {};
    }
  }

  // In 32-bit case we don't care about the upper 32-bits because nan-boxing will clobber them.
  FpRegister Fmv(Register arg) { return arg; }

  Register Fmv(Decoder::FloatOperandType float_size, FpRegister arg) {
    switch (float_size) {
      case Decoder::FloatOperandType::kFloat:
        return static_cast<int64_t>(static_cast<int32_t>(arg));
      case Decoder::FloatOperandType::kDouble:
        return arg;
      default:
        Unimplemented();
        return {};
    }
  }

  Register OpFpGpRegisterTargetNoRounding(Decoder::OpFpGpRegisterTargetNoRoundingOpcode opcode,
                                          Decoder::FloatOperandType float_size,
                                          FpRegister arg1,
                                          FpRegister arg2) {
    switch (float_size) {
      case Decoder::FloatOperandType::kFloat:
        return OpFpGpRegisterTargetNoRounding<Float32>(
            opcode, FPRegToFloat<Float32>(arg1), FPRegToFloat<Float32>(arg2));
      case Decoder::FloatOperandType::kDouble:
        return OpFpGpRegisterTargetNoRounding<Float64>(
            opcode, FPRegToFloat<Float64>(arg1), FPRegToFloat<Float64>(arg2));
      default:
        Unimplemented();
        return {};
    }
  }

  Register OpFpGpRegisterTargetSingleInputNoRounding(
      Decoder::OpFpGpRegisterTargetSingleInputNoRoundingOpcode opcode,
      Decoder::FloatOperandType float_size,
      FpRegister arg) {
    switch (float_size) {
      case Decoder::FloatOperandType::kFloat:
        return OpFpGpRegisterTargetSingleInputNoRounding<Float32>(opcode,
                                                                  FPRegToFloat<Float32>(arg));
      case Decoder::FloatOperandType::kDouble:
        return OpFpGpRegisterTargetSingleInputNoRounding<Float64>(opcode,
                                                                  FPRegToFloat<Float64>(arg));
      default:
        Unimplemented();
        return {};
    }
  }

  FpRegister OpFpSingleInput(Decoder::OpFpSingleInputOpcode opcode,
                             Decoder::FloatOperandType float_size,
                             uint8_t rm,
                             FpRegister arg) {
    switch (float_size) {
      case Decoder::FloatOperandType::kFloat:
        return FloatToFPReg(OpFpSingleInput<Float32>(opcode, rm, FPRegToFloat<Float32>(arg)));
      case Decoder::FloatOperandType::kDouble:
        return FloatToFPReg(OpFpSingleInput<Float64>(opcode, rm, FPRegToFloat<Float64>(arg)));
      default:
        Unimplemented();
        return {};
    }
  }

  // TODO(b/278812060): switch to intrinsics when they would become available and stop using
  // ExecuteFloatOperation directly.
  template <typename FloatType>
  FloatType OpFp(Decoder::OpFpOpcode opcode, uint8_t rm, FloatType arg1, FloatType arg2) {
    switch (opcode) {
      case Decoder::OpFpOpcode::kFAdd:
        return intrinsics::ExecuteFloatOperation<FloatType>(
            rm, state_->cpu.frm, [](auto x, auto y) { return x + y; }, arg1, arg2);
      case Decoder::OpFpOpcode::kFSub:
        return intrinsics::ExecuteFloatOperation<FloatType>(
            rm, state_->cpu.frm, [](auto x, auto y) { return x - y; }, arg1, arg2);
      case Decoder::OpFpOpcode::kFMul:
        return intrinsics::ExecuteFloatOperation<FloatType>(
            rm, state_->cpu.frm, [](auto x, auto y) { return x * y; }, arg1, arg2);
      case Decoder::OpFpOpcode::kFDiv:
        return intrinsics::ExecuteFloatOperation<FloatType>(
            rm, state_->cpu.frm, [](auto x, auto y) { return x / y; }, arg1, arg2);
      default:
        Unimplemented();
        return {};
    }
  }

  template <typename FloatType>
  FloatType OpFpNoRounding(Decoder::OpFpNoRoundingOpcode opcode, FloatType arg1, FloatType arg2) {
    switch (opcode) {
      case Decoder::OpFpNoRoundingOpcode::kFSgnj:
        return std::get<0>(FSgnj(arg1, arg2));
      case Decoder::OpFpNoRoundingOpcode::kFSgnjn:
        return std::get<0>(FSgnjn(arg1, arg2));
      case Decoder::OpFpNoRoundingOpcode::kFSgnjx:
        return std::get<0>(FSgnjx(arg1, arg2));
      case Decoder::OpFpNoRoundingOpcode::kFMin:
        return Min(arg1, arg2);
      case Decoder::OpFpNoRoundingOpcode::kFMax:
        return Max(arg1, arg2);
      default:
        Unimplemented();
        return {};
    }
  }

  template <typename FloatType>
  Register OpFpGpRegisterTargetNoRounding(Decoder::OpFpGpRegisterTargetNoRoundingOpcode opcode,
                                          FloatType arg1,
                                          FloatType arg2) {
    switch (opcode) {
      case Decoder::OpFpGpRegisterTargetNoRoundingOpcode::kFle:
        return arg1 <= arg2;
      case Decoder::OpFpGpRegisterTargetNoRoundingOpcode::kFlt:
        return arg1 < arg2;
      case Decoder::OpFpGpRegisterTargetNoRoundingOpcode::kFeq:
        return arg1 == arg2;
      default:
        Unimplemented();
        return {};
    }
  }

  template <typename FloatType>
  Register OpFpGpRegisterTargetSingleInputNoRounding(
      Decoder::OpFpGpRegisterTargetSingleInputNoRoundingOpcode opcode,
      FloatType arg) {
    using IntType = std::make_unsigned_t<typename TypeTraits<FloatType>::Int>;
    // TODO(b/284735067): make it constexpr when C++20 would be available.
    IntType quiet_bit = bit_cast<IntType>(std::numeric_limits<FloatType>::quiet_NaN()) &
                        ~bit_cast<IntType>(std::numeric_limits<FloatType>::signaling_NaN());
    IntType raw_bits = bit_cast<IntType>(arg);

    switch (opcode) {
      case Decoder::OpFpGpRegisterTargetSingleInputNoRoundingOpcode::kFclass:
        switch (FPClassify(arg)) {
          case intrinsics::FPInfo::kNaN:
            return (raw_bits & quiet_bit) ? 0b10'0000'0000 : 0b01'0000'0000;
          case intrinsics::FPInfo::kInfinite:
            return intrinsics::SignBit(arg) ? 0b00'0000'0001 : 0b00'1000'0000;
          case intrinsics::FPInfo::kNormal:
            return intrinsics::SignBit(arg) ? 0b00'0000'0010 : 0b00'0100'0000;
          case intrinsics::FPInfo::kSubnormal:
            return intrinsics::SignBit(arg) ? 0b00'0000'0100 : 0b00'0010'0000;
          case intrinsics::FPInfo::kZero:
            return intrinsics::SignBit(arg) ? 0b00'0000'1000 : 0b00'0001'0000;
        }
        [[fallthrough]];
      default:
        Unimplemented();
        return {};
    }
  }

  template <typename FloatType>
  FloatType OpFpSingleInput(Decoder::OpFpSingleInputOpcode opcode, uint8_t rm, FloatType arg) {
    switch (opcode) {
      case Decoder::OpFpSingleInputOpcode::kFSqrt:
        return intrinsics::ExecuteFloatOperation<FloatType>(
            rm, state_->cpu.frm, [](auto x) { return intrinsics::Sqrt(x); }, arg);
      default:
        Unimplemented();
        return {};
    }
  }

  Register ShiftImm(Decoder::ShiftImmOpcode opcode, Register arg, uint16_t imm) {
    switch (opcode) {
      case Decoder::ShiftImmOpcode::kSlli:
        return arg << imm;
      case Decoder::ShiftImmOpcode::kSrli:
        return arg >> imm;
      case Decoder::ShiftImmOpcode::kSrai:
        return bit_cast<int64_t>(arg) >> imm;
      default:
        Unimplemented();
        return {};
    }
  }

  Register ShiftImm32(Decoder::ShiftImm32Opcode opcode, Register arg, uint16_t imm) {
    switch (opcode) {
      case Decoder::ShiftImm32Opcode::kSlliw:
        return int32_t(arg) << int32_t{imm};
      case Decoder::ShiftImm32Opcode::kSrliw:
        return bit_cast<int32_t>(uint32_t(arg) >> uint32_t{imm});
      case Decoder::ShiftImm32Opcode::kSraiw:
        return int32_t(arg) >> int32_t{imm};
      default:
        Unimplemented();
        return {};
    }
  }

  void Store(Decoder::StoreOperandType operand_type, Register arg, int16_t offset, Register data) {
    void* ptr = ToHostAddr<void>(arg + offset);
    switch (operand_type) {
      case Decoder::StoreOperandType::k8bit:
        Store<uint8_t>(ptr, data);
        break;
      case Decoder::StoreOperandType::k16bit:
        Store<uint16_t>(ptr, data);
        break;
      case Decoder::StoreOperandType::k32bit:
        Store<uint32_t>(ptr, data);
        break;
      case Decoder::StoreOperandType::k64bit:
        Store<uint64_t>(ptr, data);
        break;
      default:
        return Unimplemented();
    }
  }

  void StoreFp(Decoder::FloatOperandType opcode, Register arg, int16_t offset, FpRegister data) {
    void* ptr = ToHostAddr<void>(arg + offset);
    switch (opcode) {
      case Decoder::FloatOperandType::kFloat:
        StoreFp<float>(ptr, data);
        break;
      case Decoder::FloatOperandType::kDouble:
        StoreFp<double>(ptr, data);
        break;
      default:
        return Unimplemented();
    }
  }

  void CompareAndBranch(Decoder::BranchOpcode opcode,
                        Register arg1,
                        Register arg2,
                        int16_t offset) {
    bool cond_value;
    switch (opcode) {
      case Decoder::BranchOpcode::kBeq:
        cond_value = arg1 == arg2;
        break;
      case Decoder::BranchOpcode::kBne:
        cond_value = arg1 != arg2;
        break;
      case Decoder::BranchOpcode::kBltu:
        cond_value = arg1 < arg2;
        break;
      case Decoder::BranchOpcode::kBgeu:
        cond_value = arg1 >= arg2;
        break;
      case Decoder::BranchOpcode::kBlt:
        cond_value = bit_cast<int64_t>(arg1) < bit_cast<int64_t>(arg2);
        break;
      case Decoder::BranchOpcode::kBge:
        cond_value = bit_cast<int64_t>(arg1) >= bit_cast<int64_t>(arg2);
        break;
      default:
        return Unimplemented();
    }

    if (cond_value) {
      state_->cpu.insn_addr += offset;
      branch_taken_ = true;
    }
  }

  void Branch(int32_t offset) {
    state_->cpu.insn_addr += offset;
    branch_taken_ = true;
  }

  void BranchRegister(Register base, int16_t offset) {
    state_->cpu.insn_addr = (base + offset) & ~uint64_t{1};
    branch_taken_ = true;
  }

  void Nop() {}

  void Unimplemented() { FATAL("Unimplemented riscv64 instruction"); }

  //
  // Guest state getters/setters.
  //

  Register GetReg(uint8_t reg) const {
    CheckRegIsValid(reg);
    return state_->cpu.x[reg];
  }

  void SetReg(uint8_t reg, Register value) {
    CheckRegIsValid(reg);
    state_->cpu.x[reg] = value;
  }

  FpRegister GetFpReg(uint8_t reg) const {
    CheckFpRegIsValid(reg);
    return state_->cpu.f[reg];
  }

  FpRegister GetFRegAndUnboxNaN(uint8_t reg, Decoder::FloatOperandType operand_type) {
    CheckFpRegIsValid(reg);
    switch (operand_type) {
      case Decoder::FloatOperandType::kFloat: {
        FpRegister value = state_->cpu.f[reg];
        if ((value & 0xffff'ffff'0000'0000) != 0xffff'ffff'0000'0000) {
          return 0x0ffff'ffff'7fc0'0000;
        }
        return value;
      }
      case Decoder::FloatOperandType::kDouble:
        return state_->cpu.f[reg];
      // No support for half-precision and quad-precision operands.
      default:
        Unimplemented();
        return {};
    }
  }

  FpRegister CanonicalizeNans(FpRegister value, Decoder::FloatOperandType operand_type) {
    switch (operand_type) {
      case Decoder::FloatOperandType::kFloat: {
        intrinsics::Float32 result = FPRegToFloat<intrinsics::Float32>(value);
        if (IsNan(result)) {
          return 0x0ffff'ffff'7fc0'0000;
        }
        return value;
      }
      case Decoder::FloatOperandType::kDouble: {
        intrinsics::Float64 result = FPRegToFloat<intrinsics::Float64>(value);
        if (IsNan(result)) {
          return 0x7ff8'0000'0000'0000;
        }
        return value;
      }
      // No support for half-precision and quad-precision operands.
      default:
        Unimplemented();
        return {};
    }
  }

  void NanBoxAndSetFpReg(uint8_t reg, FpRegister value, Decoder::FloatOperandType operand_type) {
    CheckFpRegIsValid(reg);
    switch (operand_type) {
      case Decoder::FloatOperandType::kFloat:
        state_->cpu.f[reg] = value | 0xffff'ffff'0000'0000;
        break;
      case Decoder::FloatOperandType::kDouble:
        state_->cpu.f[reg] = value;
        break;
      // No support for half-precision and quad-precision operands.
      default:
        return Unimplemented();
    }
  }

  //
  // Various helper methods.
  //

  uint64_t GetImm(uint64_t imm) const { return imm; }

  GuestAddr GetInsnAddr() const { return state_->cpu.insn_addr; }

  void FinalizeInsn(uint8_t insn_len) {
    if (!branch_taken_) {
      state_->cpu.insn_addr += insn_len;
    }
  }

 private:
  template <typename DataType>
  Register Load(const void* ptr) const {
    static_assert(std::is_integral_v<DataType>);
    DataType data;
    memcpy(&data, ptr, sizeof(data));
    // Signed types automatically sign-extend to int64_t.
    return static_cast<uint64_t>(data);
  }

  template <typename DataType>
  FpRegister LoadFp(const void* ptr) const {
    static_assert(std::is_floating_point_v<DataType>);
    FpRegister reg = 0;
    memcpy(&reg, ptr, sizeof(DataType));
    return reg;
  }

  template <typename DataType>
  void Store(void* ptr, uint64_t data) const {
    static_assert(std::is_integral_v<DataType>);
    memcpy(ptr, &data, sizeof(DataType));
  }

  template <typename DataType>
  void StoreFp(void* ptr, uint64_t data) const {
    static_assert(std::is_floating_point_v<DataType>);
    memcpy(ptr, &data, sizeof(DataType));
  }

  void CheckRegIsValid(uint8_t reg) const {
    CHECK_GT(reg, 0u);
    CHECK_LE(reg, arraysize(state_->cpu.x));
  }

  void CheckFpRegIsValid(uint8_t reg) const { CHECK_LT(reg, arraysize(state_->cpu.f)); }

  ThreadState* state_;
  bool branch_taken_;
};

}  // namespace

void InterpretInsn(ThreadState* state) {
  GuestAddr pc = state->cpu.insn_addr;

  Interpreter interpreter(state);
  SemanticsPlayer sem_player(&interpreter);
  Decoder decoder(&sem_player);
  uint8_t insn_len = decoder.Decode(ToHostAddr<const uint16_t>(pc));
  interpreter.FinalizeInsn(insn_len);
}

}  // namespace berberis
