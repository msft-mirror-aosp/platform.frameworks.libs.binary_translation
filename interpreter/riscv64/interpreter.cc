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
      case Decoder::OpOpcode::kAndn:
        return arg1 & (~arg2);
      case Decoder::OpOpcode::kOrn:
        return arg1 | (~arg2);
      case Decoder::OpOpcode::kXnor:
        return ~(arg1 ^ arg2);
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

  template <typename DataType>
  FpRegister LoadFp(Register arg, int16_t offset) {
    static_assert(std::is_same_v<DataType, Float32> || std::is_same_v<DataType, Float64>);
    DataType* ptr = ToHostAddr<DataType>(arg + offset);
    FpRegister reg = 0;
    memcpy(&reg, ptr, sizeof(DataType));
    return reg;
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

  Register Slli(Register arg, int8_t imm) { return arg << imm; }

  Register Srli(Register arg, int8_t imm) { return arg >> imm; }

  Register Srai(Register arg, int8_t imm) { return bit_cast<int64_t>(arg) >> imm; }

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

  Register Rori(Register arg, int8_t shamt) {
    CheckShamtIsValid(shamt);
    return (((uint64_t(arg) >> shamt)) | (uint64_t(arg) << (64 - shamt)));
  }

  Register Roriw(Register arg, int8_t shamt) {
    CheckShamt32IsValid(shamt);
    return int32_t(((uint32_t(arg) >> shamt)) | (uint32_t(arg) << (32 - shamt)));
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

  template <typename DataType>
  void StoreFp(Register arg, int16_t offset, FpRegister data) {
    static_assert(std::is_same_v<DataType, Float32> || std::is_same_v<DataType, Float64>);
    DataType* ptr = ToHostAddr<DataType>(arg + offset);
    memcpy(ptr, &data, sizeof(DataType));
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

  FpRegister GetFRegAndUnboxNan(uint8_t reg, Decoder::FloatOperandType operand_type) {
    CheckFpRegIsValid(reg);
    switch (operand_type) {
      case Decoder::FloatOperandType::kFloat: {
        FpRegister value = state_->cpu.f[reg];
        return UnboxNan<Float32>(value);
      }
      case Decoder::FloatOperandType::kDouble:
        return state_->cpu.f[reg];
      // No support for half-precision and quad-precision operands.
      default:
        Unimplemented();
        return {};
    }
  }

  FpRegister CanonicalizeNan(FpRegister value, Decoder::FloatOperandType operand_type) {
    switch (operand_type) {
      case Decoder::FloatOperandType::kFloat: {
        return CanonicalizeNan<Float32>(value);
      }
      case Decoder::FloatOperandType::kDouble: {
        return CanonicalizeNan<Float64>(value);
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
        state_->cpu.f[reg] = NanBox<Float32>(value);
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

  [[nodiscard]] uint8_t GetFrm() const { return state_->cpu.frm; }

  [[nodiscard]] uint64_t GetImm(uint64_t imm) const { return imm; }

  [[nodiscard]] GuestAddr GetInsnAddr() const { return state_->cpu.insn_addr; }

  void FinalizeInsn(uint8_t insn_len) {
    if (!branch_taken_) {
      state_->cpu.insn_addr += insn_len;
    }
  }

#include "berberis/intrinsics/interpreter_intrinsics_hooks-inl.h"

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
  void Store(void* ptr, uint64_t data) const {
    static_assert(std::is_integral_v<DataType>);
    memcpy(ptr, &data, sizeof(DataType));
  }

  template <typename DataType>
  void StoreFp(void* ptr, uint64_t data) const {
    static_assert(std::is_floating_point_v<DataType>);
    memcpy(ptr, &data, sizeof(DataType));
  }

  void CheckShamtIsValid(int8_t shamt) const {
    CHECK_GE(shamt, 0);
    CHECK_LT(shamt, 64);
  }

  void CheckShamt32IsValid(int8_t shamt) const {
    CHECK_GE(shamt, 0);
    CHECK_LT(shamt, 32);
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
