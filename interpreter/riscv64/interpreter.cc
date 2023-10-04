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

#include <atomic>
#include <cfenv>
#include <cstdint>
#include <cstring>

#include "berberis/base/bit_util.h"
#include "berberis/base/checks.h"
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
#include "berberis/runtime_primitives/memory_region_reservation.h"
#include "berberis/runtime_primitives/recovery_code.h"

#include "fp_regs.h"

namespace berberis {

namespace {

inline constexpr std::memory_order AqRlToStdMemoryOrder(bool aq, bool rl) {
  if (aq) {
    if (rl) {
      return std::memory_order_acq_rel;
    } else {
      return std::memory_order_acquire;
    }
  } else {
    if (rl) {
      return std::memory_order_release;
    } else {
      return std::memory_order_relaxed;
    }
  }
}

class Interpreter {
 public:
  using CsrName = berberis::CsrName;
  using Decoder = Decoder<SemanticsPlayer<Interpreter>>;
  using Register = uint64_t;
  using FpRegister = uint64_t;
  using Float32 = intrinsics::Float32;
  using Float64 = intrinsics::Float64;

  explicit Interpreter(ThreadState* state) : state_(state), branch_taken_(false) {}

  //
  // Instruction implementations.
  //

  Register UpdateCsr(Decoder::CsrOpcode opcode, Register arg, Register csr) {
    switch (opcode) {
      case Decoder::CsrOpcode::kCsrrs:
        return arg | csr;
      case Decoder::CsrOpcode::kCsrrc:
        return ~arg & csr;
      default:
        Unimplemented();
        return {};
    }
  }

  Register UpdateCsr(Decoder::CsrImmOpcode opcode, uint8_t imm, Register csr) {
    return UpdateCsr(static_cast<Decoder::CsrOpcode>(opcode), imm, csr);
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

  template <typename IntType, bool aq, bool rl>
  Register Lr(int64_t addr) {
    static_assert(std::is_integral_v<IntType>, "Lr: IntType must be integral");
    static_assert(std::is_signed_v<IntType>, "Lr: IntType must be signed");
    // Address must be aligned on size of IntType.
    CHECK((addr % sizeof(IntType)) == 0ULL);
    return MemoryRegionReservation::Load<IntType>(&state_->cpu, addr, AqRlToStdMemoryOrder(aq, rl));
  }

  template <typename IntType, bool aq, bool rl>
  Register Sc(int64_t addr, IntType val) {
    static_assert(std::is_integral_v<IntType>, "Sc: IntType must be integral");
    static_assert(std::is_signed_v<IntType>, "Sc: IntType must be signed");
    // Address must be aligned on size of IntType.
    CHECK((addr % sizeof(IntType)) == 0ULL);
    return static_cast<Register>(MemoryRegionReservation::Store<IntType>(
        &state_->cpu, addr, val, AqRlToStdMemoryOrder(aq, rl)));
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

  FpRegister Fmv(FpRegister arg) { return arg; }

  //
  // V extensions.
  //

  using TailProcessing = intrinsics::TailProcessing;
  using InactiveProcessing = intrinsics::InactiveProcessing;

  enum class VectorSelecteElementWidth {
    k8bit = 0b000,
    k16bit = 0b001,
    k32bit = 0b010,
    k64bit = 0b011,
    kMaxValue = 0b111,
  };

  enum class VectorRegisterGroupMultiplier {
    k1register = 0b000,
    k2registers = 0b001,
    k4registers = 0b010,
    k8registers = 0b011,
    kEigthOfRegister = 0b101,
    kQuarterOfRegister = 0b110,
    kHalfOfRegister = 0b111,
    kMaxValue = 0b111,
  };

  static constexpr size_t NumberOfRegistersInvolved(VectorRegisterGroupMultiplier vlmul) {
    switch (vlmul) {
      case VectorRegisterGroupMultiplier::k2registers:
        return 2;
      case VectorRegisterGroupMultiplier::k4registers:
        return 4;
      case VectorRegisterGroupMultiplier::k8registers:
        return 8;
      default:
        return 1;
    }
  }

  template <typename VOpArgs, typename... ExtraArgs>
  void OpVector(const VOpArgs& args, ExtraArgs... extra_args) {
    // RISC-V V extensions are using 8bit “opcode extension” vtype Csr to make sure 32bit encoding
    // would be usable.
    //
    // Great care is made to ensure that vector code wouldn't need to change vtype Csr often (e.g.
    // there are special mask instructions which allow one to manipulate on masks without the need
    // to change the CPU mode.
    //
    // Currently we don't have support for multiple CPU mode in Berberis thus we can only handle
    // these instrtuctions in the interpreter.
    //
    // TODO(300690740): develop and implement strategy which would allow us to support vector
    // intrinsics not just in the interpreter. Move code from this function to semantics player.
    Register vtype = GetCsr<CsrName::kVtype>();
    if (static_cast<std::make_signed_t<Register>>(vtype) < 0) {
      return Unimplemented();
    }
    switch (static_cast<VectorSelecteElementWidth>((vtype >> 3) & 0b111)) {
      case VectorSelecteElementWidth::k8bit:
        return OpVector<uint8_t>(args, vtype, extra_args...);
      case VectorSelecteElementWidth::k16bit:
        return OpVector<uint16_t>(args, vtype, extra_args...);
      case VectorSelecteElementWidth::k32bit:
        return OpVector<uint32_t>(args, vtype, extra_args...);
      case VectorSelecteElementWidth::k64bit:
        return OpVector<uint64_t>(args, vtype, extra_args...);
      default:
        return Unimplemented();
    }
  }

  template <typename ElementType, typename VOpArgs, typename... ExtraArgs>
  void OpVector(const VOpArgs& args, Register vtype, ExtraArgs... extra_args) {
    switch (static_cast<VectorRegisterGroupMultiplier>(vtype & 0b111)) {
      case VectorRegisterGroupMultiplier::k1register:
        return OpVector<ElementType, VectorRegisterGroupMultiplier::k1register>(
            args, vtype, extra_args...);
      case VectorRegisterGroupMultiplier::k2registers:
        return OpVector<ElementType, VectorRegisterGroupMultiplier::k2registers>(
            args, vtype, extra_args...);
      case VectorRegisterGroupMultiplier::k4registers:
        return OpVector<ElementType, VectorRegisterGroupMultiplier::k4registers>(
            args, vtype, extra_args...);
      case VectorRegisterGroupMultiplier::k8registers:
        return OpVector<ElementType, VectorRegisterGroupMultiplier::k8registers>(
            args, vtype, extra_args...);
      case VectorRegisterGroupMultiplier::kEigthOfRegister:
        return OpVector<ElementType, VectorRegisterGroupMultiplier::kEigthOfRegister>(
            args, vtype, extra_args...);
      case VectorRegisterGroupMultiplier::kQuarterOfRegister:
        return OpVector<ElementType, VectorRegisterGroupMultiplier::kQuarterOfRegister>(
            args, vtype, extra_args...);
      case VectorRegisterGroupMultiplier::kHalfOfRegister:
        return OpVector<ElementType, VectorRegisterGroupMultiplier::kHalfOfRegister>(
            args, vtype, extra_args...);
      default:
        return Unimplemented();
    }
  }

  template <typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            typename VOpArgs,
            typename... ExtraArgs>
  void OpVector(const VOpArgs& args, Register vtype, ExtraArgs... extra_args) {
    if ((vtype >> 6) & 1) {
      return OpVector<ElementType, vlmul, TailProcessing::kAgnostic>(args, vtype, extra_args...);
    }
    return OpVector<ElementType, vlmul, TailProcessing::kUndisturbed>(args, vtype, extra_args...);
  }

  template <typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            typename VOpArgs,
            typename... ExtraArgs>
  void OpVector(const VOpArgs& args, Register vtype, ExtraArgs... extra_args) {
    if (args.vm) {
      return OpVector<ElementType, vlmul, vta>(args, extra_args...);
    }
    if (vtype >> 7) {
      return OpVector<ElementType, vlmul, vta, InactiveProcessing::kAgnostic>(args, extra_args...);
    }
    return OpVector<ElementType, vlmul, vta, InactiveProcessing::kUndisturbed>(args, extra_args...);
  }

  template <typename ElementType, VectorRegisterGroupMultiplier vlmul, TailProcessing vta>
  void OpVector(const Decoder::VOpViArgs& args) {
    switch (args.opcode) {
      case Decoder::VOpViOpcode::kVaddvi:
        return OpVectorvx<intrinsics::Vaddvx<ElementType, vta>, ElementType, vlmul, vta>(
            args.dst, args.src, args.imm);
      case Decoder::VOpViOpcode::kVrsubvi:
        return OpVectorvx<intrinsics::Vrsubvx<ElementType, vta>, ElementType, vlmul, vta>(
            args.dst, args.src, args.imm);
      default:
        Unimplemented();
    }
  }

  template <typename ElementType, VectorRegisterGroupMultiplier vlmul, TailProcessing vta>
  void OpVector(const Decoder::VOpVvArgs& args) {
    switch (args.opcode) {
      case Decoder::VOpVvOpcode::kVaddvv:
        return OpVectorvv<intrinsics::Vaddvv<ElementType, vta>, ElementType, vlmul, vta>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpVvOpcode::kVsubvv:
        return OpVectorvv<intrinsics::Vsubvv<ElementType, vta>, ElementType, vlmul, vta>(
            args.dst, args.src1, args.src2);
      default:
        Unimplemented();
    }
  }

  template <typename ElementType, VectorRegisterGroupMultiplier vlmul, TailProcessing vta>
  void OpVector(const Decoder::VOpVxArgs& args, Register arg2) {
    switch (args.opcode) {
      case Decoder::VOpVxOpcode::kVaddvx:
        return OpVectorvx<intrinsics::Vaddvx<ElementType, vta>, ElementType, vlmul, vta>(
            args.dst, args.src1, arg2);
      case Decoder::VOpVxOpcode::kVsubvx:
        return OpVectorvx<intrinsics::Vsubvx<ElementType, vta>, ElementType, vlmul, vta>(
            args.dst, args.src1, arg2);
      case Decoder::VOpVxOpcode::kVrsubvx:
        return OpVectorvx<intrinsics::Vrsubvx<ElementType, vta>, ElementType, vlmul, vta>(
            args.dst, args.src1, arg2);
      default:
        Unimplemented();
    }
  }

  template <auto Intrinsic,
            typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta>
  void OpVectorvv(uint8_t dst, uint8_t src1, uint8_t src2) {
    constexpr size_t registers_involved = NumberOfRegistersInvolved(vlmul);
    if ((dst & (registers_involved - 1)) != 0 || (src1 & (registers_involved - 1)) != 0 ||
        (src2 & (registers_involved - 1)) != 0) {
      return Unimplemented();
    }
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    SIMD128Register result, arg1, arg2;
    for (size_t index = 0; index < registers_involved; ++index) {
      result.Set(state_->cpu.v[dst + index]);
      arg1.Set(state_->cpu.v[src1 + index]);
      arg2.Set(state_->cpu.v[src2 + index]);
      std::tie(result) = Intrinsic(vstart - index * (16 / sizeof(ElementType)),
                                   vl - index * (16 / sizeof(ElementType)),
                                   result,
                                   arg1,
                                   arg2);
      state_->cpu.v[dst + index] = result.Get<__uint128_t>();
    }
    SetCsr<CsrName::kVstart>(0);
  }

  template <auto Intrinsic,
            typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta>
  void OpVectorvx(uint8_t dst, uint8_t src1, ElementType arg2) {
    constexpr size_t registers_involved = NumberOfRegistersInvolved(vlmul);
    if ((dst & (registers_involved - 1)) != 0 || (src1 & (registers_involved - 1)) != 0) {
      return Unimplemented();
    }
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    SIMD128Register result, arg1;
    for (size_t index = 0; index < registers_involved; ++index) {
      result.Set(state_->cpu.v[dst + index]);
      arg1.Set(state_->cpu.v[src1 + index]);
      std::tie(result) = Intrinsic(vstart - index * (16 / sizeof(ElementType)),
                                   vl - index * (16 / sizeof(ElementType)),
                                   result,
                                   arg1,
                                   arg2);
      state_->cpu.v[dst + index] = result.Get<__uint128_t>();
    }
    SetCsr<CsrName::kVstart>(0);
  }

  template <typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            InactiveProcessing vma>
  void OpVector(const Decoder::VOpViArgs& args) {
    switch (args.opcode) {
      case Decoder::VOpViOpcode::kVaddvi:
        return OpVectorvx<intrinsics::Vaddvxm<ElementType, vta, vma>, ElementType, vlmul, vta, vma>(
            args.dst, args.src, args.imm);
      case Decoder::VOpViOpcode::kVrsubvi:
        return OpVectorvx<intrinsics::Vrsubvxm<ElementType, vta, vma>, ElementType, vlmul, vta, vma>(
            args.dst, args.src, args.imm);
      default:
        Unimplemented();
    }
  }

  template <typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            InactiveProcessing vma>
  void OpVector(const Decoder::VOpVvArgs& args) {
    switch (args.opcode) {
      case Decoder::VOpVvOpcode::kVaddvv:
        return OpVectorvv<intrinsics::Vaddvvm<ElementType, vta, vma>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpVvOpcode::kVsubvv:
        return OpVectorvv<intrinsics::Vsubvvm<ElementType, vta, vma>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      default:
        Unimplemented();
    }
  }

  template <typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            InactiveProcessing vma>
  void OpVector(const Decoder::VOpVxArgs& args, Register arg2) {
    switch (args.opcode) {
      case Decoder::VOpVxOpcode::kVaddvx:
        return OpVectorvx<intrinsics::Vaddvxm<ElementType, vta, vma>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, arg2);
      case Decoder::VOpVxOpcode::kVsubvx:
        return OpVectorvx<intrinsics::Vsubvxm<ElementType, vta, vma>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, arg2);
      case Decoder::VOpVxOpcode::kVrsubvx:
        return OpVectorvx<intrinsics::Vrsubvxm<ElementType, vta, vma>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, arg2);
      default:
        Unimplemented();
    }
  }

  template <auto Intrinsic,
            typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            InactiveProcessing vma>
  void OpVectorvv(uint8_t dst, uint8_t src1, uint8_t src2) {
    constexpr size_t registers_involved = NumberOfRegistersInvolved(vlmul);
    if ((dst & (registers_involved - 1)) != 0 || (src1 & (registers_involved - 1)) != 0 ||
        (src2 & (registers_involved - 1)) != 0) {
      return Unimplemented();
    }
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    SIMD128Register mask, result, arg1, arg2;
    mask.Set(state_->cpu.v[0]);
    for (size_t index = 0; index < registers_involved; ++index) {
      result.Set(state_->cpu.v[dst + index]);
      arg1.Set(state_->cpu.v[src1 + index]);
      arg2.Set(state_->cpu.v[src2 + index]);
      std::tie(result) = Intrinsic(vstart - index * (16 / sizeof(ElementType)),
                                   vl - index * (16 / sizeof(ElementType)),
                                   intrinsics::MaskForRegisterInSequence<ElementType>(mask, index),
                                   result,
                                   arg1,
                                   arg2);
      state_->cpu.v[dst + index] = result.Get<__uint128_t>();
    }
    SetCsr<CsrName::kVstart>(0);
  }

  template <auto Intrinsic,
            typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            InactiveProcessing vma>
  void OpVectorvx(uint8_t dst, uint8_t src1, ElementType arg2) {
    constexpr size_t registers_involved = NumberOfRegistersInvolved(vlmul);
    if ((dst & (registers_involved - 1)) != 0 || (src1 & (registers_involved - 1)) != 0) {
      return Unimplemented();
    }
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    SIMD128Register mask, result, arg1;
    mask.Set(state_->cpu.v[0]);
    for (size_t index = 0; index < registers_involved; ++index) {
      result.Set(state_->cpu.v[dst + index]);
      arg1.Set(state_->cpu.v[src1 + index]);
      std::tie(result) = Intrinsic(vstart - index * (16 / sizeof(ElementType)),
                                   vl - index * (16 / sizeof(ElementType)),
                                   intrinsics::MaskForRegisterInSequence<ElementType>(mask, index),
                                   result,
                                   arg1,
                                   arg2);
      state_->cpu.v[dst + index] = result.Get<__uint128_t>();
    }
    SetCsr<CsrName::kVstart>(0);
  }

  void Nop() {}

  void Unimplemented() {
    auto* addr = ToHostAddr<const uint16_t>(GetInsnAddr());
    uint8_t size = Decoder::GetInsnSize(addr);
    if (size == 2) {
      FATAL("Unimplemented riscv64 instruction 0x%" PRIx16 " at %p", *addr, addr);
    } else {
      CHECK_EQ(size, 4);
      // Warning: do not cast and dereference the pointer
      // since the address may not be 4-bytes aligned.
      uint32_t code;
      memcpy(&code, addr, sizeof(code));
      FATAL("Unimplemented riscv64 instruction 0x%" PRIx32 " at %p", code, addr);
    }
  }

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

  template <typename FloatType>
  FpRegister GetFRegAndUnboxNan(uint8_t reg);

  template <typename FloatType>
  void NanBoxAndSetFpReg(uint8_t reg, FpRegister value);

  //
  // Various helper methods.
  //

  template <CsrName kName>
  [[nodiscard]] Register GetCsr() const {
    return state_->cpu.*CsrFieldAddr<kName>;
  }

  template <CsrName kName>
  void SetCsr(Register arg) {
    state_->cpu.*CsrFieldAddr<kName> = arg & kCsrMask<kName>;
  }

  [[nodiscard]] uint64_t GetImm(uint64_t imm) const { return imm; }

  [[nodiscard]] Register Copy(Register value) const { return value; }

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

template <>
[[nodiscard]] Interpreter::Register Interpreter::GetCsr<CsrName::kFCsr>() const {
  return FeGetExceptions() | (state_->cpu.frm << 5);
}

template <>
[[nodiscard]] Interpreter::Register Interpreter::GetCsr<CsrName::kFFlags>() const {
  return FeGetExceptions();
}

template <>
[[nodiscard]] Interpreter::Register Interpreter::GetCsr<CsrName::kVlenb>() const {
  return 16;
}

template <>
[[nodiscard]] Interpreter::Register Interpreter::GetCsr<CsrName::kVxrm>() const {
  return state_->cpu.*CsrFieldAddr<CsrName::kVcsr> & 0b11;
}

template <>
[[nodiscard]] Interpreter::Register Interpreter::GetCsr<CsrName::kVxsat>() const {
  return state_->cpu.*CsrFieldAddr<CsrName::kVcsr> >> 2;
}

template <>
void Interpreter::SetCsr<CsrName::kFCsr>(Register arg) {
  FeSetExceptions(arg & 0b1'1111);
  arg = (arg >> 5) & kCsrMask<CsrName::kFrm>;
  state_->cpu.frm = arg;
  FeSetRound(arg);
}

template <>
void Interpreter::SetCsr<CsrName::kFFlags>(Register arg) {
  FeSetExceptions(arg & 0b1'1111);
}

template <>
void Interpreter::SetCsr<CsrName::kFrm>(Register arg) {
  arg &= kCsrMask<CsrName::kFrm>;
  state_->cpu.frm = arg;
  FeSetRound(arg);
}

template <>
void Interpreter::SetCsr<CsrName::kVxrm>(Register arg) {
  state_->cpu.*CsrFieldAddr<CsrName::kVcsr> =
      (state_->cpu.*CsrFieldAddr<CsrName::kVcsr> & 0b100) | (arg & 0b11);
}

template <>
void Interpreter::SetCsr<CsrName::kVxsat>(Register arg) {
  state_->cpu.*CsrFieldAddr<CsrName::kVcsr> =
      (state_->cpu.*CsrFieldAddr<CsrName::kVcsr> & 0b11) | ((arg & 0b1) << 2);
}

template <>
Interpreter::FpRegister Interpreter::GetFRegAndUnboxNan<Interpreter::Float32>(uint8_t reg) {
  CheckFpRegIsValid(reg);
  FpRegister value = state_->cpu.f[reg];
  return UnboxNan<Float32>(value);
}

template <>
Interpreter::FpRegister Interpreter::GetFRegAndUnboxNan<Interpreter::Float64>(uint8_t reg) {
  CheckFpRegIsValid(reg);
  return state_->cpu.f[reg];
}

template <>
void Interpreter::NanBoxAndSetFpReg<Interpreter::Float32>(uint8_t reg, FpRegister value) {
  CheckFpRegIsValid(reg);
  state_->cpu.f[reg] = NanBox<Float32>(value);
}

template <>
void Interpreter::NanBoxAndSetFpReg<Interpreter::Float64>(uint8_t reg, FpRegister value) {
  CheckFpRegIsValid(reg);
  state_->cpu.f[reg] = value;
}

}  // namespace

void InitInterpreter() {
  // TODO(b/232598137): Currently we just call it to initialize the recovery map.
  // We need to add real faulty instructions with recovery here.
  InitExtraRecoveryCodeUnsafe({});
}

void InterpretInsn(ThreadState* state) {
  GuestAddr pc = state->cpu.insn_addr;

  Interpreter interpreter(state);
  SemanticsPlayer sem_player(&interpreter);
  Decoder decoder(&sem_player);
  uint8_t insn_len = decoder.Decode(ToHostAddr<const uint16_t>(pc));
  interpreter.FinalizeInsn(insn_len);
}

}  // namespace berberis
