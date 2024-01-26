/*
 * Copyright (C) 2023 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file excenaupt in compliance with the License.
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
#include "berberis/intrinsics/simd_register.h"
#include "berberis/intrinsics/type_traits.h"
#include "berberis/kernel_api/run_guest_syscall.h"
#include "berberis/runtime_primitives/interpret_helpers.h"
#include "berberis/runtime_primitives/memory_region_reservation.h"
#include "berberis/runtime_primitives/recovery_code.h"

#include "faulty_memory_accesses.h"
#include "regs.h"

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

  explicit Interpreter(ThreadState* state)
      : state_(state), branch_taken_(false), exception_raised_(false) {}

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
    CHECK(!exception_raised_);
    // Address must be aligned on size of IntType.
    CHECK((addr % sizeof(IntType)) == 0ULL);
    return MemoryRegionReservation::Load<IntType>(&state_->cpu, addr, AqRlToStdMemoryOrder(aq, rl));
  }

  template <typename IntType, bool aq, bool rl>
  Register Sc(int64_t addr, IntType val) {
    static_assert(std::is_integral_v<IntType>, "Sc: IntType must be integral");
    static_assert(std::is_signed_v<IntType>, "Sc: IntType must be signed");
    CHECK(!exception_raised_);
    // Address must be aligned on size of IntType.
    CHECK((addr % sizeof(IntType)) == 0ULL);
    return static_cast<Register>(MemoryRegionReservation::Store<IntType>(
        &state_->cpu, addr, val, AqRlToStdMemoryOrder(aq, rl)));
  }

  Register Op(Decoder::OpOpcode opcode, Register arg1, Register arg2) {
    switch (opcode) {
      case Decoder::OpOpcode::kAdd:
        return Int64(arg1) + Int64(arg2);
      case Decoder::OpOpcode::kSub:
        return Int64(arg1) - Int64(arg2);
      case Decoder::OpOpcode::kAnd:
        return Int64(arg1) & Int64(arg2);
      case Decoder::OpOpcode::kOr:
        return Int64(arg1) | Int64(arg2);
      case Decoder::OpOpcode::kXor:
        return Int64(arg1) ^ Int64(arg2);
      case Decoder::OpOpcode::kSll:
        return Int64(arg1) << Int64(arg2);
      case Decoder::OpOpcode::kSrl:
        return UInt64(arg1) >> Int64(arg2);
      case Decoder::OpOpcode::kSra:
        return Int64(arg1) >> Int64(arg2);
      case Decoder::OpOpcode::kSlt:
        return Int64(arg1) < Int64(arg2) ? 1 : 0;
      case Decoder::OpOpcode::kSltu:
        return UInt64(arg1) < UInt64(arg2) ? 1 : 0;
      case Decoder::OpOpcode::kMul:
        return Int64(arg1) * Int64(arg2);
      case Decoder::OpOpcode::kMulh:
        return NarrowTopHalf(Widen(Int64(arg1)) * Widen(Int64(arg2)));
      case Decoder::OpOpcode::kMulhsu:
        return NarrowTopHalf(Widen(Int64(arg1)) * BitCastToSigned(Widen(UInt64(arg2))));
      case Decoder::OpOpcode::kMulhu:
        return NarrowTopHalf(Widen(UInt64(arg1)) * Widen(UInt64(arg2)));
      case Decoder::OpOpcode::kDiv:
        return Int64(arg1) / Int64(arg2);
      case Decoder::OpOpcode::kDivu:
        return UInt64(arg1) / UInt64(arg2);
      case Decoder::OpOpcode::kRem:
        return Int64(arg1) % Int64(arg2);
      case Decoder::OpOpcode::kRemu:
        return UInt64(arg1) % UInt64(arg2);
      case Decoder::OpOpcode::kAndn:
        return Int64(arg1) & (~Int64(arg2));
      case Decoder::OpOpcode::kOrn:
        return Int64(arg1) | (~Int64(arg2));
      case Decoder::OpOpcode::kXnor:
        return ~(Int64(arg1) ^ Int64(arg2));
      default:
        Unimplemented();
        return {};
    }
  }

  Register Op32(Decoder::Op32Opcode opcode, Register arg1, Register arg2) {
    switch (opcode) {
      case Decoder::Op32Opcode::kAddw:
        return Widen(TruncateTo<Int32>(arg1) + TruncateTo<Int32>(arg2));
      case Decoder::Op32Opcode::kSubw:
        return Widen(TruncateTo<Int32>(arg1) - TruncateTo<Int32>(arg2));
      case Decoder::Op32Opcode::kSllw:
        return Widen(TruncateTo<Int32>(arg1) << TruncateTo<Int32>(arg2));
      case Decoder::Op32Opcode::kSrlw:
        return Widen(BitCastToSigned(TruncateTo<UInt32>(arg1) >> TruncateTo<Int32>(arg2)));
      case Decoder::Op32Opcode::kSraw:
        return Widen(TruncateTo<Int32>(arg1) >> TruncateTo<Int32>(arg2));
      case Decoder::Op32Opcode::kMulw:
        return Widen(TruncateTo<Int32>(arg1) * TruncateTo<Int32>(arg2));
      case Decoder::Op32Opcode::kDivw:
        return Widen(TruncateTo<Int32>(arg1) / TruncateTo<Int32>(arg2));
      case Decoder::Op32Opcode::kDivuw:
        return Widen(BitCastToSigned(TruncateTo<UInt32>(arg1) / TruncateTo<UInt32>(arg2)));
      case Decoder::Op32Opcode::kRemw:
        return Widen(TruncateTo<Int32>(arg1) % TruncateTo<Int32>(arg2));
      case Decoder::Op32Opcode::kRemuw:
        return Widen(BitCastToSigned(TruncateTo<UInt32>(arg1) % TruncateTo<UInt32>(arg2)));
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
    CHECK(!exception_raised_);
    DataType* ptr = ToHostAddr<DataType>(arg + offset);
    FaultyLoadResult result = FaultyLoad(ptr, sizeof(DataType));
    if (result.is_fault) {
      exception_raised_ = true;
      return {};
    }
    return result.value;
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

  Register Ecall(Register syscall_nr,
                 Register arg0,
                 Register arg1,
                 Register arg2,
                 Register arg3,
                 Register arg4,
                 Register arg5) {
    CHECK(!exception_raised_);
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
    CHECK(!exception_raised_);
    DataType* ptr = ToHostAddr<DataType>(arg + offset);
    exception_raised_ = FaultyStore(ptr, sizeof(DataType), data);
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
      Branch(offset);
    }
  }

  void Branch(int32_t offset) {
    CHECK(!exception_raised_);
    state_->cpu.insn_addr += offset;
    branch_taken_ = true;
  }

  void BranchRegister(Register base, int16_t offset) {
    CHECK(!exception_raised_);
    state_->cpu.insn_addr = (base + offset) & ~uint64_t{1};
    branch_taken_ = true;
  }

  FpRegister Fmv(FpRegister arg) { return arg; }

  //
  // V extensions.
  //

  using TailProcessing = intrinsics::TailProcessing;
  using InactiveProcessing = intrinsics::InactiveProcessing;

  enum class VectorSelectElementWidth {
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

  static constexpr size_t NumDestRegistersInvolvedForW(VectorRegisterGroupMultiplier vlmul) {
    switch (vlmul) {
      case VectorRegisterGroupMultiplier::k1register:
        return 2;
      case VectorRegisterGroupMultiplier::k2registers:
        return 4;
      case VectorRegisterGroupMultiplier::k4registers:
        return 8;
      default:
        return 1;
    }
  }

  template <typename VOpArgs, typename... ExtraArgs>
  void OpVector(const VOpArgs& args, ExtraArgs... extra_args) {
    // Note: whole register instructions are not dependent on vtype and are supposed to work even
    // if vill is set!  Handle them before processing other instructions.
    // Note: other tupes of loads and store are not special and would be processed as usual.
    if constexpr (std::is_same_v<VOpArgs, Decoder::VLoadUnitStrideArgs>) {
      if (args.opcode == Decoder::VLoadUnitStrideOpcode::kVlXreXX) {
        if (!IsPowerOf2(args.nf + 1)) {
          return Unimplemented();
        }
        if ((args.dst & args.nf) != 0) {
          return Unimplemented();
        }
        auto [src] = std::tuple{extra_args...};
        __uint128_t* ptr = bit_cast<__uint128_t*>(src);
        for (size_t index = 0; index <= args.nf; index++) {
          state_->cpu.v[args.dst + index] = ptr[index];
        }
        return;
      }
    }

    if constexpr (std::is_same_v<VOpArgs, Decoder::VStoreUnitStrideArgs>) {
      if (args.opcode == Decoder::VStoreUnitStrideOpcode::kVsX) {
        if (args.width != Decoder::StoreOperandType::k8bit) {
          return Unimplemented();
        }
        if (!IsPowerOf2(args.nf + 1)) {
          return Unimplemented();
        }
        if ((args.data & args.nf) != 0) {
          return Unimplemented();
        }
        auto [src] = std::tuple{extra_args...};
        __uint128_t* ptr = bit_cast<__uint128_t*>(src);
        for (size_t index = 0; index <= args.nf; index++) {
          ptr[index] = state_->cpu.v[args.data + index];
        }
        return;
      }
    }
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
    // TODO(b/300690740): develop and implement strategy which would allow us to support vector
    // intrinsics not just in the interpreter. Move code from this function to semantics player.
    Register vtype = GetCsr<CsrName::kVtype>();
    if (static_cast<std::make_signed_t<Register>>(vtype) < 0) {
      return Unimplemented();
    }
    switch (static_cast<VectorSelectElementWidth>((vtype >> 3) & 0b111)) {
      case VectorSelectElementWidth::k8bit:
        return OpVector<UInt8>(args, vtype, extra_args...);
      case VectorSelectElementWidth::k16bit:
        return OpVector<UInt16>(args, vtype, extra_args...);
      case VectorSelectElementWidth::k32bit:
        return OpVector<UInt32>(args, vtype, extra_args...);
      case VectorSelectElementWidth::k64bit:
        return OpVector<UInt64>(args, vtype, extra_args...);
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
      return OpVector<ElementType, vlmul, vta, intrinsics::NoInactiveProcessing{}>(args,
                                                                                   extra_args...);
    }
    if (vtype >> 7) {
      return OpVector<ElementType, vlmul, vta, InactiveProcessing::kAgnostic>(args, extra_args...);
    }
    return OpVector<ElementType, vlmul, vta, InactiveProcessing::kUndisturbed>(args, extra_args...);
  }

  template <typename ElementType, VectorRegisterGroupMultiplier vlmul, TailProcessing vta, auto vma>
  void OpVector(const Decoder::VLoadUnitStrideArgs& /*args*/, Register /*src*/) {
    Unimplemented();
  }

  template <typename ElementType, VectorRegisterGroupMultiplier vlmul, TailProcessing vta, auto vma>
  void OpVector(const Decoder::VOpIViArgs& args) {
    using SignedType = berberis::SignedType<ElementType>;
    using UnsignedType = berberis::UnsignedType<ElementType>;
    switch (args.opcode) {
      case Decoder::VOpIViOpcode::kVaddvi:
        return OpVectorvx<intrinsics::Vaddvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
      case Decoder::VOpIViOpcode::kVrsubvi:
        return OpVectorvx<intrinsics::Vrsubvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
      case Decoder::VOpIViOpcode::kVandvi:
        return OpVectorvx<intrinsics::Vandvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
      case Decoder::VOpIViOpcode::kVorvi:
        return OpVectorvx<intrinsics::Vorvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
      case Decoder::VOpIViOpcode::kVxorvi:
        return OpVectorvx<intrinsics::Vxorvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
      case Decoder::VOpIViOpcode::kVmseqvi:
        return OpVectormvx<intrinsics::Vseqvx<ElementType>, ElementType, vlmul, vma>(
            args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
      case Decoder::VOpIViOpcode::kVmsnevi:
        return OpVectormvx<intrinsics::Vsnevx<ElementType>, ElementType, vlmul, vma>(
            args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
      case Decoder::VOpIViOpcode::kVmsleuvi:
        return OpVectormvx<intrinsics::Vslevx<UnsignedType>, UnsignedType, vlmul, vma>(
            args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
      case Decoder::VOpIViOpcode::kVmslevi:
        return OpVectormvx<intrinsics::Vslevx<SignedType>, SignedType, vlmul, vma>(
            args.dst, args.src, SignedType{args.imm});
      case Decoder::VOpIViOpcode::kVmsgtuvi:
        return OpVectormvx<intrinsics::Vsgtvx<UnsignedType>, UnsignedType, vlmul, vma>(
            args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
      case Decoder::VOpIViOpcode::kVmsgtvi:
        return OpVectormvx<intrinsics::Vsgtvx<SignedType>, SignedType, vlmul, vma>(
            args.dst, args.src, SignedType{args.imm});
      case Decoder::VOpIViOpcode::kVsllvi:
        return OpVectorvx<intrinsics::Vslvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
      case Decoder::VOpIViOpcode::kVsrlvi:
        return OpVectorvx<intrinsics::Vsrvx<UnsignedType>, UnsignedType, vlmul, vta, vma>(
            args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
      case Decoder::VOpIViOpcode::kVsravi:
        return OpVectorvx<intrinsics::Vsrvx<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src, SignedType{args.imm});
      case Decoder::VOpIViOpcode::kVmergevi:
        if constexpr (std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>) {
          return OpVectorvx<intrinsics::Vmergevx<ElementType>, ElementType, vlmul, vta, vma>(
              args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}));
        } else {
          return OpVectorvx<intrinsics::Vmergevx<ElementType>,
                            ElementType,
                            vlmul,
                            vta,
                            // Always use "undisturbed" value from source register.
                            InactiveProcessing::kUndisturbed>(
              args.dst, args.src, BitCastToUnsigned(SignedType{args.imm}), /*dst_mask=*/args.src);
        }
      default:
        Unimplemented();
    }
  }

  template <typename ElementType, VectorRegisterGroupMultiplier vlmul, TailProcessing vta, auto vma>
  void OpVector(const Decoder::VOpIVvArgs& args) {
    using SignedType = berberis::SignedType<ElementType>;
    using UnsignedType = berberis::UnsignedType<ElementType>;
    switch (args.opcode) {
      case Decoder::VOpIVvOpcode::kVaddvv:
        return OpVectorvv<intrinsics::Vaddvv<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVsubvv:
        return OpVectorvv<intrinsics::Vsubvv<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVandvv:
        return OpVectorvv<intrinsics::Vandvv<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVorvv:
        return OpVectorvv<intrinsics::Vorvv<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVxorvv:
        return OpVectorvv<intrinsics::Vxorvv<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVmseqvv:
        return OpVectormvv<intrinsics::Vseqvv<ElementType>, ElementType, vlmul, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVmsnevv:
        return OpVectormvv<intrinsics::Vsnevv<ElementType>, ElementType, vlmul, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVmsltuvv:
        return OpVectormvv<intrinsics::Vsltvv<UnsignedType>, ElementType, vlmul, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVmsltvv:
        return OpVectormvv<intrinsics::Vsltvv<SignedType>, ElementType, vlmul, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVmsleuvv:
        return OpVectormvv<intrinsics::Vslevv<UnsignedType>, ElementType, vlmul, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVmslevv:
        return OpVectormvv<intrinsics::Vslevv<SignedType>, ElementType, vlmul, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVsllvv:
        return OpVectorvv<intrinsics::Vslvv<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVsrlvv:
        return OpVectorvv<intrinsics::Vsrvv<UnsignedType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVsravv:
        return OpVectorvv<intrinsics::Vsrvv<SignedType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVminuvv:
        return OpVectorvv<intrinsics::Vminvv<UnsignedType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVminvv:
        return OpVectorvv<intrinsics::Vminvv<SignedType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVmaxuvv:
        return OpVectorvv<intrinsics::Vmaxvv<UnsignedType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVmaxvv:
        return OpVectorvv<intrinsics::Vmaxvv<SignedType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpIVvOpcode::kVmergevv:
        if constexpr (std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>) {
          return OpVectorvv<intrinsics::Vmergevv<ElementType>, ElementType, vlmul, vta, vma>(
              args.dst, args.src1, args.src2);
        } else {
          return OpVectorvv<intrinsics::Vmergevv<ElementType>,
                            ElementType,
                            vlmul,
                            vta,
                            // Always use "undisturbed" value from source register.
                            InactiveProcessing::kUndisturbed>(
              args.dst, args.src1, args.src2, /*dst_mask=*/args.src1);
        }
      default:
        Unimplemented();
    }
  }

  template <typename ElementType, VectorRegisterGroupMultiplier vlmul, TailProcessing vta, auto vma>
  void OpVector(const Decoder::VOpMVvArgs& args) {
    using SignedType = berberis::SignedType<ElementType>;
    using UnsignedType = berberis::UnsignedType<ElementType>;
    if constexpr (std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>) {
      switch (args.opcode) {
        case Decoder::VOpMVvOpcode::kVmandnmm:
          return OpVectormm<[](SIMD128Register lhs, SIMD128Register rhs) { return lhs & ~rhs; }>(
              args.dst, args.src1, args.src2);
        case Decoder::VOpMVvOpcode::kVmandmm:
          return OpVectormm<[](SIMD128Register lhs, SIMD128Register rhs) { return lhs & rhs; }>(
              args.dst, args.src1, args.src2);
        case Decoder::VOpMVvOpcode::kVmormm:
          return OpVectormm<[](SIMD128Register lhs, SIMD128Register rhs) { return lhs | rhs; }>(
              args.dst, args.src1, args.src2);
        case Decoder::VOpMVvOpcode::kVmxormm:
          return OpVectormm<[](SIMD128Register lhs, SIMD128Register rhs) { return lhs ^ rhs; }>(
              args.dst, args.src1, args.src2);
        case Decoder::VOpMVvOpcode::kVmornmm:
          return OpVectormm<[](SIMD128Register lhs, SIMD128Register rhs) { return lhs | ~rhs; }>(
              args.dst, args.src1, args.src2);
        case Decoder::VOpMVvOpcode::kVmnandmm:
          return OpVectormm<[](SIMD128Register lhs, SIMD128Register rhs) { return ~(lhs & rhs); }>(
              args.dst, args.src1, args.src2);
        case Decoder::VOpMVvOpcode::kVmnormm:
          return OpVectormm<[](SIMD128Register lhs, SIMD128Register rhs) { return ~(lhs | rhs); }>(
              args.dst, args.src1, args.src2);
        case Decoder::VOpMVvOpcode::kVmxnormm:
          return OpVectormm<[](SIMD128Register lhs, SIMD128Register rhs) { return ~(lhs ^ rhs); }>(
              args.dst, args.src1, args.src2);
        default:;  // Do nothing: handled in next switch.
      }
    }
    switch (args.opcode) {
      case Decoder::VOpMVvOpcode::kVredsumvs:
        return OpVectorvs<intrinsics::Vredsumvs<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVredandvs:
        return OpVectorvs<intrinsics::Vredandvs<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVredorvs:
        return OpVectorvs<intrinsics::Vredorvs<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVredxorvs:
        return OpVectorvs<intrinsics::Vredxorvs<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVredminuvs:
        return OpVectorvs<intrinsics::Vredminvs<UnsignedType>, UnsignedType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVredminvs:
        return OpVectorvs<intrinsics::Vredminvs<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVredmaxuvs:
        return OpVectorvs<intrinsics::Vredmaxvs<UnsignedType>, UnsignedType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVredmaxvs:
        return OpVectorvs<intrinsics::Vredmaxvs<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVXmXXs:
        switch (args.vXmXXs_opcode) {
          case Decoder::VXmXXsOpcode::kVcpopm:
              return OpVectorVXmXXs<intrinsics::Vcpopm<Int128>, vma>(args.dst, args.src1);
          case Decoder::VXmXXsOpcode::kVfirstm:
              return OpVectorVXmXXs<intrinsics::Vfirstm<Int128>, vma>(args.dst, args.src1);
          default:
              return Unimplemented();
        }
      case Decoder::VOpMVvOpcode::kVmsXf:
        switch (args.vmsXf_opcode) {
          case Decoder::VmsXfOpcode::kVmsbfm:
              return OpVectorVmsXf<intrinsics::Vmsbf<>, vma>(args.dst, args.src1);
          case Decoder::VmsXfOpcode::kVmsofm:
              return OpVectorVmsXf<intrinsics::Vmsof<>, vma>(args.dst, args.src1);
          case Decoder::VmsXfOpcode::kVmsifm:
              return OpVectorVmsXf<intrinsics::Vmsif<>, vma>(args.dst, args.src1);
          default:
              return Unimplemented();
        }
      case Decoder::VOpMVvOpcode::kVmaddvv:
        return OpVectorvvv<intrinsics::Vmaddvv<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVnmsubvv:
        return OpVectorvvv<intrinsics::Vnmsubvv<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVmaccvv:
        return OpVectorvvv<intrinsics::Vmaccvv<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVnmsacvv:
        return OpVectorvvv<intrinsics::Vnmsacvv<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVmulhuvv:
        return OpVectorvv<intrinsics::Vmulhvv<UnsignedType>, UnsignedType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVmulvv:
        return OpVectorvv<intrinsics::Vmulvv<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVmulhsuvv:
        return OpVectorvv<intrinsics::Vmulhsuvv<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVmulhvv:
        return OpVectorvv<intrinsics::Vmulhvv<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src1, args.src2);
      case Decoder::VOpMVvOpcode::kVwaddvv:
        if constexpr (sizeof(ElementType) == sizeof(Int64) ||
                      vlmul == VectorRegisterGroupMultiplier::k8registers) {
          return Unimplemented();
        } else {
          return OpVectorwvv<intrinsics::Vwaddvv<ElementType>, ElementType, vlmul, vta, vma>(
              args.dst, args.src1, args.src2);
        }
      default:
        Unimplemented();
    }
  }

  template <typename ElementType, VectorRegisterGroupMultiplier vlmul, TailProcessing vta, auto vma>
  void OpVector(const Decoder::VOpIVxArgs& args, Register arg2) {
    using SignedType = berberis::SignedType<ElementType>;
    using UnsignedType = berberis::UnsignedType<ElementType>;
    switch (args.opcode) {
      case Decoder::VOpIVxOpcode::kVaddvx:
        return OpVectorvx<intrinsics::Vaddvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpIVxOpcode::kVsubvx:
        return OpVectorvx<intrinsics::Vsubvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpIVxOpcode::kVrsubvx:
        return OpVectorvx<intrinsics::Vrsubvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpIVxOpcode::kVandvx:
        return OpVectorvx<intrinsics::Vandvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpIVxOpcode::kVorvx:
        return OpVectorvx<intrinsics::Vorvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpIVxOpcode::kVxorvx:
        return OpVectorvx<intrinsics::Vxorvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpIVxOpcode::kVmseqvx:
        return OpVectormvx<intrinsics::Vseqvx<ElementType>, ElementType, vlmul, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpIVxOpcode::kVmsnevx:
        return OpVectormvx<intrinsics::Vsnevx<ElementType>, ElementType, vlmul, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpIVxOpcode::kVmsltuvx:
        return OpVectormvx<intrinsics::Vsltvx<UnsignedType>, UnsignedType, vlmul, vma>(
            args.dst, args.src1, MaybeTruncateTo<UnsignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVmsltvx:
        return OpVectormvx<intrinsics::Vsltvx<SignedType>, SignedType, vlmul, vma>(
            args.dst, args.src1, MaybeTruncateTo<SignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVmsleuvx:
        return OpVectormvx<intrinsics::Vslevx<UnsignedType>, UnsignedType, vlmul, vma>(
            args.dst, args.src1, MaybeTruncateTo<UnsignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVmslevx:
        return OpVectormvx<intrinsics::Vslevx<SignedType>, SignedType, vlmul, vma>(
            args.dst, args.src1, MaybeTruncateTo<SignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVmsgtuvx:
        return OpVectormvx<intrinsics::Vsgtvx<UnsignedType>, UnsignedType, vlmul, vma>(
            args.dst, args.src1, MaybeTruncateTo<UnsignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVmsgtvx:
        return OpVectormvx<intrinsics::Vsgtvx<SignedType>, SignedType, vlmul, vma>(
            args.dst, args.src1, MaybeTruncateTo<SignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVsllvx:
        return OpVectorvx<intrinsics::Vslvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpIVxOpcode::kVsrlvx:
        return OpVectorvx<intrinsics::Vsrvx<UnsignedType>, UnsignedType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<UnsignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVsravx:
        return OpVectorvx<intrinsics::Vsrvx<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<SignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVminuvx:
        return OpVectorvx<intrinsics::Vminvx<UnsignedType>, UnsignedType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<UnsignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVminvx:
        return OpVectorvx<intrinsics::Vminvx<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<SignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVmaxuvx:
        return OpVectorvx<intrinsics::Vmaxvx<UnsignedType>, UnsignedType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<UnsignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVmaxvx:
        return OpVectorvx<intrinsics::Vmaxvx<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<SignedType>(arg2));
      case Decoder::VOpIVxOpcode::kVmergevx:
        if constexpr (std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>) {
          return OpVectorvx<intrinsics::Vmergevx<ElementType>, ElementType, vlmul, vta, vma>(
              args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
        } else {
          return OpVectorvx<intrinsics::Vmergevx<ElementType>,
                            ElementType,
                            vlmul,
                            vta,
                            // Always use "undisturbed" value from source register.
                            InactiveProcessing::kUndisturbed>(
              args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2), /*dst_mask=*/args.src1);
        }
      default:
        Unimplemented();
    }
  }

  template <typename ElementType, VectorRegisterGroupMultiplier vlmul, TailProcessing vta, auto vma>
  void OpVector(const Decoder::VOpMVxArgs& args, Register arg2) {
    using SignedType = berberis::SignedType<ElementType>;
    using UnsignedType = berberis::UnsignedType<ElementType>;
    switch (args.opcode) {
      case Decoder::VOpMVxOpcode::kVmaddvx:
        return OpVectorvxv<intrinsics::Vmaddvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpMVxOpcode::kVnmsubvx:
        return OpVectorvxv<intrinsics::Vnmsubvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpMVxOpcode::kVmaccvx:
        return OpVectorvxv<intrinsics::Vmaccvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpMVxOpcode::kVnmsacvx:
        return OpVectorvxv<intrinsics::Vnmsacvx<ElementType>, ElementType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<ElementType>(arg2));
      case Decoder::VOpMVxOpcode::kVmulhuvx:
        return OpVectorvx<intrinsics::Vmulhvx<UnsignedType>, UnsignedType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<UnsignedType>(arg2));
      case Decoder::VOpMVxOpcode::kVmulvx:
        return OpVectorvx<intrinsics::Vmulvx<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<SignedType>(arg2));
      case Decoder::VOpMVxOpcode::kVmulhsuvx:
        return OpVectorvx<intrinsics::Vmulhsuvx<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<SignedType>(arg2));
      case Decoder::VOpMVxOpcode::kVmulhvx:
        return OpVectorvx<intrinsics::Vmulhvx<SignedType>, SignedType, vlmul, vta, vma>(
            args.dst, args.src1, MaybeTruncateTo<SignedType>(arg2));
      default:
        Unimplemented();
    }
  }

  template <typename ElementType, VectorRegisterGroupMultiplier vlmul, TailProcessing vta, auto vma>
  void OpVector(const Decoder::VStoreUnitStrideArgs& /*args*/, Register /*src*/) {
    Unimplemented();
  }

  template <auto Intrinsic, auto vma>
  void OpVectorVXmXXs(uint8_t dst, uint8_t src1) {
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    if (vstart != 0) {
      return Unimplemented();
    }
    SIMD128Register arg1(state_->cpu.v[src1]);
    if constexpr (!std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>) {
      SIMD128Register mask(state_->cpu.v[0]);
      arg1 &= mask;
    }
    arg1 &= ~intrinsics::MakeBitmaskFromVl(vl);
    SIMD128Register result = std::get<0>(Intrinsic(arg1.Get<Int128>()));
    SetReg(dst, TruncateTo<UInt64>(BitCastToUnsigned(result.Get<Int128>())));
  }

  template <auto Intrinsic>
  void OpVectormm(uint8_t dst, uint8_t src1, uint8_t src2) {
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    SIMD128Register result, arg1, arg2;
    arg1.Set(state_->cpu.v[src1]);
    arg2.Set(state_->cpu.v[src2]);
    if (vstart > 0) [[unlikely]] {
      if (vstart >= vl) [[unlikely]] {
        result.Set(state_->cpu.v[dst]);
        result = result | intrinsics::MakeBitmaskFromVl(vl);
      } else {
        SIMD128Register start_mask = intrinsics::MakeBitmaskFromVl(vstart);
        result.Set(state_->cpu.v[dst]);
        result = (result & ~start_mask) | (Intrinsic(arg1, arg2) & start_mask) |
                 intrinsics::MakeBitmaskFromVl(vl);
      }
      SetCsr<CsrName::kVstart>(0);
    } else {
      result = Intrinsic(arg1, arg2) | intrinsics::MakeBitmaskFromVl(vl);
    }
    state_->cpu.v[dst] = result.Get<__uint128_t>();
  }

  template <auto Intrinsic, auto vma>
  void OpVectorVmsXf(uint8_t dst, uint8_t src1) {
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    if (vstart != 0) {
      return Unimplemented();
    }
    SIMD128Register arg1(state_->cpu.v[src1]);
    SIMD128Register tail_mask = intrinsics::MakeBitmaskFromVl(vl);
    SIMD128Register mask;
    if constexpr (!std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>) {
      mask.Set<__uint128_t>(state_->cpu.v[0]);
      arg1 &= mask;
    }
    arg1 &= ~tail_mask;
    SIMD128Register result = std::get<0>(Intrinsic(arg1.Get<Int128>()));
    if constexpr (!std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>) {
      arg1 &= mask;
      if (vma == InactiveProcessing::kUndisturbed) {
        result = (result & mask) | (SIMD128Register(state_->cpu.v[dst]) & ~mask);
      } else {
        result |= ~mask;
      }
    }
    result |= tail_mask;
    state_->cpu.v[dst] = result.Get<__uint128_t>();
  }

  template <auto Intrinsic, typename ElementType, VectorRegisterGroupMultiplier vlmul, auto vma>
  void OpVectormvv(uint8_t dst, uint8_t src1, uint8_t src2) {
    constexpr size_t kRegistersInvolved = NumberOfRegistersInvolved(vlmul);
    if (!IsAligned<kRegistersInvolved>(src1 | src2)) {
      return Unimplemented();
    }
    SIMD128Register original_result(state_->cpu.v[dst]);
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    SIMD128Register result_before_vl_masking;
    if (vstart >= vl) [[unlikely]] {
      result_before_vl_masking = original_result;
      SetCsr<CsrName::kVstart>(0);
    } else {
      result_before_vl_masking =
          CollectBitmaskResult<ElementType, vlmul>([this, src1, src2](auto index) {
            SIMD128Register arg1(state_->cpu.v[src1 + static_cast<size_t>(index)]);
            SIMD128Register arg2(state_->cpu.v[src2 + static_cast<size_t>(index)]);
            return Intrinsic(arg1, arg2);
          });
      SIMD128Register mask(state_->cpu.v[0]);
      if constexpr (!std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>) {
        if constexpr (vma == InactiveProcessing::kAgnostic) {
          result_before_vl_masking |= ~mask;
        } else {
          result_before_vl_masking = (mask & result_before_vl_masking) | (original_result & ~mask);
        }
      }
      if (vstart > 0) [[unlikely]] {
        SIMD128Register start_mask = intrinsics::MakeBitmaskFromVl(vstart);
        result_before_vl_masking =
            (original_result & ~start_mask) | (result_before_vl_masking & start_mask);
        SetCsr<CsrName::kVstart>(0);
      }
    }
    state_->cpu.v[dst] =
        (result_before_vl_masking | intrinsics::MakeBitmaskFromVl(vl)).Get<__uint128_t>();
  }

  template <auto Intrinsic, typename ElementType, VectorRegisterGroupMultiplier vlmul, auto vma>
  void OpVectormvx(uint8_t dst, uint8_t src1, ElementType arg2) {
    constexpr size_t kRegistersInvolved = NumberOfRegistersInvolved(vlmul);
    if (!IsAligned<kRegistersInvolved>(src1)) {
      return Unimplemented();
    }
    SIMD128Register original_result(state_->cpu.v[dst]);
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    SIMD128Register result_before_vl_masking;
    if (vstart >= vl) [[unlikely]] {
      result_before_vl_masking = original_result;
      SetCsr<CsrName::kVstart>(0);
    } else {
      result_before_vl_masking =
          CollectBitmaskResult<ElementType, vlmul>([this, src1, arg2](auto index) {
            SIMD128Register arg1(state_->cpu.v[src1 + static_cast<size_t>(index)]);
            return Intrinsic(arg1, arg2);
          });
      if constexpr (!std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>) {
        SIMD128Register mask(state_->cpu.v[0]);
        if constexpr (vma == InactiveProcessing::kAgnostic) {
          result_before_vl_masking |= ~mask;
        } else {
          result_before_vl_masking = (mask & result_before_vl_masking) | (original_result & ~mask);
        }
      }
      if (vstart > 0) [[unlikely]] {
        SIMD128Register start_mask = intrinsics::MakeBitmaskFromVl(vstart);
        result_before_vl_masking =
            (original_result & ~start_mask) | (result_before_vl_masking & start_mask);
        SetCsr<CsrName::kVstart>(0);
      }
    }
    state_->cpu.v[dst] =
        (result_before_vl_masking | intrinsics::MakeBitmaskFromVl(vl)).Get<__uint128_t>();
  }

  template <auto Intrinsic,
            typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            auto vma>
  void OpVectorvs(uint8_t dst, uint8_t src1, uint8_t src2) {
    constexpr size_t kRegistersInvolved = NumberOfRegistersInvolved(vlmul);
    if (!IsAligned<kRegistersInvolved>(dst | src1 | src2)) {
      return Unimplemented();
    }
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    if (vstart != 0) {
      return Unimplemented();
    }
    SIMD128Register result;
    auto mask = GetMaskForVectorOperations<vma>();
    ElementType arg1 = SIMD128Register{state_->cpu.v[src1]}.Get<ElementType>(0);
    for (size_t index = 0; index < kRegistersInvolved; ++index) {
      using MaskType = std::conditional_t<sizeof(ElementType) == sizeof(Int8), UInt16, UInt8>;
      const MaskType element_count{static_cast<typename MaskType::BaseType>(
          std::min(static_cast<int>(16 / sizeof(ElementType)), vl))};
      auto mask_bits = std::get<0>(intrinsics::MaskForRegisterInSequence<ElementType>(mask, index));
      SIMD128Register arg2(state_->cpu.v[src2 + index]);
      for (MaskType element_index = MaskType{0}; element_index < element_count;
           element_index += MaskType{1}) {
        if constexpr (!std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>) {
          if ((MaskType{mask_bits} & (MaskType{1} << element_index)) == MaskType{0}) {
              continue;
          }
        }
        result = std::get<0>(Intrinsic(arg1, arg2.Get<ElementType>(element_index)));
        arg1 = result.Get<ElementType>(0);
      }
    }
    result.Set(state_->cpu.v[dst]);
    result.Set(arg1, 0);
    result = intrinsics::VectorMasking<ElementType, vta>(result, result, 0, 1);
    state_->cpu.v[dst] = result.Get<__uint128_t>();
    SetCsr<CsrName::kVstart>(0);
  }

  template <auto Intrinsic,
            typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            auto vma,
            typename... DstMaskType>
  void OpVectorvv(uint8_t dst, uint8_t src1, uint8_t src2, DstMaskType... dst_mask) {
    // Note: for the most instructions dst_mask is the same as dst and thus is not supplied
    // separately, but for vmerge.vvm it's the same as src1.
    // Since it's always one of dst, src1, or src2 there are no need to check alignment separately.
    static_assert(sizeof...(dst_mask) <= 1);
    constexpr size_t kRegistersInvolved = NumberOfRegistersInvolved(vlmul);
    if (!IsAligned<kRegistersInvolved>(dst | src1 | src2)) {
      return Unimplemented();
    }
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    auto mask = GetMaskForVectorOperations<vma>();
    for (size_t index = 0; index < kRegistersInvolved; ++index) {
      SIMD128Register result{state_->cpu.v[dst + index]};
      SIMD128Register result_mask;
      if constexpr (sizeof...(DstMaskType) == 0) {
        result_mask.Set(state_->cpu.v[dst + index]);
      } else {
        uint8_t dst_mask_unpacked[1] = {dst_mask...};
        result_mask.Set(state_->cpu.v[dst_mask_unpacked[0] + index]);
      }
      SIMD128Register arg1{state_->cpu.v[src1 + index]};
      SIMD128Register arg2{state_->cpu.v[src2 + index]};
      result = intrinsics::VectorMasking<ElementType, vta, vma>(
          result,
          std::get<0>(Intrinsic(arg1, arg2)),
          result_mask,
          vstart - index * (16 / sizeof(ElementType)),
          vl - index * (16 / sizeof(ElementType)),
          std::get<0>(intrinsics::MaskForRegisterInSequence<ElementType>(mask, index)));
      state_->cpu.v[dst + index] = result.Get<__uint128_t>();
    }
    SetCsr<CsrName::kVstart>(0);
  }

  template <auto Intrinsic,
            typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            auto vma>
  void OpVectorvvv(uint8_t dst, uint8_t src1, uint8_t src2) {
    constexpr size_t kRegistersInvolved = NumberOfRegistersInvolved(vlmul);
    if (!IsAligned<kRegistersInvolved>(dst | src1 | src2)) {
      return Unimplemented();
    }
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    auto mask = GetMaskForVectorOperations<vma>();
    for (size_t index = 0; index < kRegistersInvolved; ++index) {
      SIMD128Register result(state_->cpu.v[dst + index]);
      SIMD128Register arg1(state_->cpu.v[src1 + index]);
      SIMD128Register arg2(state_->cpu.v[src2 + index]);
      result = intrinsics::VectorMasking<ElementType, vta, vma>(
          result,
          std::get<0>(Intrinsic(arg1, arg2, result)),
          vstart - index * (16 / sizeof(ElementType)),
          vl - index * (16 / sizeof(ElementType)),
          std::get<0>(intrinsics::MaskForRegisterInSequence<ElementType>(mask, index)));
      state_->cpu.v[dst + index] = result.Get<__uint128_t>();
    }
    SetCsr<CsrName::kVstart>(0);
  }

  template <auto Intrinsic,
            typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            auto vma>
  void OpVectorwvv(uint8_t dst, uint8_t src1, uint8_t src2) {
    constexpr size_t kRegistersInvolved = NumberOfRegistersInvolved(vlmul);
    constexpr size_t kDestRegistersInvolved = NumDestRegistersInvolvedForW(vlmul);
    if (!IsAligned<kDestRegistersInvolved>(dst) || !IsAligned<kRegistersInvolved>(src1 | src2)) {
      return Unimplemented();
    }
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    auto mask = GetMaskForVectorOperations<vma>();
    for (size_t index = 0; index < kRegistersInvolved; ++index) {
      SIMD128Register result(state_->cpu.v[dst + 2 * index]);
      SIMD128Register arg1(state_->cpu.v[src1 + index]);
      SIMD128Register arg2(state_->cpu.v[src2 + index]);
      result = intrinsics::VectorMasking<decltype(Widen(ElementType{0})), vta, vma>(
          result,
          std::get<0>(Intrinsic(arg1, arg2)),
          vstart - index * (16 / sizeof(ElementType)),
          vl - index * (16 / sizeof(ElementType)),
          std::get<0>(intrinsics::MaskForRegisterInSequence<decltype(Widen(ElementType{0}))>(
              mask, (2 * index))));
      state_->cpu.v[dst + 2 * index] = result.Get<__uint128_t>();
      if constexpr (kDestRegistersInvolved > 1) {  // if lmul is one full register or more
        result.Set(state_->cpu.v[dst + 2 * index + 1]);
        std::tie(arg1) = intrinsics::VMovTopHalfToBottom<ElementType>(arg1);
        std::tie(arg2) = intrinsics::VMovTopHalfToBottom<ElementType>(arg2);
        result = intrinsics::VectorMasking<decltype(Widen(ElementType{0})), vta, vma>(
            result,
            std::get<0>(Intrinsic(arg1, arg2)),
            vstart - index * (16 / sizeof(ElementType)) - (8 / sizeof(ElementType)),
            vl - index * (16 / sizeof(ElementType)) - (8 / sizeof(ElementType)),
            std::get<0>(intrinsics::MaskForRegisterInSequence<decltype(Widen(ElementType{0}))>(
                mask, (2 * index) + 1)));
        state_->cpu.v[dst + 2 * index + 1] = result.Get<__uint128_t>();
      }
    }
    SetCsr<CsrName::kVstart>(0);
  }

  template <auto Intrinsic,
            typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            auto vma,
            typename... DstMaskType>
  void OpVectorvx(uint8_t dst, uint8_t src1, ElementType arg2, DstMaskType... dst_mask) {
    // Note: for the most instructions dst_mask is the same as dst and thus is not supplied
    // separately, but for vmerge.vxm it's the same as src1.
    // Since it's always one of dst, src1, or src2 there are no need to check alignment separately.
    static_assert(sizeof...(dst_mask) <= 1);
    constexpr size_t kRegistersInvolved = NumberOfRegistersInvolved(vlmul);
    if (!IsAligned<kRegistersInvolved>(dst | src1)) {
      return Unimplemented();
    }
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    auto mask = GetMaskForVectorOperations<vma>();
    for (size_t index = 0; index < kRegistersInvolved; ++index) {
      SIMD128Register result(state_->cpu.v[dst + index]);
      SIMD128Register result_mask;
      if constexpr (sizeof...(DstMaskType) == 0) {
        result_mask.Set(state_->cpu.v[dst + index]);
      } else {
        uint8_t dst_mask_unpacked[1] = {dst_mask...};
        result_mask.Set(state_->cpu.v[dst_mask_unpacked[0] + index]);
      }
      SIMD128Register arg1(state_->cpu.v[src1 + index]);
      result = intrinsics::VectorMasking<ElementType, vta, vma>(
          result,
          std::get<0>(Intrinsic(arg1, arg2)),
          result_mask,
          vstart - index * (16 / sizeof(ElementType)),
          vl - index * (16 / sizeof(ElementType)),
          std::get<0>(intrinsics::MaskForRegisterInSequence<ElementType>(mask, index)));
      state_->cpu.v[dst + index] = result.Get<__uint128_t>();
    }
    SetCsr<CsrName::kVstart>(0);
  }

  template <auto Intrinsic,
            typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            TailProcessing vta,
            auto vma>
  void OpVectorvxv(uint8_t dst, uint8_t src1, ElementType arg2) {
    constexpr size_t kRegistersInvolved = NumberOfRegistersInvolved(vlmul);
    if (!IsAligned<kRegistersInvolved>(dst | src1)) {
      return Unimplemented();
    }
    int vstart = GetCsr<CsrName::kVstart>();
    int vl = GetCsr<CsrName::kVl>();
    auto mask = GetMaskForVectorOperations<vma>();
    for (size_t index = 0; index < kRegistersInvolved; ++index) {
      SIMD128Register result(state_->cpu.v[dst + index]);
      SIMD128Register arg1(state_->cpu.v[src1 + index]);
      result = intrinsics::VectorMasking<ElementType, vta, vma>(
          result,
          std::get<0>(Intrinsic(arg1, arg2, result)),
          vstart - index * (16 / sizeof(ElementType)),
          vl - index * (16 / sizeof(ElementType)),
          std::get<0>(intrinsics::MaskForRegisterInSequence<ElementType>(mask, index)));
      state_->cpu.v[dst + index] = result.Get<__uint128_t>();
    }
    SetCsr<CsrName::kVstart>(0);
  }

  // Helper function needed to generate bitmak result from non-bitmask inputs.
  // We are processing between 1 and 8 registers here and each register produces between 2 bits
  // (for 64 bit inputs) and 16 bits (for 8 bit inputs) bitmasks which are then combined into
  // final result (between 2 and 128 bits long).
  // Note that we are not handling tail here! These bits remain undefined and should be handled
  // later.
  // TODO(b/317757595): Add separate tests to verify the logic.
  template <typename ElementType, VectorRegisterGroupMultiplier vlmul, typename Intrinsic>
  SIMD128Register CollectBitmaskResult(Intrinsic intrinsic) {
    constexpr size_t kRegistersInvolved = NumberOfRegistersInvolved(vlmul);
    // We employ two distinct tactics to handle all possibilities:
    //   1. For 8bit/16bit types we get full UInt8/UInt16 result and thus use SIMD128Register.Set.
    //   2. For 32bit/64bit types we only get 2bit or 4bit from each call and thus need to use
    //      shifts to accumulate the result.
    //      But since each of up to 8 results is at most 4bits total bitmask is 32bit (or less).
    std::conditional_t<sizeof(ElementType) < sizeof(UInt32), SIMD128Register, UInt32>
        bitmask_result{};
    for (UInt32 index = UInt32{0}; index < UInt32(kRegistersInvolved); index += UInt32{1}) {
      auto raw_result = intrinsics::SimdMaskToBitMask<ElementType>(std::get<0>(intrinsic(index)));
      if constexpr (sizeof(ElementType) < sizeof(Int32)) {
        bitmask_result.Set(raw_result, index);
      } else {
        constexpr UInt32 kElemNum =
            UInt32{static_cast<uint32_t>((sizeof(SIMD128Register) / sizeof(ElementType)))};
        bitmask_result |= UInt32(UInt8(raw_result)) << (index * kElemNum);
      }
    }
    return SIMD128Register(bitmask_result);
  }

  void Nop() {}

  void Unimplemented() {
    UndefinedInsn(GetInsnAddr());
    // If there is a guest handler registered for SIGILL we'll delay its processing until the next
    // sync point (likely the main dispatching loop) due to enabled pending signals. Thus we must
    // ensure that insn_addr isn't automatically advanced in FinalizeInsn.
    exception_raised_ = true;
  }

  //
  // Guest state getters/setters.
  //

  Register GetReg(uint8_t reg) const {
    CheckRegIsValid(reg);
    return state_->cpu.x[reg];
  }

  void SetReg(uint8_t reg, Register value) {
    if (exception_raised_) {
      // Do not produce side effects.
      return;
    }
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
    CHECK(!exception_raised_);
    state_->cpu.*CsrFieldAddr<kName> = arg & kCsrMask<kName>;
  }

  [[nodiscard]] uint64_t GetImm(uint64_t imm) const { return imm; }

  [[nodiscard]] Register Copy(Register value) const { return value; }

  [[nodiscard]] GuestAddr GetInsnAddr() const { return state_->cpu.insn_addr; }

  void FinalizeInsn(uint8_t insn_len) {
    if (!branch_taken_ && !exception_raised_) {
      state_->cpu.insn_addr += insn_len;
    }
  }

#include "berberis/intrinsics/interpreter_intrinsics_hooks-inl.h"

 private:
  template <typename DataType>
  Register Load(const void* ptr) {
    static_assert(std::is_integral_v<DataType>);
    CHECK(!exception_raised_);
    FaultyLoadResult result = FaultyLoad(ptr, sizeof(DataType));
    if (result.is_fault) {
      exception_raised_ = true;
      return {};
    }
    return static_cast<DataType>(result.value);
  }

  template <typename DataType>
  void Store(void* ptr, uint64_t data) {
    static_assert(std::is_integral_v<DataType>);
    CHECK(!exception_raised_);
    exception_raised_ = FaultyStore(ptr, sizeof(DataType), data);
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
    CHECK_LE(reg, std::size(state_->cpu.x));
  }

  void CheckFpRegIsValid(uint8_t reg) const { CHECK_LT(reg, std::size(state_->cpu.f)); }

  template <auto vma>
  std::conditional_t<std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>,
                     intrinsics::NoInactiveProcessing,
                     SIMD128Register>
  GetMaskForVectorOperations() {
    if constexpr (std::is_same_v<decltype(vma), intrinsics::NoInactiveProcessing>) {
      return intrinsics::NoInactiveProcessing{};
    } else {
      return {state_->cpu.v[0]};
    }
  }

  ThreadState* state_;
  bool branch_taken_;
  // This flag is set by illegal instructions and faulted memory accesses. The former must always
  // stop the playback of the current instruction, so we don't need to do anything special. The
  // latter may result in having more operations with side effects called before the end of the
  // current instruction:
  //   Load (faulted)    -> SetReg
  //   LoadFp (faulted)  -> NanBoxAndSetFpReg
  // If an exception is raised before these operations, we skip them. For all other operations with
  // side-effects we check that this flag is never raised.
  bool exception_raised_;
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
  CHECK(!exception_raised_);
  FeSetExceptions(arg & 0b1'1111);
  arg = (arg >> 5) & kCsrMask<CsrName::kFrm>;
  state_->cpu.frm = arg;
  FeSetRound(arg);
}

template <>
void Interpreter::SetCsr<CsrName::kFFlags>(Register arg) {
  CHECK(!exception_raised_);
  FeSetExceptions(arg & 0b1'1111);
}

template <>
void Interpreter::SetCsr<CsrName::kFrm>(Register arg) {
  CHECK(!exception_raised_);
  arg &= kCsrMask<CsrName::kFrm>;
  state_->cpu.frm = arg;
  FeSetRound(arg);
}

template <>
void Interpreter::SetCsr<CsrName::kVxrm>(Register arg) {
  CHECK(!exception_raised_);
  state_->cpu.*CsrFieldAddr<CsrName::kVcsr> =
      (state_->cpu.*CsrFieldAddr<CsrName::kVcsr> & 0b100) | (arg & 0b11);
}

template <>
void Interpreter::SetCsr<CsrName::kVxsat>(Register arg) {
  CHECK(!exception_raised_);
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
  if (exception_raised_) {
    // Do not produce side effects.
    return;
  }
  CheckFpRegIsValid(reg);
  state_->cpu.f[reg] = NanBox<Float32>(value);
}

template <>
void Interpreter::NanBoxAndSetFpReg<Interpreter::Float64>(uint8_t reg, FpRegister value) {
  if (exception_raised_) {
    // Do not produce side effects.
    return;
  }
  CheckFpRegIsValid(reg);
  state_->cpu.f[reg] = value;
}

}  // namespace

void InitInterpreter() {
  AddFaultyMemoryAccessRecoveryCode();
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
