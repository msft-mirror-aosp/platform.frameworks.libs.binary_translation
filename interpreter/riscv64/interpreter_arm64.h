/*
 * Copyright (C) 2024 The Android Open Source Project
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

#include <cstdint>
#include <cstdlib>

#include "berberis/base/bit_util.h"
#include "berberis/decoder/riscv64/decoder.h"
#include "berberis/decoder/riscv64/semantics_player.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

class Interpreter {
 public:
  using CsrName = berberis::CsrName;
  using Decoder = Decoder<SemanticsPlayer<Interpreter>>;
  using Register = uint64_t;
  static constexpr Register no_register = 0;
  using FpRegister = uint64_t;
  static constexpr FpRegister no_fp_register = 0;
  using Float32 = float;
  using Float64 = double;

  explicit Interpreter(ThreadState* state)
      : state_(state), branch_taken_(false), exception_raised_(false) {}

  //
  // Instruction implementations.
  //

  Register UpdateCsr(Decoder::CsrOpcode opcode, Register arg, Register csr) {
    UNUSED(opcode, arg, csr);
    Undefined();
    return {};
  }

  Register UpdateCsr(Decoder::CsrImmOpcode opcode, uint8_t imm, Register csr) {
    UNUSED(opcode, imm, csr);
    Undefined();
    return {};
  }

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
    UNUSED(sw, sr, pw, pr);
    Undefined();
    return;
  }

  template <typename IntType, bool aq, bool rl>
  Register Lr(int64_t addr) {
    UNUSED(addr);
    Undefined();
    return {};
  }

  template <typename IntType, bool aq, bool rl>
  Register Sc(int64_t addr, IntType val) {
    UNUSED(addr, val);
    Undefined();
    return {};
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
      case Decoder::OpOpcode::kAndn:
        return Int64(arg1) & (~Int64(arg2));
      case Decoder::OpOpcode::kOrn:
        return Int64(arg1) | (~Int64(arg2));
      case Decoder::OpOpcode::kXnor:
        return ~(Int64(arg1) ^ Int64(arg2));
      default:
        Undefined();
        return {};
    }
  }

  Register Op32(Decoder::Op32Opcode opcode, Register arg1, Register arg2) {
    UNUSED(opcode, arg1, arg2);
    Undefined();
    return {};
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
        Undefined();
        return {};
    }
  }

  template <typename DataType>
  FpRegister LoadFp(Register arg, int16_t offset) {
    UNUSED(arg, offset);
    Undefined();
    return {};
  }

  Register OpImm(Decoder::OpImmOpcode opcode, Register arg, int16_t imm) {
    UNUSED(opcode, arg, imm);
    Undefined();
    return {};
  }

  Register Lui(int32_t imm) {
    UNUSED(imm);
    Undefined();
    return {};
  }

  Register Auipc(int32_t imm) {
    UNUSED(imm);
    Undefined();
    return {};
  }

  Register OpImm32(Decoder::OpImm32Opcode opcode, Register arg, int16_t imm) {
    UNUSED(opcode, arg, imm);
    Undefined();
    return {};
  }

  // TODO(b/232598137): rework ecall to not take parameters explicitly.
  Register Ecall(Register /* syscall_nr */,
                 Register /* arg0 */,
                 Register /* arg1 */,
                 Register /* arg2 */,
                 Register /* arg3 */,
                 Register /* arg4 */,
                 Register /* arg5 */) {
    Undefined();
    return {};
  }

  Register Slli(Register arg, int8_t imm) { return arg << imm; }

  Register Srli(Register arg, int8_t imm) { return arg >> imm; }

  Register Srai(Register arg, int8_t imm) { return bit_cast<int64_t>(arg) >> imm; }

  Register ShiftImm32(Decoder::ShiftImm32Opcode opcode, Register arg, uint16_t imm) {
    UNUSED(opcode, arg, imm);
    Undefined();
    return {};
  }

  Register Rori(Register arg, int8_t shamt) {
    UNUSED(arg, shamt);
    Undefined();
    return {};
  }

  Register Roriw(Register arg, int8_t shamt) {
    UNUSED(arg, shamt);
    Undefined();
    return {};
  }

  void Store(Decoder::MemoryDataOperandType operand_type,
             Register arg,
             int16_t offset,
             Register data) {
    void* ptr = ToHostAddr<void>(arg + offset);
    switch (operand_type) {
      case Decoder::MemoryDataOperandType::k8bit:
        Store<uint8_t>(ptr, data);
        break;
      case Decoder::MemoryDataOperandType::k16bit:
        Store<uint16_t>(ptr, data);
        break;
      case Decoder::MemoryDataOperandType::k32bit:
        Store<uint32_t>(ptr, data);
        break;
      case Decoder::MemoryDataOperandType::k64bit:
        Store<uint64_t>(ptr, data);
        break;
      default:
        return Undefined();
    }
  }

  template <typename DataType>
  void StoreFp(Register arg, int16_t offset, FpRegister data) {
    UNUSED(arg, offset, data);
    Undefined();
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
        return Undefined();
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

  enum class TailProcessing {
    kUndisturbed = 0,
    kAgnostic = 1,
  };

  enum class InactiveProcessing {
    kUndisturbed = 0,
    kAgnostic = 1,
  };

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

  static constexpr size_t NumRegistersInvolvedForWideOperand(VectorRegisterGroupMultiplier vlmul) {
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

  template <typename ElementType, VectorRegisterGroupMultiplier vlmul>
  static constexpr size_t GetVlmax() {
    return 0;
  }

  template <typename VOpArgs, typename... ExtraArgs>
  void OpVector(const VOpArgs& args, [[maybe_unused]] ExtraArgs... extra_args) {
    UNUSED(args);
    Undefined();
  }

  template <typename ElementType, typename VOpArgs, typename... ExtraArgs>
  void OpVector(const VOpArgs& args, Register vtype, [[maybe_unused]] ExtraArgs... extra_args) {
    UNUSED(args, vtype);
    Undefined();
  }

  template <typename ElementType, typename VOpArgs, typename... ExtraArgs>
  void OpVector(const VOpArgs& args,
                VectorRegisterGroupMultiplier vlmul,
                Register vtype,
                [[maybe_unused]] ExtraArgs... extra_args) {
    UNUSED(args, vlmul, vtype);
    Undefined();
  }

  template <typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            typename VOpArgs,
            typename... ExtraArgs>
  void OpVector(const VOpArgs& args, Register vtype, [[maybe_unused]] ExtraArgs... extra_args) {
    UNUSED(args, vtype);
    Undefined();
  }

  template <typename ElementType,
            VectorRegisterGroupMultiplier vlmul,
            auto vma,
            typename VOpArgs,
            typename... ExtraArgs>
  void OpVector(const VOpArgs& args, Register vtype, [[maybe_unused]] ExtraArgs... extra_args) {
    UNUSED(args, vtype);
    Undefined();
  }

  template <typename ElementType,
            size_t kSegmentSize,
            VectorRegisterGroupMultiplier vlmul,
            auto vma,
            typename VOpArgs,
            typename... ExtraArgs>
  void OpVector(const VOpArgs& args, Register vtype, [[maybe_unused]] ExtraArgs... extra_args) {
    UNUSED(args, vtype);
    Undefined();
  }

  template <size_t kSegmentSize,
            typename IndexElementType,
            size_t kIndexRegistersInvolved,
            TailProcessing vta,
            auto vma,
            typename VOpArgs,
            typename... ExtraArgs>
  void OpVector(const VOpArgs& args, Register vtype, [[maybe_unused]] ExtraArgs... extra_args) {
    UNUSED(args, vtype);
    Undefined();
  }

  template <typename DataElementType,
            size_t kSegmentSize,
            typename IndexElementType,
            size_t kIndexRegistersInvolved,
            TailProcessing vta,
            auto vma,
            typename VOpArgs,
            typename... ExtraArgs>
  void OpVector(const VOpArgs& args,
                VectorRegisterGroupMultiplier vlmul,
                [[maybe_unused]] ExtraArgs... extra_args) {
    UNUSED(args, vlmul);
    Undefined();
  }

  void Nop() {}

  void Undefined() {
    // If there is a guest handler registered for SIGILL we'll delay its processing until the next
    // sync point (likely the main dispatching loop) due to enabled pending signals. Thus we must
    // ensure that insn_addr isn't automatically advanced in FinalizeInsn.
    exception_raised_ = true;
    abort();
  }

  void Unimplemented() {
    // TODO(b/265372622): Replace with fatal from logging.h.
    abort();
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
  [[nodiscard]] Register GetCsr() {
    Undefined();
    return {};
  }

  template <CsrName kName>
  void SetCsr(Register arg) {
    UNUSED(arg);
    Undefined();
  }

  uint64_t GetImm(uint64_t imm) const { return imm; }

  [[nodiscard]] Register Copy(Register value) const { return value; }

  void FinalizeInsn(uint8_t insn_len) {
    if (!branch_taken_ && !exception_raised_) {
      state_->cpu.insn_addr += insn_len;
    }
  }

  [[nodiscard]] GuestAddr GetInsnAddr() const { return state_->cpu.insn_addr; }

#include "berberis/intrinsics/interpreter_intrinsics_hooks-inl.h"

 private:
  template <typename DataType>
  Register Load(const void* ptr) {
    // TODO(b/346603273): update to use faulty load
    static_assert(std::is_integral_v<DataType>);
    CHECK(!exception_raised_);
    return *static_cast<const DataType*>(ptr);
  }

  template <typename DataType>
  void Store(void* ptr, uint64_t data) const {
    // TODO(b/346603273): update to use faulty store
    auto* typed_ptr = static_cast<DataType*>(ptr);
    *typed_ptr = DataType(data);
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

  ProcessState* state_;
  bool branch_taken_;
  bool exception_raised_;
};

template <>
[[nodiscard]] Interpreter::FpRegister inline Interpreter::GetFRegAndUnboxNan<Interpreter::Float32>(
    uint8_t reg) {
  UNUSED(reg);
  Interpreter::Undefined();
  return {};
}

template <>
[[nodiscard]] Interpreter::FpRegister inline Interpreter::GetFRegAndUnboxNan<Interpreter::Float64>(
    uint8_t reg) {
  UNUSED(reg);
  Interpreter::Undefined();
  return {};
}

template <>
void inline Interpreter::NanBoxAndSetFpReg<Interpreter::Float32>(uint8_t reg, FpRegister value) {
  if (exception_raised_) {
    // Do not produce side effects.
    return;
  }
  CheckFpRegIsValid(reg);
  state_->cpu.f[reg] = NanBox<Float32>(value);
}

template <>
void inline Interpreter::NanBoxAndSetFpReg<Interpreter::Float64>(uint8_t reg, FpRegister value) {
  if (exception_raised_) {
    // Do not produce side effects.
    return;
  }
  CheckFpRegIsValid(reg);
  state_->cpu.f[reg] = value;
}

#ifdef BERBERIS_RISCV64_INTERPRETER_SEPARATE_INSTANTIATION_OF_VECTOR_OPERATIONS
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VLoadIndexedArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VLoadStrideArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VLoadUnitStrideArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VOpFVfArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VOpFVvArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VOpIViArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VOpIVvArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VOpIVxArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VOpMVvArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VOpMVxArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VStoreIndexedArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VStoreStrideArgs& args);
template <>
extern void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VStoreUnitStrideArgs& args);
#endif

}  // namespace berberis
