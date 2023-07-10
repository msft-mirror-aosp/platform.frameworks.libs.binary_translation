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

#ifndef BERBERIS_DECODER_RISCV64_SEMANTICS_PLAYER_H_
#define BERBERIS_DECODER_RISCV64_SEMANTICS_PLAYER_H_

#include "berberis/base/overloaded.h"
#include "berberis/decoder/riscv64/decoder.h"

namespace berberis {

// This class expresses the semantics of instructions by calling a sequence of SemanticsListener
// callbacks.
template <class SemanticsListener>
class SemanticsPlayer {
 public:
  using Decoder = Decoder<SemanticsPlayer>;
  using Register = typename SemanticsListener::Register;
  using Float32 = typename SemanticsListener::Float32;
  using Float64 = typename SemanticsListener::Float64;
  using FpRegister = typename SemanticsListener::FpRegister;

  explicit SemanticsPlayer(SemanticsListener* listener) : listener_(listener) {}

  // Decoder's InsnConsumer implementation.

  void Amo(const typename Decoder::AmoArgs& args) {
    Register arg1 = GetRegOrZero(args.src1);
    Register arg2 = GetRegOrZero(args.src2);
    Register result = listener_->Amo(args.opcode, arg1, arg2, args.aq, args.rl);
    SetRegOrIgnore(args.dst, result);
  };

  void Auipc(const typename Decoder::UpperImmArgs& args) {
    Register result = listener_->Auipc(args.imm);
    SetRegOrIgnore(args.dst, result);
  }

  void CompareAndBranch(const typename Decoder::BranchArgs& args) {
    Register arg1 = GetRegOrZero(args.src1);
    Register arg2 = GetRegOrZero(args.src2);
    listener_->CompareAndBranch(args.opcode, arg1, arg2, args.offset);
  };

  void Csr(const typename Decoder::CsrArgs& args) {
    Register result;
    Register arg = GetRegOrZero(args.src);
    result = listener_->Csr(args.opcode, arg, args.csr);
    SetRegOrIgnore(args.dst, result);
  }

  void Csr(const typename Decoder::CsrImmArgs& args) {
    Register result;
    result = listener_->Csr(args.opcode, args.imm, args.csr);
    SetRegOrIgnore(args.dst, result);
  }

  void Fcvt(const typename Decoder::FcvtFloatToFloatArgs& args) {
    FpRegister arg = GetFRegAndUnboxNan(args.src, args.src_type);
    Register frm = listener_->GetFrm();
    FpRegister result;
    if (args.dst_type == Decoder::FloatOperandType::kFloat &&
        args.src_type == Decoder::FloatOperandType::kDouble) {
      result = listener_->template FCvtFloatToFloat<Float32, Float64>(args.rm, frm, arg);
    } else if (args.dst_type == Decoder::FloatOperandType::kDouble &&
               args.src_type == Decoder::FloatOperandType::kFloat) {
      result = listener_->template FCvtFloatToFloat<Float64, Float32>(args.rm, frm, arg);
    } else {
      Unimplemented();
      return;
    }
    NanBoxAndSetFpReg(args.dst, result, args.dst_type);
  }

  void Fcvt(const typename Decoder::FcvtFloatToIntegerArgs& args) {
    FpRegister arg = GetFRegAndUnboxNan(args.src, args.src_type);
    Register frm = listener_->GetFrm();
    Register result;
    switch (args.src_type) {
      case Decoder::FloatOperandType::kFloat:
        switch (args.dst_type) {
          case Decoder::FcvtOperandType::k32bitSigned:
            result = listener_->template FCvtFloatToInteger<int32_t, Float32>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k32bitUnsigned:
            result = listener_->template FCvtFloatToInteger<uint32_t, Float32>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k64bitSigned:
            result = listener_->template FCvtFloatToInteger<int64_t, Float32>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k64bitUnsigned:
            result = listener_->template FCvtFloatToInteger<uint64_t, Float32>(args.rm, frm, arg);
            break;
          default:
            Unimplemented();
            return;
        }
        break;
      case Decoder::FloatOperandType::kDouble:
        switch (args.dst_type) {
          case Decoder::FcvtOperandType::k32bitSigned:
            result = listener_->template FCvtFloatToInteger<int32_t, Float64>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k32bitUnsigned:
            result = listener_->template FCvtFloatToInteger<uint32_t, Float64>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k64bitSigned:
            result = listener_->template FCvtFloatToInteger<int64_t, Float64>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k64bitUnsigned:
            result = listener_->template FCvtFloatToInteger<uint64_t, Float64>(args.rm, frm, arg);
            break;
          default:
            Unimplemented();
            return;
        }
        break;
      default:
        Unimplemented();
        return;
    }
    SetRegOrIgnore(args.dst, result);
  }

  void Fcvt(const typename Decoder::FcvtIntegerToFloatArgs& args) {
    Register arg = GetRegOrZero(args.src);
    Register frm = listener_->GetFrm();
    FpRegister result;
    switch (args.dst_type) {
      case Decoder::FloatOperandType::kFloat:
        switch (args.src_type) {
          case Decoder::FcvtOperandType::k32bitSigned:
            result = listener_->template FCvtIntegerToFloat<Float32, int32_t>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k32bitUnsigned:
            result = listener_->template FCvtIntegerToFloat<Float32, uint32_t>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k64bitSigned:
            result = listener_->template FCvtIntegerToFloat<Float32, int64_t>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k64bitUnsigned:
            result = listener_->template FCvtIntegerToFloat<Float32, uint64_t>(args.rm, frm, arg);
            break;
          default:
            Unimplemented();
            return;
        }
        break;
      case Decoder::FloatOperandType::kDouble:
        switch (args.src_type) {
          case Decoder::FcvtOperandType::k32bitSigned:
            result = listener_->template FCvtIntegerToFloat<Float64, int32_t>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k32bitUnsigned:
            result = listener_->template FCvtIntegerToFloat<Float64, uint32_t>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k64bitSigned:
            result = listener_->template FCvtIntegerToFloat<Float64, int64_t>(args.rm, frm, arg);
            break;
          case Decoder::FcvtOperandType::k64bitUnsigned:
            result = listener_->template FCvtIntegerToFloat<Float64, uint64_t>(args.rm, frm, arg);
            break;
          default:
            Unimplemented();
            return;
        }
        break;
      default:
        Unimplemented();
        return;
    }
    NanBoxAndSetFpReg(args.dst, result, args.dst_type);
  }

  void Fma(const typename Decoder::FmaArgs& args) {
    FpRegister arg1 = GetFRegAndUnboxNan(args.src1, args.operand_type);
    FpRegister arg2 = GetFRegAndUnboxNan(args.src2, args.operand_type);
    FpRegister arg3 = GetFRegAndUnboxNan(args.src3, args.operand_type);
    FpRegister result;
    Register frm = listener_->GetFrm();
    switch (args.operand_type) {
      case Decoder::FloatOperandType::kFloat:
        result = Fma<Float32>(args.opcode, args.rm, frm, arg1, arg2, arg3);
        break;
      case Decoder::FloatOperandType::kDouble:
        result = Fma<Float64>(args.opcode, args.rm, frm, arg1, arg2, arg3);
        break;
      default:
        Unimplemented();
        return;
    }
    result = CanonicalizeNan(result, args.operand_type);
    NanBoxAndSetFpReg(args.dst, result, args.operand_type);
  }

  template <typename FloatType>
  FpRegister Fma(typename Decoder::FmaOpcode opcode,
                 int8_t rm,
                 Register frm,
                 FpRegister arg1,
                 FpRegister arg2,
                 FpRegister arg3) {
    switch (opcode) {
      case Decoder::FmaOpcode::kFmadd:
        return listener_->template FMAdd<FloatType>(rm, frm, arg1, arg2, arg3);
      case Decoder::FmaOpcode::kFmsub:
        return listener_->template FMSub<FloatType>(rm, frm, arg1, arg2, arg3);
      // Note (from RISC-V manual): The FNMSUB and FNMADD instructions are counterintuitively named,
      // owing to the naming of the corresponding instructions in MIPS-IV. The MIPS instructions
      // were defined to negate the sum, rather than negating the product as the RISC-V instructions
      // do, so the naming scheme was more rational at the time. The two definitions differ with
      // respect to signed-zero results. The RISC-V definition matches the behavior of the x86 and
      // ARM fused multiply-add instructions, but unfortunately the RISC-V FNMSUB and FNMADD
      // instruction names are swapped compared to x86 and ARM.
      //
      // Since even official documentation calls the names “counterintuitive” it's better to use x86
      // ones for intrinsics.
      case Decoder::FmaOpcode::kFnmsub:
        return listener_->template FNMAdd<FloatType>(rm, frm, arg1, arg2, arg3);
      case Decoder::FmaOpcode::kFnmadd:
        return listener_->template FNMSub<FloatType>(rm, frm, arg1, arg2, arg3);
      default:
        Unimplemented();
        return {};
    }
  }

  void Fence(const typename Decoder::FenceArgs& args) {
    listener_->Fence(args.opcode,
                     // args.src is currently unused - read below.
                     Register{},
                     args.sw,
                     args.sr,
                     args.so,
                     args.si,
                     args.pw,
                     args.pr,
                     args.po,
                     args.pi);
    // The unused fields in the FENCE instructions — args.src and args.dst — are reserved for
    // finer-grain fences in future extensions. For forward compatibility, base implementations
    // shall ignore these fields, and standard software shall zero these fields. Likewise, many
    // args.opcode and predecessor/successor set settings are also reserved for future use. Base
    // implementations shall treat all such reserved configurations as normal fences with
    // args.opcode=0000, and standard software shall use only non-reserved configurations.
  }

  void FenceI(const typename Decoder::FenceIArgs& args) {
    Register arg = GetRegOrZero(args.src);
    listener_->FenceI(arg, args.imm);
    // The unused fields in the FENCE.I instruction, imm[11:0], rs1, and rd, are reserved for
    // finer-grain fences in future extensions. For forward compatibility, base implementations
    // shall ignore these fields, and standard software shall zero these fields.
  }

  void JumpAndLink(const typename Decoder::JumpAndLinkArgs& args) {
    Register result = listener_->GetImm(listener_->GetInsnAddr() + args.insn_len);
    SetRegOrIgnore(args.dst, result);
    listener_->Branch(args.offset);
  };

  void JumpAndLinkRegister(const typename Decoder::JumpAndLinkRegisterArgs& args) {
    Register result = listener_->GetImm(listener_->GetInsnAddr() + args.insn_len);
    SetRegOrIgnore(args.dst, result);
    Register base = GetRegOrZero(args.base);
    listener_->BranchRegister(base, args.offset);
  };

  void Load(const typename Decoder::LoadArgs& args) {
    Register arg = GetRegOrZero(args.src);
    Register result = listener_->Load(args.operand_type, arg, args.offset);
    SetRegOrIgnore(args.dst, result);
  };

  void Load(const typename Decoder::LoadFpArgs& args) {
    Register arg = GetRegOrZero(args.src);
    FpRegister result;
    switch (args.operand_type) {
      case Decoder::FloatOperandType::kFloat:
        result = listener_->template LoadFp<Float32>(arg, args.offset);
        break;
      case Decoder::FloatOperandType::kDouble:
        result = listener_->template LoadFp<Float64>(arg, args.offset);
        break;
      default:
        Unimplemented();
        return;
    }
    NanBoxAndSetFpReg(args.dst, result, args.operand_type);
  };

  void Lui(const typename Decoder::UpperImmArgs& args) {
    Register result = listener_->Lui(args.imm);
    SetRegOrIgnore(args.dst, result);
  }

  void Nop() { listener_->Nop(); }

  template <typename OpArgs>
  void Op(OpArgs&& args) {
    Register arg1 = GetRegOrZero(args.src1);
    Register arg2 = GetRegOrZero(args.src2);
    Register result = Overloaded{[&](const typename Decoder::OpArgs& args) {
                                   return listener_->Op(args.opcode, arg1, arg2);
                                 },
                                 [&](const typename Decoder::Op32Args& args) {
                                   return listener_->Op32(args.opcode, arg1, arg2);
                                 }}(args);
    SetRegOrIgnore(args.dst, result);
  };

  void OpFp(const typename Decoder::OpFpArgs& args) {
    FpRegister arg1 = GetFRegAndUnboxNan(args.src1, args.operand_type);
    FpRegister arg2 = GetFRegAndUnboxNan(args.src2, args.operand_type);
    FpRegister result;
    Register frm = listener_->GetFrm();
    switch (args.operand_type) {
      case Decoder::FloatOperandType::kFloat:
        result = OpFp<Float32>(args.opcode, args.rm, frm, arg1, arg2);
        break;
      case Decoder::FloatOperandType::kDouble:
        result = OpFp<Float64>(args.opcode, args.rm, frm, arg1, arg2);
        break;
      default:
        Unimplemented();
        return;
    }
    result = CanonicalizeNan(result, args.operand_type);
    NanBoxAndSetFpReg(args.dst, result, args.operand_type);
  }

  template <typename FloatType>
  FpRegister OpFp(typename Decoder::OpFpOpcode opcode,
                  int8_t rm,
                  Register frm,
                  FpRegister arg1,
                  FpRegister arg2) {
    switch (opcode) {
      case Decoder::OpFpOpcode::kFAdd:
        return listener_->template FAdd<FloatType>(rm, frm, arg1, arg2);
      case Decoder::OpFpOpcode::kFSub:
        return listener_->template FSub<FloatType>(rm, frm, arg1, arg2);
      case Decoder::OpFpOpcode::kFMul:
        return listener_->template FMul<FloatType>(rm, frm, arg1, arg2);
      case Decoder::OpFpOpcode::kFDiv:
        return listener_->template FDiv<FloatType>(rm, frm, arg1, arg2);
      default:
        Unimplemented();
        return {};
    }
  }

  void OpFpGpRegisterTargetNoRounding(
      const typename Decoder::OpFpGpRegisterTargetNoRoundingArgs& args) {
    FpRegister arg1 = GetFRegAndUnboxNan(args.src1, args.operand_type);
    FpRegister arg2 = GetFRegAndUnboxNan(args.src2, args.operand_type);
    Register result;
    switch (args.operand_type) {
      case Decoder::FloatOperandType::kFloat:
        result = OpFpGpRegisterTargetNoRounding<Float32>(args.opcode, arg1, arg2);
        break;
      case Decoder::FloatOperandType::kDouble:
        result = OpFpGpRegisterTargetNoRounding<Float64>(args.opcode, arg1, arg2);
        break;
      default:
        Unimplemented();
        return;
    }
    SetRegOrIgnore(args.dst, result);
  }

  template <typename FloatType>
  Register OpFpGpRegisterTargetNoRounding(
      typename Decoder::OpFpGpRegisterTargetNoRoundingOpcode opcode,
      FpRegister arg1,
      FpRegister arg2) {
    switch (opcode) {
      case Decoder::OpFpGpRegisterTargetNoRoundingOpcode::kFle:
        return listener_->template Fle<FloatType>(arg1, arg2);
      case Decoder::OpFpGpRegisterTargetNoRoundingOpcode::kFlt:
        return listener_->template Flt<FloatType>(arg1, arg2);
      case Decoder::OpFpGpRegisterTargetNoRoundingOpcode::kFeq:
        return listener_->template Feq<FloatType>(arg1, arg2);
      default:
        Unimplemented();
        return {};
    }
  }

  void OpFpGpRegisterTargetSingleInputNoRounding(
      const typename Decoder::OpFpGpRegisterTargetSingleInputNoRoundingArgs& args) {
    FpRegister arg = GetFRegAndUnboxNan(args.src, args.operand_type);
    Register result;
    switch (args.operand_type) {
      case Decoder::FloatOperandType::kFloat:
        result = OpFpGpRegisterTargetSingleInputNoRounding<Float32>(args.opcode, arg);
        break;
      case Decoder::FloatOperandType::kDouble:
        result = OpFpGpRegisterTargetSingleInputNoRounding<Float64>(args.opcode, arg);
        break;
      default:
        Unimplemented();
        return;
    }
    SetRegOrIgnore(args.dst, result);
  }

  template <typename FloatType>
  Register OpFpGpRegisterTargetSingleInputNoRounding(
      typename Decoder::OpFpGpRegisterTargetSingleInputNoRoundingOpcode opcode,
      FpRegister arg) {
    switch (opcode) {
      case Decoder::OpFpGpRegisterTargetSingleInputNoRoundingOpcode::kFclass:
        return listener_->template FClass<FloatType>(arg);
      default:
        Unimplemented();
        return {};
    }
  }

  void OpFpNoRounding(const typename Decoder::OpFpNoRoundingArgs& args) {
    FpRegister arg1 = GetFRegAndUnboxNan(args.src1, args.operand_type);
    FpRegister arg2 = GetFRegAndUnboxNan(args.src2, args.operand_type);
    FpRegister result;
    switch (args.operand_type) {
      case Decoder::FloatOperandType::kFloat:
        result = OpFpNoRounding<Float32>(args.opcode, arg1, arg2);
        break;
      case Decoder::FloatOperandType::kDouble:
        result = OpFpNoRounding<Float64>(args.opcode, arg1, arg2);
        break;
      default:
        Unimplemented();
        return;
    }
    result = CanonicalizeNan(result, args.operand_type);
    NanBoxAndSetFpReg(args.dst, result, args.operand_type);
  }

  template <typename FloatType>
  FpRegister OpFpNoRounding(const typename Decoder::OpFpNoRoundingOpcode opcode,
                            FpRegister arg1,
                            FpRegister arg2) {
    switch (opcode) {
      case Decoder::OpFpNoRoundingOpcode::kFSgnj:
        return listener_->template FSgnj<FloatType>(arg1, arg2);
      case Decoder::OpFpNoRoundingOpcode::kFSgnjn:
        return listener_->template FSgnjn<FloatType>(arg1, arg2);
      case Decoder::OpFpNoRoundingOpcode::kFSgnjx:
        return listener_->template FSgnjx<FloatType>(arg1, arg2);
      case Decoder::OpFpNoRoundingOpcode::kFMin:
        return listener_->template FMin<FloatType>(arg1, arg2);
      case Decoder::OpFpNoRoundingOpcode::kFMax:
        return listener_->template FMax<FloatType>(arg1, arg2);
      default:
        Unimplemented();
        return {};
    }
  }

  void FmvFloatToInteger(const typename Decoder::FmvFloatToIntegerArgs& args) {
    FpRegister arg = GetFpReg(args.src);
    Register result;
    switch (args.operand_type) {
      case Decoder::FloatOperandType::kFloat:
        result = listener_->template FmvFloatToInteger<int32_t, Float32>(arg);
        break;
      case Decoder::FloatOperandType::kDouble:
        result = listener_->template FmvFloatToInteger<int64_t, Float64>(arg);
        break;
      default:
        Unimplemented();
        return;
    }
    SetRegOrIgnore(args.dst, result);
  }

  void FmvIntegerToFloat(const typename Decoder::FmvIntegerToFloatArgs& args) {
    Register arg = GetRegOrZero(args.src);
    FpRegister result;
    switch (args.operand_type) {
      case Decoder::FloatOperandType::kFloat:
        result = listener_->template FmvIntegerToFloat<Float32, int32_t>(arg);
        break;
      case Decoder::FloatOperandType::kDouble:
        result = listener_->template FmvIntegerToFloat<Float64, int64_t>(arg);
        break;
      default:
        Unimplemented();
        return;
    }
    NanBoxAndSetFpReg(args.dst, result, args.operand_type);
  }

  void OpFpSingleInput(const typename Decoder::OpFpSingleInputArgs& args) {
    FpRegister arg = GetFRegAndUnboxNan(args.src, args.operand_type);
    FpRegister result;
    Register frm = listener_->GetFrm();
    switch (args.operand_type) {
      case Decoder::FloatOperandType::kFloat:
        result = OpFpSingleInput<Float32>(args.opcode, args.rm, frm, arg);
        break;
      case Decoder::FloatOperandType::kDouble:
        result = OpFpSingleInput<Float64>(args.opcode, args.rm, frm, arg);
        break;
      default:
        Unimplemented();
        return;
    }
    result = CanonicalizeNan(result, args.operand_type);
    NanBoxAndSetFpReg(args.dst, result, args.operand_type);
  }

  template <typename FloatType>
  FpRegister OpFpSingleInput(typename Decoder::OpFpSingleInputOpcode opcode,
                             int8_t rm,
                             Register frm,
                             FpRegister arg) {
    switch (opcode) {
      case Decoder::OpFpSingleInputOpcode::kFSqrt:
        return listener_->template FSqrt<FloatType>(rm, frm, arg);
      default:
        Unimplemented();
        return {};
    }
  }

  template <typename OpImmArgs>
  void OpImm(OpImmArgs&& args) {
    Register arg = GetRegOrZero(args.src);
    Register result = Overloaded{[&](const typename Decoder::OpImmArgs& args) {
                                   return listener_->OpImm(args.opcode, arg, args.imm);
                                 },
                                 [&](const typename Decoder::OpImm32Args& args) {
                                   return listener_->OpImm32(args.opcode, arg, args.imm);
                                 },
                                 [&](const typename Decoder::ShiftImmArgs& args) {
                                   return listener_->ShiftImm(args.opcode, arg, args.imm);
                                 },
                                 [&](const typename Decoder::ShiftImm32Args& args) {
                                   return listener_->ShiftImm32(args.opcode, arg, args.imm);
                                 }}(args);
    SetRegOrIgnore(args.dst, result);
  };

  void Store(const typename Decoder::StoreArgs& args) {
    Register arg = GetRegOrZero(args.src);
    Register data = GetRegOrZero(args.data);
    listener_->Store(args.operand_type, arg, args.offset, data);
  };

  void Store(const typename Decoder::StoreFpArgs& args) {
    Register arg = GetRegOrZero(args.src);
    FpRegister data = GetFpReg(args.data);
    switch (args.operand_type) {
      case Decoder::FloatOperandType::kFloat:
        listener_->template StoreFp<Float32>(arg, args.offset, data);
        break;
      case Decoder::FloatOperandType::kDouble:
        listener_->template StoreFp<Float64>(arg, args.offset, data);
        break;
      default:
        Unimplemented();
        return;
    }
  };

  // We may have executed a signal handler just after the syscall. If that handler changed x10, then
  // overwriting x10 here would be incorrect. On the other hand asynchronous signals are unlikely to
  // change CPU state, so we don't support this at the moment for simplicity."
  void System(const typename Decoder::SystemArgs& args) {
    if (args.opcode != Decoder::SystemOpcode::kEcall) {
      return Unimplemented();
    }
    Register syscall_nr = GetRegOrZero(17);
    Register arg0 = GetRegOrZero(10);
    Register arg1 = GetRegOrZero(11);
    Register arg2 = GetRegOrZero(12);
    Register arg3 = GetRegOrZero(13);
    Register arg4 = GetRegOrZero(14);
    Register arg5 = GetRegOrZero(15);
    Register result = listener_->Ecall(syscall_nr, arg0, arg1, arg2, arg3, arg4, arg5);
    SetRegOrIgnore(10, result);
  }

  void Unimplemented() { listener_->Unimplemented(); };

 private:
  Register GetRegOrZero(uint8_t reg) {
    return reg == 0 ? listener_->GetImm(0) : listener_->GetReg(reg);
  }

  void SetRegOrIgnore(uint8_t reg, Register value) {
    if (reg != 0) {
      listener_->SetReg(reg, value);
    }
  }

  // Floating point instructions in RISC-V are encoded in a way where you may find out size of
  // operand (single-precision, double-precision, half-precision or quad-precesion; the latter
  // two optional) from the instruction encoding without determining the full form of instruction.
  //
  // Sources and targets are also specified via dedicated bits in opcodes.
  //
  // This allows us to split instruction handling in four steps:
  //   1. Load operands from register and convert it into a form suitable for host.
  //   2. Execute operations specified by opcode.
  //   3. Normalize NaNs if host and guest architctures handled them differently.
  //   4. Encode results as required by RISC-V (if host doesn't do that).
  //
  // Note that in case of execution of RISC-V on RISC-V all steps except #2 are not doing anything.

  // Step #1:
  //  • GetFpReg — for instructions like fsw or fmv.x.w use GetFpReg which doesn't change value.
  //  • GetFRegAndBoxNan — for most instructions (improperly boxed narrow float is turned into NaN).
  FpRegister GetFpReg(uint8_t reg) { return listener_->GetFpReg(reg); }
  FpRegister GetFRegAndUnboxNan(uint8_t reg, typename Decoder::FloatOperandType operand_type) {
    return listener_->GetFRegAndUnboxNan(reg, operand_type);
  }

  // Step #3.
  FpRegister CanonicalizeNan(FpRegister value, typename Decoder::FloatOperandType operand_type) {
    return listener_->CanonicalizeNan(value, operand_type);
  }

  // Step #4. Note the assymetry: step #1 may skip the NaN unboxing (would use GetFpReg if so),
  // but step #4 boxes uncoditionally (if actual instruction doesn't do that on host).
  void NanBoxAndSetFpReg(uint8_t reg,
                         FpRegister value,
                         typename Decoder::FloatOperandType operand_type) {
    listener_->NanBoxAndSetFpReg(reg, value, operand_type);
  }

  SemanticsListener* listener_;
};

}  // namespace berberis

#endif  // BERBERIS_DECODER_RISCV64_SEMANTICS_PLAYER_H_
