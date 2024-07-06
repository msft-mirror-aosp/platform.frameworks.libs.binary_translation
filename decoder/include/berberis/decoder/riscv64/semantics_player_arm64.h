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

#ifndef BERBERIS_DECODER_RISCV64_SEMANTICS_PLAYER_ARM64_H_
#define BERBERIS_DECODER_RISCV64_SEMANTICS_PLAYER_ARM64_H_

#include "berberis/decoder/riscv64/decoder.h"

namespace berberis {

// TODO(b/346805222): This file is a temperate file and will be removed once intrinsics part is
// finished for Arm64.
// This class expresses the semantics of instructions by calling a sequence of SemanticsListener
// callbacks.
template <class SemanticsListener>
class SemanticsPlayer {
 public:
  using Decoder = Decoder<SemanticsPlayer>;
  using Register = typename SemanticsListener::Register;

  explicit SemanticsPlayer(SemanticsListener* listener) : listener_(listener) {}

  // Decoder's InsnConsumer implementation.
  void Op(const typename Decoder::OpArgs& args) {
    Register arg1 = GetRegOrZero(args.src1);
    Register arg2 = GetRegOrZero(args.src2);
    Register result = listener_->Op(args.opcode, arg1, arg2);
    SetRegOrIgnore(args.dst, result);
  };

  void Unimplemented() { listener_->Unimplemented(); };

  void Undefined() { listener_->Undefined(); };

  template <typename OpImmArgs>
  void OpImm([[maybe_unused]] OpImmArgs&& args) {
    Undefined();
  };
  void Load([[maybe_unused]] const typename Decoder::LoadArgs& args) { Undefined(); };
  void Load(const typename Decoder::LoadFpArgs& args) {
    switch (args.operand_type) {
      default:
        return Undefined();
    }
  }

  template <typename FloatType>
  void Load([[maybe_unused]] int8_t dst,
            [[maybe_unused]] int8_t src,
            [[maybe_unused]] int16_t offset) {
    Undefined();
  };

  void Store([[maybe_unused]] const typename Decoder::StoreArgs& args) { Undefined(); };
  void Store([[maybe_unused]] const typename Decoder::StoreFpArgs& args) { Undefined(); };
  void Nop() { Undefined(); }
  void Lui([[maybe_unused]] const typename Decoder::UpperImmArgs& args) { Undefined(); }
  template <typename OpArgs>
  void Op([[maybe_unused]] OpArgs&& args) {
    Undefined();
  }
  void JumpAndLink([[maybe_unused]] const typename Decoder::JumpAndLinkArgs& args) { Undefined(); }
  void CompareAndBranch([[maybe_unused]] const typename Decoder::BranchArgs& args) { Undefined(); }
  void System([[maybe_unused]] const typename Decoder::SystemArgs& args) { Undefined(); }
  template <typename OpVectorArgs>
  void OpVector([[maybe_unused]] OpVectorArgs& args) {
    Undefined();
  }
  void FenceI(const typename Decoder::FenceIArgs& /* args */) {
    // This instruction is not supported on linux. The recommendation is to use the
    // riscv_flush_icache syscall instead.
    Undefined();
    // The unused fields in the FENCE.I instruction, imm[11:0], rs1, and rd, are reserved for
    // finer-grain fences in future extensions. For forward compatibility, base implementations
    // shall ignore these fields, and standard software shall zero these fields.
  }
  void Auipc([[maybe_unused]] const typename Decoder::UpperImmArgs& args) { Undefined(); }
  void OpSingleInput([[maybe_unused]] const typename Decoder::OpSingleInputArgs& args) {
    Undefined();
  }
  void Fma([[maybe_unused]] const typename Decoder::FmaArgs& args) { Undefined(); }
  void OpFp([[maybe_unused]] const typename Decoder::OpFpArgs& args) { Undefined(); }
  void OpFpSingleInputNoRounding(
      [[maybe_unused]] const typename Decoder::OpFpSingleInputNoRoundingArgs& args) {
    Undefined();
  }
  void JumpAndLinkRegister([[maybe_unused]] const typename Decoder::JumpAndLinkRegisterArgs& args) {
    Undefined();
  }
  void Fence([[maybe_unused]] const typename Decoder::FenceArgs& args) { Undefined(); }
  void Amo([[maybe_unused]] const typename Decoder::AmoArgs& args) { Undefined(); }
  void OpFpNoRounding([[maybe_unused]] const typename Decoder::OpFpNoRoundingArgs& args) {
    Undefined();
  }
  void OpFpSingleInput([[maybe_unused]] const typename Decoder::OpFpSingleInputArgs& args) {
    Undefined();
  }
  void OpFpGpRegisterTargetNoRounding(
      [[maybe_unused]] const typename Decoder::OpFpGpRegisterTargetNoRoundingArgs& args) {
    Undefined();
  }
  template <typename FloatType>
  void OpFpGpRegisterTargetNoRounding([[maybe_unused]]
                                      typename Decoder::OpFpGpRegisterTargetNoRoundingOpcode opcode,
                                      [[maybe_unused]] int8_t dst,
                                      [[maybe_unused]] int8_t src1,
                                      [[maybe_unused]] int8_t src2) {
    Undefined();
  }
  void Fcvt([[maybe_unused]] const typename Decoder::FcvtFloatToFloatArgs& args) { Undefined(); }
  void Fcvt([[maybe_unused]] const typename Decoder::FcvtFloatToIntegerArgs& args) { Undefined(); }
  void Fcvt([[maybe_unused]] const typename Decoder::FcvtIntegerToFloatArgs& args) { Undefined(); }
  void OpFpGpRegisterTargetSingleInputNoRounding(
      [[maybe_unused]] const typename Decoder::OpFpGpRegisterTargetSingleInputNoRoundingArgs&
          args) {
    Undefined();
  }
  template <typename FloatType>
  void OpFpGpRegisterTargetSingleInputNoRounding(
      [[maybe_unused]] typename Decoder::OpFpGpRegisterTargetSingleInputNoRoundingOpcode opcode,
      [[maybe_unused]] int8_t dst,
      [[maybe_unused]] int8_t src) {
    Undefined();
  }
  void FmvFloatToInteger([[maybe_unused]] const typename Decoder::FmvFloatToIntegerArgs& args) {
    Undefined();
  }
  void Vsetvli([[maybe_unused]] const typename Decoder::VsetvliArgs& args) { Undefined(); }
  void Csr([[maybe_unused]] const typename Decoder::CsrArgs& args) { Undefined(); }
  void Csr([[maybe_unused]] const typename Decoder::CsrImmArgs& args) { Undefined(); }
  void FmvIntegerToFloat([[maybe_unused]] const typename Decoder::FmvIntegerToFloatArgs& args) {
    Undefined();
  }
  void Vsetivli([[maybe_unused]] const typename Decoder::VsetivliArgs& args) { Undefined(); }
  void Vsetvl([[maybe_unused]] const typename Decoder::VsetvlArgs& args) { Undefined(); }

 private:
  Register GetRegOrZero(uint8_t reg) {
    return reg == 0 ? listener_->GetImm(0) : listener_->GetReg(reg);
  }

  void SetRegOrIgnore(uint8_t reg, Register value) {
    if (reg != 0) {
      listener_->SetReg(reg, value);
    }
  }

  SemanticsListener* listener_;
};

}  // namespace berberis

#endif  // BERBERIS_DECODER_RISCV64_SEMANTICS_PLAYER_ARM64_H_