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

#ifndef BERBERIS_DECODER_RISCV64_DECODER_H_
#define BERBERIS_DECODER_RISCV64_DECODER_H_

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace berberis {

// Decode() method takes a sequence of bytes and decodes it into the instruction opcode and fields.
// The InsnConsumer's method corresponding to the decoded opcode is called with the decoded fields
// as an argument. Returned is the instruction size.
template <class InsnConsumer>
class Decoder {
 public:
  explicit Decoder(InsnConsumer* insn_consumer) : insn_consumer_(insn_consumer) {}

  enum class BaseOpcode {
    kLoad = 0b00'000,
    kLoadFp = 0b00'001,
    kCustom0 = 0b00'010,
    kMiscMem = 0b00'011,
    kOpImm = 0b00'100,
    kAuipc = 0b00'101,
    kOpImm32 = 0b00'110,
    // Reserved 0b00'111,
    kStore = 0b01'000,
    kStoreFp = 0b01'001,
    kCustom1 = 0b01'010,
    kAmo = 0b01'011,
    kOp = 0b01'100,
    kLui = 0b01'101,
    kOp32 = 0b01'110,
    // Reserved 0b01'111,
    kMAdd = 0b10'000,
    kMSub = 0b10'001,
    kNmSub = 0b10'010,
    kNmAdd = 0b10'011,
    kOpFp = 0b10'100,
    // Reserved 0b10'101,
    kCustom2 = 0b10'110,
    // Reserved 0b10'111,
    kBranch = 0b11'000,
    kJalr = 0b11'001,
    // Reserved 0b11'010,
    kJal = 0b11'011,
    kSystem = 0b11'100,
    // Reserved 0b11'101,
    kCustom3 = 0b11'110,
    // Reserved 0b11'111,
  };

  enum class OpOpcode {
    kAdd = 0b0000'000'000,
    kSub = 0b0100'000'000,
    kSll = 0b0000'000'001,
    kSlt = 0b0000'000'010,
    kSltu = 0b0000'000'011,
    kXor = 0b0000'000'100,
    kSlr = 0b0000'000'101,
    kSra = 0b0100'000'101,
    kOr = 0b0000'000'110,
    kAnd = 0b0000'000'111,
  };

  struct OpArgs {
    OpOpcode opcode;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
  };

  uint8_t Decode(const uint16_t* code) {
    constexpr uint16_t kInsnLenMask = uint16_t{0b11};
    if ((*code & kInsnLenMask) != kInsnLenMask)  {
      // TODO(b/265372622): Support 16-bit instructions.
      insn_consumer_->Unimplemented();
      return 2;
    }

    // Warning: do not cast and dereference the pointer
    // since the address may not be 4-bytes aligned.
    memcpy(&code_, code, sizeof(code_));

    BaseOpcode opcode_bits = BaseOpcode{GetBits<uint8_t, 2, 5>()};

    switch (opcode_bits) {
      case BaseOpcode::kOp:
        DecodeOp();
        break;
      default:
        insn_consumer_->Unimplemented();
    }
    return 4;
  }

 private:
  template <typename ResultType, uint32_t start, uint32_t size>
  ResultType GetBits() {
    static_assert(std::is_unsigned_v<ResultType>, "Only unsigned types are supported");
    static_assert(sizeof(ResultType) * CHAR_BIT >= size, "Too small ResultType for size");
    static_assert((start + size) <= 32 && size > 0, "Invalid start or size value");
    uint32_t shifted_val = code_ << (32 - start - size);
    return static_cast<ResultType>(shifted_val >> (32 - size));
  }

  void DecodeOp() {
    uint16_t low_opcode = GetBits<uint16_t, 12, 3>();
    uint16_t high_opcode = GetBits<uint16_t, 25, 7>();
    OpOpcode opcode = OpOpcode{low_opcode | (high_opcode << 3)};
    const OpArgs args = {
        .opcode = opcode,
        .dst = GetBits<uint8_t, 7, 5>(),
        .src1 = GetBits<uint8_t, 15, 5>(),
        .src2 = GetBits<uint8_t, 20, 5>(),
    };
    insn_consumer_->Op(args);
  }

  InsnConsumer* insn_consumer_;
  uint32_t code_;
};

}  // namespace berberis

#endif  // BERBERIS_DECODER_RISCV64_DECODER_H_
