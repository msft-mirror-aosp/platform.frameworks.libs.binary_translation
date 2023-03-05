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

#include "gtest/gtest.h"

#include "berberis/decoder/riscv64/decoder.h"

#include <cstdint>

namespace berberis {

namespace {

struct TestInsnConsumer;
using Decoder = Decoder<TestInsnConsumer>;

struct TestInsnConsumer {
  void Op(const Decoder::OpArgs& args) { op_args = args; };
  void Unimplemented() { is_unimplemented = true; };
  void Load(const typename Decoder::LoadArgs&){};
  void Store(const typename Decoder::StoreArgs&){};
  void Branch(const typename Decoder::BranchArgs&){};
  void JumpAndLink(const typename Decoder::JumpAndLinkArgs&){};
  void JumpAndLinkRegister(const typename Decoder::JumpAndLinkRegisterArgs&){};

  Decoder::OpArgs op_args;
  bool is_unimplemented = false;
};

TEST(Riscv64Decoder, Add) {
  static const uint32_t code[] = {
      0x003100b3,  // add x1, x2, x3
  };

  TestInsnConsumer insn_consumer;
  Decoder decoder(&insn_consumer);

  uint8_t size = decoder.Decode(reinterpret_cast<const uint16_t*>(&code[0]));

  EXPECT_EQ(size, 4);
  EXPECT_EQ(insn_consumer.op_args.opcode, Decoder::OpOpcode::kAdd);
  EXPECT_EQ(insn_consumer.op_args.dst, 1u);
  EXPECT_EQ(insn_consumer.op_args.src1, 2u);
  EXPECT_EQ(insn_consumer.op_args.src2, 3u);
  EXPECT_FALSE(insn_consumer.is_unimplemented);
}

TEST(Riscv64Decoder, Unimplemented) {
  static const uint16_t code[] = {
      0x0,  // undefined insn
  };

  TestInsnConsumer insn_consumer;
  Decoder decoder(&insn_consumer);

  uint8_t size = decoder.Decode(code);

  EXPECT_EQ(size, 2);
  EXPECT_TRUE(insn_consumer.is_unimplemented);
}

}  // namespace

}  // namespace berberis
