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

#include <cstdint>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_addr.h"

#include "lite_translator.h"
#include "register_maintainer.h"

namespace berberis {

namespace {

TEST(Riscv64LiteTranslatorTest, GetMappedRegs) {
  MachineCode machine_code;
  LiteTranslator translator(&machine_code, 0);
  auto [mapped_reg, is_new] = translator.GetMappedRegisterOrMap(1);
  EXPECT_TRUE(is_new);
  auto [mapped_reg2, is_new2] = translator.GetMappedRegisterOrMap(1);
  EXPECT_FALSE(is_new2);
  EXPECT_EQ(mapped_reg, mapped_reg2);
  auto [mapped_reg3, is_new3] = translator.GetMappedRegisterOrMap(2);
  EXPECT_TRUE(is_new3);
  EXPECT_NE(mapped_reg, mapped_reg3);
}

TEST(Riscv64LiteTranslatorTest, GetRegs) {
  // if size of machine_code does not change load is skipped.
  MachineCode machine_code;
  LiteTranslator translator(&machine_code, 0);
  translator.GetReg(1);
  int size1 = machine_code.install_size();
  translator.GetReg(1);
  int size2 = machine_code.install_size();
  EXPECT_EQ(size1, size2);
  translator.GetReg(2);
  int size3 = machine_code.install_size();
  EXPECT_LT(size2, size3);
  translator.GetReg(3);
  int size4 = machine_code.install_size();
  EXPECT_LT(size3, size4);
}

}  // namespace

}  // namespace berberis
