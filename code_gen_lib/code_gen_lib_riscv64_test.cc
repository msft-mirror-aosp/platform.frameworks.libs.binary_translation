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

#if defined(__i386__)
#include "berberis/assembler/x86_32.h"
#elif defined(__x86_64__)
#include "berberis/assembler/x86_64.h"
#endif

#include "berberis/assembler/machine_code.h"
#include "berberis/base/bit_util.h"
#include "berberis/code_gen_lib/gen_adaptor.h"
#include "berberis/code_gen_lib/gen_wrapper.h"
#include "berberis/guest_abi/guest_arguments.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/test_utils/scoped_exec_region.h"

namespace berberis {

namespace {

bool g_called;
uint32_t g_insn;

void DummyRunner2(GuestAddr pc, GuestArgumentBuffer* buf) {
  g_called = true;
  ASSERT_EQ(pc, ToGuestAddr(&g_insn));
  ASSERT_NE(nullptr, buf);
  ASSERT_EQ(1, buf->argc);
  ASSERT_EQ(0, buf->resc);
  ASSERT_EQ(1234u, buf->argv[0]);
}

TEST(CodeGenLib, GenWrapGuestFunction) {
  MachineCode machine_code;

  GenWrapGuestFunction(
      &machine_code, ToGuestAddr(&g_insn), "vi", AsHostCode(DummyRunner2), "DummyRunner2");

  ScopedExecRegion exec(&machine_code);

  g_called = false;
  exec.get<void(int)>()(1234);

  ASSERT_TRUE(g_called);
}

void Run10Int(GuestAddr pc, GuestArgumentBuffer* buf) {
  ASSERT_EQ(pc, ToGuestAddr(&g_insn));
  ASSERT_NE(nullptr, buf);
  ASSERT_EQ(8, buf->argc);
  ASSERT_EQ(16, buf->stack_argc);
  ASSERT_EQ(1, buf->resc);
  // For 32-bit parameters, only least-significant bits are defined!
  ASSERT_EQ(0u, static_cast<uint32_t>(buf->argv[0]));
  ASSERT_EQ(1u, static_cast<uint32_t>(buf->argv[1]));
  ASSERT_EQ(2u, static_cast<uint32_t>(buf->argv[2]));
  ASSERT_EQ(3u, static_cast<uint32_t>(buf->argv[3]));
  ASSERT_EQ(4u, static_cast<uint32_t>(buf->argv[4]));
  ASSERT_EQ(5u, static_cast<uint32_t>(buf->argv[5]));
  ASSERT_EQ(6u, static_cast<uint32_t>(buf->argv[6]));
  ASSERT_EQ(7u, static_cast<uint32_t>(buf->argv[7]));
  ASSERT_EQ(8u, static_cast<uint32_t>(buf->stack_argv[0]));
  ASSERT_EQ(9u, static_cast<uint32_t>(buf->stack_argv[1]));
  buf->argv[0] = 45;
}

TEST(CodeGenLib, GenWrapGuestFunction_Run10Int) {
  MachineCode machine_code;

  GenWrapGuestFunction(
      &machine_code, ToGuestAddr(&g_insn), "iiiiiiiiiii", AsHostCode(Run10Int), "Run10Int");

  ScopedExecRegion exec(&machine_code);

  typedef int Func(int, int, int, int, int, int, int, int, int, int);
  int res = exec.get<Func>()(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  ASSERT_EQ(45, res);
}

void Run10Fp(GuestAddr pc, GuestArgumentBuffer* buf) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  ASSERT_EQ(pc, ToGuestAddr(&g_insn));
  ASSERT_NE(nullptr, buf);
  // riscv verification
  ASSERT_EQ(0, buf->argc);
  ASSERT_EQ(8, buf->fp_argc);
  ASSERT_EQ(16, buf->stack_argc);
  ASSERT_EQ(0, buf->resc);
  ASSERT_EQ(1, buf->fp_resc);
  // For 32-bit parameters, only least-significant bits are defined!
  ASSERT_FLOAT_EQ(0.0f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[0])));
  ASSERT_FLOAT_EQ(1.1f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[1])));
  ASSERT_FLOAT_EQ(2.2f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[2])));
  ASSERT_FLOAT_EQ(3.3f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[3])));
  ASSERT_FLOAT_EQ(4.4f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[4])));
  ASSERT_FLOAT_EQ(5.5f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[5])));
  ASSERT_FLOAT_EQ(6.6f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[6])));
  ASSERT_FLOAT_EQ(7.7f, bit_cast<float>(static_cast<uint32_t>(buf->fp_argv[7])));
  ASSERT_FLOAT_EQ(8.8f, bit_cast<float>(static_cast<uint32_t>(buf->stack_argv[0])));
  ASSERT_FLOAT_EQ(9.9f, bit_cast<float>(static_cast<uint32_t>(buf->stack_argv[1])));
  buf->fp_argv[0] = bit_cast<uint32_t>(45.45f);
}

TEST(CodeGenLib, GenWrapGuestFunction_Run10Fp) {
  MachineCode machine_code;

  GenWrapGuestFunction(
      &machine_code, ToGuestAddr(&g_insn), "fffffffffff", AsHostCode(Run10Fp), "Run10Fp");

  ScopedExecRegion exec(&machine_code);

  typedef float Func(float, float, float, float, float, float, float, float, float, float);
  float res = exec.get<Func>()(0.0f, 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f);
  ASSERT_FLOAT_EQ(45.45f, res);
}

}  // namespace

}  // namespace berberis
