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

#include <array>
#include <cstdint>

#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/host_call_frame.h"

namespace berberis {

namespace {

TEST(HostCallFrame, InitPC) {
  constexpr GuestAddr kHostCallFrameGuestPC = 0xbeefface;
  ScopedHostCallFrame::SetGuestPC(kHostCallFrameGuestPC);

  CPUState cpu{};

  alignas(uint64_t) std::array<char, 128> stack;
  SetXReg<2>(cpu, ToGuestAddr(stack.data() + stack.size()));

  ScopedHostCallFrame host_call_frame(&cpu, 0xdeadbeef);

  EXPECT_EQ(kHostCallFrameGuestPC, GetXReg<1>(cpu));

  // Pretend guest code executed up to return address.
  cpu.insn_addr = GetXReg<1>(cpu);
}

void RunHostCall(CPUState* cpu) {
  ScopedHostCallFrame host_call_frame(cpu, 0xbaaaaaad);

  // Pretend guest code executed up to return address.
  cpu->insn_addr = GetXReg<1>(*cpu);

  // Host call frame allows random adjustments of ra.
  SetXReg<1>(*cpu, 0xbaadf00d);
}

TEST(HostCallFrame, Restore) {
  CPUState cpu{};

  alignas(uint64_t) std::array<char, 128> stack;
  const GuestAddr sp = ToGuestAddr(stack.data() + stack.size());
  const GuestAddr ra = 0xdeadbeef;
  const GuestAddr fp = 0xdeadc0de;

  SetXReg<1>(cpu, ra);
  SetXReg<2>(cpu, sp);
  SetXReg<8>(cpu, fp);

  RunHostCall(&cpu);

  EXPECT_EQ(ra, GetXReg<1>(cpu));
  EXPECT_EQ(sp, GetXReg<2>(cpu));
  EXPECT_EQ(fp, GetXReg<8>(cpu));
}

}  // namespace

}  // namespace berberis
