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
  cpu.x[2] = ToGuestAddr(stack.data() + stack.size());

  ScopedHostCallFrame host_call_frame(&cpu, 0xdeadbeef);

  EXPECT_EQ(kHostCallFrameGuestPC, cpu.x[1]);

  // Pretend guest code executed up to return address.
  cpu.insn_addr = cpu.x[1];
}

void RunHostCall(CPUState* cpu) {
  ScopedHostCallFrame host_call_frame(cpu, 0xbaaaaaad);

  // Pretend guest code executed up to return address.
  cpu->insn_addr = cpu->x[1];

  // Host call frame allows random adjustments of ra.
  cpu->x[1] = 0xbaadf00d;
}

TEST(HostCallFrame, Restore) {
  CPUState cpu{};

  alignas(uint64_t) std::array<char, 128> stack;
  const GuestAddr sp = ToGuestAddr(stack.data() + stack.size());
  const GuestAddr ra = 0xdeadbeef;
  const GuestAddr fp = 0xdeadc0de;

  cpu.x[1] = ra;
  cpu.x[2] = sp;
  cpu.x[8] = fp;

  RunHostCall(&cpu);

  EXPECT_EQ(ra, cpu.x[1]);
  EXPECT_EQ(sp, cpu.x[2]);
  EXPECT_EQ(fp, cpu.x[8]);
}

}  // namespace

}  // namespace berberis
