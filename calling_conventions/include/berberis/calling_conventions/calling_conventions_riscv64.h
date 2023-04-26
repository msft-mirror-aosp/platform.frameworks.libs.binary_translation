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

#ifndef BERBERIS_CALLING_CONVENTIONS_CALLING_CONVENTIONS_RISCV64_H_
#define BERBERIS_CALLING_CONVENTIONS_CALLING_CONVENTIONS_RISCV64_H_

#include "berberis/base/bit_util.h"
#include "berberis/base/logging.h"

namespace berberis::riscv64 {

enum ArgLocationKind {
  kArgLocationNone = 0,
  kArgLocationStack,
  kArgLocationInt,  // x10 - x17
  kArgLocationFp,   // f10 - f17
};

struct ArgLocation {
  ArgLocationKind kind;
  unsigned offset;  // meaning of offset depends on kind!
};

class CallingConventions {
 public:
  static constexpr unsigned kStackAlignmentBeforeCall = 16;

  constexpr ArgLocation GetNextIntArgLoc(unsigned size, unsigned alignment) {
    // Fundamental integer type - 1/1, 2/2, 4/4, 8/8, 16/16.
    CHECK_LE(size, 16u);
    CHECK_EQ(size, alignment);

    unsigned size_in_regs = size > 8 ? 2 : 1;
    unsigned alignment_in_regs = size_in_regs;

    unsigned aligned_int_offset = AlignUp(int_offset_, alignment_in_regs);

    if (aligned_int_offset + size_in_regs <= kMaxIntOffset) {
      // Use 1 or 2 int regs.
      ArgLocation loc{kArgLocationInt, aligned_int_offset};
      int_offset_ = aligned_int_offset + size_in_regs;
      return loc;
    }

    // ATTENTION: _ensure_ no more int regs params!
    int_offset_ = kMaxIntOffset;

    return GetNextStackArgLoc(size, alignment);
  }

  constexpr ArgLocation GetNextFpArgLoc(unsigned size, unsigned alignment) {
    // Fundamental floating-point type - 4/4, 8/8, 16/16.
    CHECK_LE(size, 16u);
    CHECK_EQ(size, alignment);

    if (fp_offset_ < kMaxFpOffset) {
      // Use 1 fp reg.
      ArgLocation loc{kArgLocationFp, fp_offset_};
      ++fp_offset_;
      return loc;
    }

    return GetNextStackArgLoc(size, alignment);
  }

  constexpr ArgLocation GetIntResLoc(unsigned size) {
    // Fundamental integer type - 1/1, 2/2, 4/4, 8/8, 16/16.
    CHECK_LE(size, 16u);

    // Use x10/a0.
    return {kArgLocationInt, 10u};
  }

  constexpr ArgLocation GetFpResLoc(unsigned size) {
    // Fundamental floating-point type - 4/4, 8/8, 16/16.
    CHECK_LE(size, 16u);

    // Use f10/fa0.
    return {kArgLocationFp, 10u};
  }

 private:
  constexpr ArgLocation GetNextStackArgLoc(unsigned size, unsigned alignment) {
    CHECK_LE(size, 16u);
    CHECK_EQ(size, alignment);

    // Arguments that fit in a pointer word are aligned at 8 bytes. Larger
    // arguments are naturally aligned (i.e. 16-byte alignment for 16 byte
    // arguments).
    unsigned alignment_in_stack = alignment > 8 ? alignment : 8;
    unsigned size_in_stack = AlignUp(size, alignment_in_stack);

    unsigned aligned_stack_offset = AlignUp(stack_offset_, alignment_in_stack);

    ArgLocation loc{kArgLocationStack, aligned_stack_offset};
    stack_offset_ = aligned_stack_offset + size_in_stack;
    return loc;
  }

  static constexpr unsigned kMaxIntOffset = 8u;
  static constexpr unsigned kMaxFpOffset = 8u;

  unsigned int_offset_ = 0;
  unsigned fp_offset_ = 0;
  unsigned stack_offset_ = 0;
};

}  // namespace berberis::riscv64

#endif  // BERBERIS_CALLING_CONVENTIONS_CALLING_CONVENTIONS_RISCV64_H_
