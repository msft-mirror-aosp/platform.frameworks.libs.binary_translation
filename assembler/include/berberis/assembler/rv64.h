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

// Assembler to produce RV64 instructions (no ABI version). Somewhat influenced by V8 assembler.

#ifndef BERBERIS_ASSEMBLER_RV64_H_
#define BERBERIS_ASSEMBLER_RV64_H_

#include <type_traits>  // std::is_same

#include "berberis/assembler/common_riscv.h"

namespace berberis::rv64 {

class Assembler : public AssemblerRiscV<Assembler> {
 public:
  explicit Assembler(MachineCode* code) : AssemblerRiscV(code) {}

  friend AssemblerRiscV<Assembler>;

// Instructions.
#include "berberis/assembler/gen_assembler_rv64-inl.h"  // NOLINT generated file!

 private:
  Assembler() = delete;
  Assembler(const Assembler&) = delete;
  Assembler(Assembler&&) = delete;
  void operator=(const Assembler&) = delete;
  void operator=(Assembler&&) = delete;
};

}  // namespace berberis::rv64

#endif  // BERBERIS_ASSEMBLER_RV64_H_
