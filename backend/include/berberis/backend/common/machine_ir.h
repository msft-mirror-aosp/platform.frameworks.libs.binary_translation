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

// Machine IR public interface.

#ifndef BERBERIS_BACKEND_COMMON_MACHINE_IR_H_
#define BERBERIS_BACKEND_COMMON_MACHINE_IR_H_

#include <cstdint>
#include <limits>

#include "berberis/base/checks.h"

namespace berberis {

// MachineReg is a machine instruction argument meaningful for optimizations and
// register allocation. It can be:
// - virtual register:  [1024, +inf)
// - hard register:     [1, 1024)
// - invalid/undefined: 0
// - (reserved):        (-1024, -1]
// - spilled register:  (-inf, -1024]
class MachineReg {
 public:
  // Creates an invalid machine register.
  constexpr MachineReg() : reg_{kInvalidMachineVRegNumber} {}
  constexpr explicit MachineReg(int reg) : reg_{reg} {}
  constexpr MachineReg(const MachineReg&) = default;
  constexpr MachineReg& operator=(const MachineReg&) = default;

  constexpr MachineReg(MachineReg&&) = default;
  constexpr MachineReg& operator=(MachineReg&&) = default;

  [[nodiscard]] constexpr int reg() const { return reg_; }

  [[nodiscard]] constexpr bool IsSpilledReg() const { return reg_ <= kLastSpilledRegNumber; }

  [[nodiscard]] constexpr bool IsHardReg() const {
    return reg_ > kInvalidMachineVRegNumber && reg_ < kFirstVRegNumber;
  }

  [[nodiscard]] constexpr bool IsVReg() const { return reg_ >= kFirstVRegNumber; }

  [[nodiscard]] constexpr uint32_t GetVRegIndex() const {
    CHECK_GE(reg_, kFirstVRegNumber);
    return reg_ - kFirstVRegNumber;
  }

  [[nodiscard]] constexpr uint32_t GetSpilledRegIndex() const {
    CHECK_LE(reg_, kLastSpilledRegNumber);
    return kLastSpilledRegNumber - reg_;
  }

  constexpr friend bool operator==(MachineReg left, MachineReg right) {
    return left.reg_ == right.reg_;
  }

  constexpr friend bool operator!=(MachineReg left, MachineReg right) { return !(left == right); }

  [[nodiscard]] static constexpr MachineReg CreateVRegFromIndex(uint32_t index) {
    CHECK_LE(index, std::numeric_limits<int>::max() - kFirstVRegNumber);
    return MachineReg{kFirstVRegNumber + static_cast<int>(index)};
  }

  [[nodiscard]] static constexpr MachineReg CreateSpilledRegFromIndex(uint32_t index) {
    CHECK_LE(index, -(std::numeric_limits<int>::min() - kLastSpilledRegNumber));
    return MachineReg{kLastSpilledRegNumber - static_cast<int>(index)};
  }

  [[nodiscard]] static constexpr int GetFirstVRegNumberForTesting() { return kFirstVRegNumber; }

  [[nodiscard]] static constexpr int GetLastSpilledRegNumberForTesting() {
    return kLastSpilledRegNumber;
  }

 private:
  static constexpr int kFirstVRegNumber = 1024;
  static constexpr int kInvalidMachineVRegNumber = 0;
  static constexpr int kLastSpilledRegNumber = -1024;

  int reg_;
};

constexpr MachineReg kInvalidMachineReg{0};

}  // namespace berberis

#endif  // BERBERIS_BACKEND_COMMON_MACHINE_IR_H_
