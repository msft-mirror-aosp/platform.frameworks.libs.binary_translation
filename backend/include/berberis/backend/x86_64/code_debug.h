// Copyright 2014 Google Inc. All rights reserved.

#ifndef NDK_TRANSLATION_BACKEND_X86_64_CODE_DEBUG_H_
#define NDK_TRANSLATION_BACKEND_X86_64_CODE_DEBUG_H_

#include <string>

#include "ndk_translation/backend/x86_64/machine_ir.h"

namespace ndk_translation {

namespace x86_64 {

using std::string;

string GetImplicitRegOperandDebugString(const MachineInsnX86_64* insn, int i);
string GetAbsoluteMemOperandDebugString(const MachineInsnX86_64* insn);
string GetBaseDispMemOperandDebugString(const MachineInsnX86_64* insn, int i);
string GetIndexDispMemOperandDebugString(const MachineInsnX86_64* insn, int i);

// The index operand must immediately follow the base operand.
string GetBaseIndexDispMemOperandDebugString(const MachineInsnX86_64* insn, int i);
string GetImmOperandDebugString(const MachineInsnX86_64* insn);
string GetCondOperandDebugString(const MachineInsnX86_64* insn);
string GetLabelOperandDebugString(const MachineInsnX86_64* insn);

}  // namespace x86_64

}  // namespace ndk_translation

#endif  // NDK_TRANSLATION_BACKEND_X86_64_CODE_DEBUG_H_
