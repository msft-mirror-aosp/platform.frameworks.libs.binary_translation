// Copyright 2014 Google Inc. All rights reserved.

#ifndef NDK_TRANSLATION_BACKEND_X86_64_CODE_EMIT_H_
#define NDK_TRANSLATION_BACKEND_X86_64_CODE_EMIT_H_

#include "ndk_translation/assembler/x86_64.h"
#include "ndk_translation/backend/code_emitter.h"
#include "ndk_translation/backend/x86_64/machine_ir.h"
#include "ndk_translation/intrinsics/intrinsics_float.h"

namespace ndk_translation::x86_64 {

Assembler::Register GetGReg(MachineReg r);
Assembler::XMMRegister GetXReg(MachineReg r);
Assembler::ScaleFactor ToScaleFactor(MachineMemOperandScale scale);

}  // namespace ndk_translation::x86_64

#endif  // NDK_TRANSLATION_BACKEND_X86_64_CODE_EMIT_H_
