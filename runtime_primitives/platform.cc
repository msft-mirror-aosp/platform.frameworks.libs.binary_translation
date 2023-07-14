/*
 * Copyright (C) 2014 The Android Open Source Project
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

#include "berberis/runtime_primitives/platform.h"

#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#endif

namespace berberis::host_platform {

namespace {

#if defined(__i386__) || defined(__x86_64__)
auto Init() {
  PlatformCapabilities platform_capabilities = {};
  unsigned int eax, ebx, ecx, edx;
  __cpuid(1, eax, ebx, ecx, edx);
  platform_capabilities.kHasSSE3 = ecx & bit_SSE3;
  platform_capabilities.kHasSSSE3 = ecx & bit_SSSE3;
  platform_capabilities.kHasSSE4_1 = ecx & bit_SSE4_1;
  platform_capabilities.kHasSSE4_2 = ecx & bit_SSE4_2;
  platform_capabilities.kHasF16C = ecx & bit_F16C;
  platform_capabilities.kHasFMA = ecx & bit_FMA;
  platform_capabilities.kHasAES = ecx & bit_AES;
  platform_capabilities.kHasAVX = ecx & bit_AVX;
  platform_capabilities.kHasCLMUL = ecx & bit_PCLMUL;
  __cpuid(0x80000001, eax, ebx, ecx, edx);
  platform_capabilities.kHasLZCNT = ecx & bit_LZCNT;
  platform_capabilities.kHasSSE4a = ecx & bit_SSE4a;
  platform_capabilities.kHasFMA4 = ecx & bit_FMA4;
  __cpuid(7, eax, ebx, ecx, edx);
  platform_capabilities.kHasSHA = ebx & bit_SHA;
  return platform_capabilities;
}
#endif

}  // namespace

#if defined(__i386__) || defined(__x86_64__)
const PlatformCapabilities kPlatformCapabilities = Init();
#endif

}  // namespace berberis::host_platform
