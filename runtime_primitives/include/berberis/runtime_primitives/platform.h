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

#ifndef BERBERIS_RUNTIME_PRIMITIVES_PLATFORM_H_
#define BERBERIS_RUNTIME_PRIMITIVES_PLATFORM_H_

namespace berberis::host_platform {

#if defined(__i386__) && !defined(__x32__)
static const bool kIsX86_32 = true;
static const bool kIsX86_64 = false;
#endif
#if defined(__x86_64__) || defined(__x32__)
static const bool kIsX86_32 = false;
static const bool kIsX86_64 = true;
#endif
static const bool kIsX86 = (kIsX86_32 || kIsX86_64);

}  // namespace berberis::host_platform

#endif  // BERBERIS_RUNTIME_PRIMITIVES_PLATFORM_H_
