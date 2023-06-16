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

namespace berberis {

// TODO(b/279068747): Ensure these paths are correct.
// Paths required by guest_loader_impl.h.
const char* kAppProcessPath = "/system/bin/riscv64/app_process64";
const char* kPtInterpPath = "/system/bin/riscv64/linker64";
const char* kVdsoPath = "/system/lib64/riscv64/libnative_bridge_vdso.so";

}  // namespace berberis