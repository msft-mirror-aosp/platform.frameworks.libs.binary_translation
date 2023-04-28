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

#include "berberis/runtime_primitives/runtime_library.h"

extern "C" {

void berberis_entry_Stop() {}
void berberis_entry_NoExec() {}
void berberis_entry_NotTranslated() {}
void berberis_entry_Translating() {}
void berberis_entry_Invalidating() {}
void berberis_entry_Wrapping() {}

}  // extern "C"
