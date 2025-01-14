/*
 * Copyright (C) 2024 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file excenaupt in compliance with the License.
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

#include "interpreter.h"

#ifdef BERBERIS_RISCV64_INTERPRETER_SEPARATE_INSTANTIATION_OF_VECTOR_OPERATIONS
namespace berberis {

template void SemanticsPlayer<Interpreter>::OpVector(const Decoder::VLoadStrideArgs& args);

}  // namespace berberis
#endif  // BERBERIS_RISCV64_INTERPRETER_SEPARATE_INSTANTIATION_OF_VECTOR_OPERATIONS
