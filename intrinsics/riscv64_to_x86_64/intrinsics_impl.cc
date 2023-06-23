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

#include "berberis/intrinsics/intrinsics.h"

namespace berberis::intrinsics {

std::tuple<Float32> FSgnjS(Float32 x, Float32 y) {
  return FSgnj(x, y);
}

std::tuple<Float64> FSgnjD(Float64 x, Float64 y) {
  return FSgnj(x, y);
}

std::tuple<Float32> FSgnjnS(Float32 x, Float32 y) {
  return FSgnjn(x, y);
}

std::tuple<Float64> FSgnjnD(Float64 x, Float64 y) {
  return FSgnjn(x, y);
}

std::tuple<Float32> FSgnjxS(Float32 x, Float32 y) {
  return FSgnjx(x, y);
}

std::tuple<Float64> FSgnjxD(Float64 x, Float64 y) {
  return FSgnjx(x, y);
}

}  // namespace berberis::intrinsics
