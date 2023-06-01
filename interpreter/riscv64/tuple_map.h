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

#ifndef BERBERIS_BASE_TUPLE_MAP_H_
#define BERBERIS_BASE_TUPLE_MAP_H_

#include <initializer_list>
#include <tuple>
#include <vector>

namespace berberis {

// Helper function for the unit tests. Can be used to normalize values before processing.
//
// “container” is supposed to be container of tuples, e.g. std::initializer_list<std::tuple<…>>.
// “transformer” would be applied to the individual elements of tuples in the following loop:
//
//   for (auto& [value1, value2, value3] : TupleMap(container, [](auto value){ return …; })) {
//     …
//   }
//
// Returns vector of tuples where each tuple element is processed by transformer.
template <typename ContainerType, typename Transformer>
decltype(auto) TupleMap(const ContainerType& container, const Transformer& transformer) {
  using std::begin;

  auto transform_tuple_func = [&transformer](auto&&... value) {
    return std::tuple{transformer(value)...};
  };

  std::vector<decltype(std::apply(transform_tuple_func, *begin(container)))> result;

  for (const auto& tuple : container) {
    result.push_back(std::apply(transform_tuple_func, tuple));
  }

  return result;
}

}  // namespace berberis

#endif  // BERBERIS_BASE_TUPLE_MAP_H_
