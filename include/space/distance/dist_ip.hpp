/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include "dist_config.hpp"

namespace alaya {

FAST_BEGIN
template <typename DataType = float, typename DistanceType = float>
inline auto ip_sqr(DataType *x, DataType *y, size_t dim) -> DistanceType {
  DistanceType sum = 0;
  for (size_t i = 0; i < dim; ++i) {
    sum += x[i] * y[i];
  }
  return -sum;
}
FAST_END

FAST_BEGIN
template <typename DataType = float, typename DistanceType = float>
inline auto ip_sqr_sq4(const uint8_t *encoded_x, const uint8_t *encoded_y, size_t dim,
                       const DataType *min, const DataType *max) -> DistanceType {
  DistanceType sum = 0;

  for (size_t i = 0; i < dim; i += 2) {
    auto x = (encoded_x[i] >> 4) & 0x0F;
    auto y = encoded_y[i] & 0x0F;
    sum += (x * (max[i] - min[i]) + min[i]) * (y * (max[i] - min[i]) + min[i]);
  }

  return -sum;
}
FAST_END

FAST_BEGIN
template <typename DataType = float, typename DistanceType = float>
inline auto ip_sqr_sq8(const uint8_t *encoded_x, const uint8_t *encoded_y, size_t dim,
                       const DataType *min, const DataType *max) -> DistanceType {
  DistanceType sum = 0;

  for (size_t i = 0; i < dim; i += 1) {
    sum +=
        (encoded_x[i] * (max[i] - min[i]) + min[i]) * (encoded_y[i] * (max[i] - min[i]) + min[i]);
  }

  return -sum;
}
FAST_END

}  // namespace alaya
