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
#include <cmath>
#include <cstddef>

namespace alaya {

inline auto cos_dist(const float *x, const float *y, size_t dim) -> float {
  float sum = 0;
  float x_norm = 0;
  float y_norm = 0;

  for (size_t i = 0; i < dim; ++i) {
    sum += x[i] * y[i];
    x_norm += x[i] * x[i];
    y_norm += y[i] * y[i];
  }
  return -sum / std::sqrt(x_norm * y_norm);
}

template <typename DataType = float>
inline void normalize(DataType *data, size_t dim) {
  float sum = 0;
  for (size_t i = 0; i < dim; ++i) {
    sum += data[i] * data[i];
  }
  sum = 1.0 / std::sqrt(sum);
  for (size_t i = 0; i < dim; ++i) {
    data[i] *= sum;
  }
}

}  // namespace alaya
