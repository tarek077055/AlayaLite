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

// borrowed from https://gist.github.com/tzutalin/fd0340a93bb8d998abb9
#pragma once
#include <chrono>

class Timer {
  using clock_ = std::chrono::high_resolution_clock;
  std::chrono::time_point<clock_> m_beg_;

 public:
  Timer() : m_beg_(clock_::now()) {}

  void reset() { m_beg_ = clock_::now(); }

  // returns elapsed time in `us`
  auto elapsed() const -> uint64_t {
    return std::chrono::duration_cast<std::chrono::microseconds>(clock_::now() - m_beg_).count();
  }
};
