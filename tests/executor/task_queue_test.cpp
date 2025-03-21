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

#include "executor/task_queue.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace alaya {
TEST(TaskQueueTest, ThreadSafePushPop) {
  alaya::TaskQueue queue;
  constexpr int kN = 1000;
  std::vector<std::thread> threads;

  threads.reserve(2);

  // Concurrent push
  for (int i = 0; i < 2; ++i) {
    threads.emplace_back([&] {
      for (int j = 0; j < kN; ++j) {
        queue.push(std::noop_coroutine());
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  // Concurrent pop
  std::atomic<int> popped{0};
  threads.clear();
  threads.reserve(4);

  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&] {
      std::coroutine_handle<> h;
      while (popped < 2 * kN) {
        if (queue.pop(h)) {
          popped++;
        }
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }
  EXPECT_EQ(popped.load(), 2 * kN);
}
}  // namespace alaya
