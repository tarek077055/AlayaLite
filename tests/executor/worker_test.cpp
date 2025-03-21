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

#include "executor/worker.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <memory>
#include <thread>

namespace alaya {

class WorkerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    total_tasks_ = std::make_shared<std::atomic<size_t>>(0);
    finished_tasks_ = std::make_shared<std::atomic<size_t>>(0);
    task_queue_ = std::make_shared<TaskQueue>();

    worker_ = std::make_shared<Worker>(1, 0, task_queue_.get(), total_tasks_.get(),
                                       finished_tasks_.get());
  }

  std::shared_ptr<TaskQueue> task_queue_;
  std::shared_ptr<std::atomic<size_t>> total_tasks_;
  std::shared_ptr<std::atomic<size_t>> finished_tasks_;
  std::shared_ptr<Worker> worker_;

  auto create_mock_task(std::atomic<int> &counter) {
    struct Task {
      std::atomic<int> &counter_;
      void operator()() { counter_++; }
    };
    return std::coroutine_handle<>::from_address(new Task{counter});
  }
};

// Validate basic properties
TEST_F(WorkerTest, Initialization) {
  EXPECT_EQ(worker_->id(), 1);
  EXPECT_EQ(worker_->cpu_id(), 0);
}

}  // namespace alaya
