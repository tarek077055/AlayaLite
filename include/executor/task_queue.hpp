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

#include <atomic>
#include <coroutine>
#include <tuple>
#include "concurrentqueue.h"  // NOLINT
namespace alaya {

/**
 * @brief A thread-safe queue for managing coroutines tasks.
 *
 * This class provides a queue to hold coroutines and allows pushing and popping tasks
 * in a thread-safe manner using `ConcurrentQueue` from the `moodycamel` library.
 * It also tracks the number of tasks in the queue with an atomic counter.
 */
class TaskQueue {
 public:
  TaskQueue() = default;
  ~TaskQueue() = default;

  /**
   * @brief Pushes a coroutine task onto the queue.
   *
   * This function increments the task counter and enqueues the provided coroutine
   * handle into the queue.
   *
   * @param item The coroutine handle representing the task to be enqueued.
   */
  void push(std::coroutine_handle<> item) {
    task_counter_.fetch_add(1, std::memory_order_relaxed);

    queue_.enqueue(item);
  }

  /**
   * @brief Pops a task (coroutine) from the queue and returns it.
   *
   * This function attempts to dequeue a coroutine handle from the queue and if successful,
   * decrements the task counter. It returns a boolean indicating whether a task was dequeued.
   *
   * @param item Reference to the coroutine handle that will hold the dequeued task.
   * @return `true` if a task was successfully dequeued, `false` otherwise.
   */
  auto pop(std::coroutine_handle<> &item) -> bool {
    auto ret = queue_.try_dequeue(item);
    if (ret) {
      task_counter_.fetch_sub(1, std::memory_order_relaxed);
    }
    return ret;
  }

 private:
  std::atomic_size_t task_counter_{0};  ///< tracks the number of tasks in the queue.
  moodycamel::ConcurrentQueue<std::coroutine_handle<>>
      queue_;  ///< A concurrent queue that holds coroutine handles.
};
}  // namespace alaya
