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

#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "utils/evaluate.hpp"

namespace alaya {

TEST(FindExactGTTest, BasicFunctionality) {
  std::vector<float> queries = {1.0, 2.0, 3.0};
  std::vector<float> data = {3.0, 2.0, 1.0, 4.0, 5.0, 6.0};
  uint32_t dim = 3;
  uint32_t topk = 2;
  auto result = find_exact_gt(queries, data, dim, topk);
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 0);  // Closest point
  EXPECT_EQ(result[1], 1);  // Second closest
}

TEST(FindExactGTTest, EmptyData) {
  std::vector<float> queries = {1.0, 2.0, 3.0};
  std::vector<float> data;
  uint32_t dim = 3;
  uint32_t topk = 1;
  auto result = find_exact_gt(queries, data, dim, topk);
  EXPECT_TRUE(result.empty());
}

TEST(FindExactGTTest, EmptyQueries) {
  std::vector<float> queries;
  std::vector<float> data = {3.0, 2.0, 1.0};
  uint32_t dim = 3;
  uint32_t topk = 1;
  auto result = find_exact_gt(queries, data, dim, topk);
  EXPECT_TRUE(result.empty());
}

TEST(FindExactGTTest, LargeDataset) {
  std::vector<float> queries(300, 1.0);
  std::vector<float> data(3000, 2.0);
  uint32_t dim = 3;
  uint32_t topk = 5;
  auto result = find_exact_gt(queries, data, dim, topk);
  ASSERT_EQ(result.size(), queries.size() / dim * topk);
}

TEST(CalcRecallTest, PerfectMatch) {
  std::vector<uint32_t> res = {0, 1, 2, 3};
  std::vector<uint32_t> gt = {0, 1, 2, 3};
  uint32_t topk = 1;
  float recall = calc_recall(res, gt, topk);
  EXPECT_FLOAT_EQ(recall, 1.0);
}

TEST(CalcRecallTest, PartialMatch) {
  std::vector<uint32_t> res = {0, 1, 2, 3};
  std::vector<uint32_t> gt = {1, 2, 3, 4};
  uint32_t topk = 1;
  float recall = calc_recall(res, gt, topk);
  EXPECT_FLOAT_EQ(recall, 0);
}

TEST(CalcRecallTest, NoMatch) {
  std::vector<uint32_t> res = {5, 6, 7, 8};
  std::vector<uint32_t> gt = {1, 2, 3, 4};
  uint32_t topk = 1;
  float recall = calc_recall(res, gt, topk);
  EXPECT_FLOAT_EQ(recall, 0.0);
}

}  // namespace alaya
