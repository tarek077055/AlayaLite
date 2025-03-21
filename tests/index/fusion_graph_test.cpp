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
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string_view>

#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/fusion_graph.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/graph/nsg/nsg_builder.hpp"
#include "space/raw_space.hpp"
#include "utils/io_utils.hpp"
#include "utils/timer.hpp"

namespace alaya {

class FusionGraphTest : public ::testing::Test {
 protected:
  // NOLINTBEGIN
  void SetUp() {
    // NOLINTEND
    max_node_ = 1000;
    dim_ = 1024;
    data_ = new float[max_node_ * dim_];

    // Init the vector data.
    srand(time(nullptr));
    for (int i = 0; i < max_node_; ++i) {
      for (int j = 0; j < dim_; ++j) {
        data_[i * dim_ + j] = rand() % max_node_;
      }
    }
    // build the unified data manager to compute distance.
    space_ = std::make_shared<RawSpace<>>(max_node_, dim_, MetricType::L2);
    space_->fit(data_, max_node_);
    nsg_ = std::make_unique<
        alaya::FusionGraphBuilder<alaya::RawSpace<>, alaya::HNSWBuilder<alaya::RawSpace<>>,
                                  alaya::NSGBuilder<alaya::RawSpace<>>>>(space_);
  }
  // NOLINTBEGIN
  void TearDown() {
    // NOLINTEND
    delete[] data_;
    if (std::filesystem::exists(filename_)) {
      remove(filename_.data());
    }
  }

  uint32_t max_node_;               ///< The number of vector data.
  uint32_t dim_;                    ///< The dim of vector data.
  std::string_view metric_ = "L2";  /// The metric type for building graph.
  float *data_ = nullptr;           // Store the vector data.
  std::unique_ptr<
      alaya::FusionGraphBuilder<alaya::RawSpace<>, alaya::HNSWBuilder<alaya::RawSpace<>>,
                                alaya::NSGBuilder<alaya::RawSpace<>>>>
      nsg_ = nullptr;
  std::shared_ptr<RawSpace<>> space_ = nullptr;
  std::string_view filename_ = "nnDescent.graph";
};

TEST_F(FusionGraphTest, BuildGraphTest) {
  auto graph = nsg_->build_graph();
  // graph->print_graph();
}

class FusionGraphSearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    alaya::load_fvecs(data_file_.c_str(), data_, points_num_, dim_);

    alaya::load_fvecs(query_file_.c_str(), queries_, query_num_, query_dim_);
    assert(dim_ == query_dim_);

    alaya::load_ivecs(gt_file_.c_str(), answers_, ans_num_, gt_col_);
    assert(ans_num_ == query_num_);
  }

  void TearDown() override {}

  std::filesystem::path dir_name_ = std::filesystem::current_path() / "siftsmall";
  std::filesystem::path data_file_ = dir_name_ / "siftsmall_base.fvecs";
  std::filesystem::path query_file_ = dir_name_ / "siftsmall_query.fvecs";
  std::filesystem::path gt_file_ = dir_name_ / "siftsmall_groundtruth.ivecs";

  std::vector<float> data_;
  uint32_t points_num_;
  uint32_t dim_;

  std::vector<float> queries_;
  uint32_t query_num_;
  uint32_t query_dim_;

  std::vector<uint32_t> answers_;
  uint32_t ans_num_;
  uint32_t gt_col_;
};

TEST_F(FusionGraphSearchTest, SimpleSearchTest) {
  const size_t kM = 32;
  size_t topk = 10;
  size_t ef = 100;
  std::string index_type = "FUSION_GRAPH";

  std::filesystem::path index_file = fmt::format("{}_M{}.{}", dir_name_.string(), kM, index_type);
  std::shared_ptr<alaya::RawSpace<>> space =
      std::make_shared<alaya::RawSpace<>>(points_num_, dim_, MetricType::L2);
  space->fit(data_.data(), points_num_);

  auto load_graph = std::make_shared<alaya::Graph<>>(points_num_, kM);

  if (!std::filesystem::exists(index_file)) {
    auto build_start = Timer();

    alaya::FusionGraphBuilder<alaya::RawSpace<>, alaya::HNSWBuilder<alaya::RawSpace<>>,
                              alaya::NSGBuilder<alaya::RawSpace<>>>
        fusion_graph =
            alaya::FusionGraphBuilder<alaya::RawSpace<>, alaya::HNSWBuilder<alaya::RawSpace<>>,
                                      alaya::NSGBuilder<alaya::RawSpace<>>>(space, kM);
    auto graph = fusion_graph.build_graph();

    LOG_INFO("The time of building hnsw is {}s.", build_start.elapsed() / 1000000.0);

    std::string_view path = index_file.native();
    graph->save(path);
  }
  std::string_view path = index_file.native();
  load_graph->load(path);

  // load_graph->print_graph();

  auto search_job = std::make_unique<alaya::GraphSearchJob<alaya::RawSpace<>>>(space, load_graph);

  LOG_INFO("creator task generator success");

  using IDType = uint32_t;

  Timer timer{};
  std::vector<std::vector<IDType>> res_pool(query_num_, std::vector<IDType>(topk));
  const size_t kSearchThreadNum = 16;
  std::vector<std::thread> tasks(kSearchThreadNum);

  auto search_knn = [&](int i) {
    for (; i < query_num_; i += kSearchThreadNum) {
      std::vector<uint32_t> ids(topk);
      auto cur_query = queries_.data() + i * query_dim_;
      search_job->search_solo(cur_query, topk, ids.data(), ef);

      auto id_set = std::set(ids.begin(), ids.end());

      if (id_set.size() < topk) {
        fmt::println("i id: {}", i);
        fmt::println("ids size: {}", id_set.size());
      }
      res_pool[i] = ids;
    }
  };

  for (int i = 0; i < kSearchThreadNum; i++) {
    tasks[i] = std::thread(search_knn, i);
  }

  for (int i = 0; i < kSearchThreadNum; i++) {
    if (tasks[i].joinable()) {
      tasks[i].join();
    }
  }

  LOG_INFO("total time: {} s.", timer.elapsed() / 1000000.0);

  // Computing recall;
  size_t cnt = 0;
  for (int i = 0; i < query_num_; i++) {
    for (int j = 0; j < topk; j++) {
      for (int k = 0; k < topk; k++) {
        if (res_pool[i][j] == answers_[i * gt_col_ + k]) {
          cnt++;
          break;
        }
      }
    }
  }

  float recall = cnt * 1.0 / query_num_ / topk;
  LOG_INFO("recall is {}.", recall);
  EXPECT_GE(recall, 0.5);
}

}  // namespace alaya
