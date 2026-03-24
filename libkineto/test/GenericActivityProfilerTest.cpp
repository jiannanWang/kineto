/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <chrono>
#include <set>

#include "include/Config.h"
#include "include/GenericTraceActivity.h"
#include "include/libkineto.h"
#include "include/time_since_epoch.h"
#include "src/ApproximateClock.h"
#include "src/GenericActivityProfiler.h"
#include "src/output_membuf.h"

#include "src/Logger.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

// Provides ability to easily create a few test CPU-side ops
struct MockCpuActivityBuffer : public CpuTraceBuffer {
  MockCpuActivityBuffer(int64_t startTime, int64_t endTime) {
    span = TraceSpan(startTime, endTime, "Test trace");
    gpuOpCount = 0;
  }

  void addOp(
      std::string name,
      int64_t startTime,
      int64_t endTime,
      int64_t correlation) {
    GenericTraceActivity op(span, ActivityType::CPU_OP, name);
    op.startTime = startTime;
    op.endTime = endTime;
    op.device = systemThreadId();
    op.resource = systemThreadId();
    op.id = correlation;

    emplace_activity(std::move(op));
    span.opCount++;
  }

  // Variant that also sets the flow (forward-backward link) on the activity.
  void addOpWithFlow(
      std::string name,
      int64_t startTime,
      int64_t endTime,
      int64_t correlation,
      uint32_t flowId,
      uint32_t flowType,
      bool flowStart) {
    GenericTraceActivity op(span, ActivityType::CPU_OP, name);
    op.startTime = startTime;
    op.endTime = endTime;
    op.device = systemThreadId();
    op.resource = systemThreadId();
    op.id = correlation;
    op.flow.id = flowId;
    op.flow.type = flowType;
    op.flow.start = flowStart ? 1 : 0;

    emplace_activity(std::move(op));
    span.opCount++;
  }
};

// Common setup / teardown and helper functions
class GenericActivityProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    profiler_ = std::make_unique<GenericActivityProfiler>(/*cpuOnly=*/true);
    cfg_ = std::make_unique<Config>();
    cfg_->validate(std::chrono::system_clock::now());
  }

  std::unique_ptr<Config> cfg_;
  std::unique_ptr<GenericActivityProfiler> profiler_;
};

TEST_F(GenericActivityProfilerTest, BackwardDuplicateFlowIds) {
  // Test that multiple backward ops sharing the same forward-backward flow ID
  // get deduplicated with unique flow IDs during processCpuTrace.

  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler_->configure(*cfg_, start_time);
  profiler_->startTrace(start_time);
  profiler_->stopTrace(start_time + nanoseconds(duration_ns));
  libkineto::get_time_converter() = [](approx_time_t t) { return t; };
  profiler_->recordThreadInfo();

  // Create CPU ops simulating forward and backward pass with duplicate flow
  // IDs. The bug: multiple backward ops share the same flow ID (e.g., 42),
  // making them indistinguishable in traces.
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);

  // Forward op with flow ID 42 (flow start)
  cpuOps->addOpWithFlow(
      "forward_op",
      start_time_ns + 10,
      start_time_ns + 50,
      1,
      /*flowId=*/42,
      /*flowType=*/kLinkFwdBwd,
      /*flowStart=*/true);

  // First backward op with flow ID 42 (flow end) - this one should keep it
  cpuOps->addOpWithFlow(
      "MulBackward0",
      start_time_ns + 60,
      start_time_ns + 100,
      2,
      /*flowId=*/42,
      /*flowType=*/kLinkFwdBwd,
      /*flowStart=*/false);

  // Second backward op with SAME flow ID 42 (flow end) - duplicate
  cpuOps->addOpWithFlow(
      "AddBackward0",
      start_time_ns + 110,
      start_time_ns + 150,
      3,
      /*flowId=*/42,
      /*flowType=*/kLinkFwdBwd,
      /*flowStart=*/false);

  // Third backward op with SAME flow ID 42 (flow end) - duplicate
  cpuOps->addOpWithFlow(
      "ReluBackward0",
      start_time_ns + 160,
      start_time_ns + 200,
      4,
      /*flowId=*/42,
      /*flowType=*/kLinkFwdBwd,
      /*flowStart=*/false);

  // A non-duplicate backward op with a different flow ID 99
  cpuOps->addOpWithFlow(
      "SigmoidBackward0",
      start_time_ns + 210,
      start_time_ns + 250,
      5,
      /*flowId=*/99,
      /*flowType=*/kLinkFwdBwd,
      /*flowStart=*/false);

  profiler_->transferCpuTrace(std::move(cpuOps));

  // Process the trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler_->processTrace(*logger);
  profiler_->reset();

  // Verify the results: collect all flow IDs from backward ops
  const auto* activities = logger->traceActivities();
  ASSERT_NE(activities, nullptr);

  std::vector<int64_t> bwdFlowIds;
  int64_t fwdFlowId = -1;
  for (auto* activity : *activities) {
    if (activity->flowType() == kLinkFwdBwd) {
      if (activity->flowStart()) {
        fwdFlowId = activity->flowId();
      } else {
        bwdFlowIds.push_back(activity->flowId());
      }
    }
  }

  // The forward op should keep its original flow ID of 42
  EXPECT_EQ(fwdFlowId, 42);

  // We should have 4 backward ops total
  ASSERT_EQ(bwdFlowIds.size(), 4);

  // All backward flow IDs should be unique (the fix deduplicates them)
  std::set<int64_t> uniqueBwdFlowIds(bwdFlowIds.begin(), bwdFlowIds.end());
  EXPECT_EQ(uniqueBwdFlowIds.size(), bwdFlowIds.size())
      << "Backward ops should have unique flow IDs after deduplication";

  // The first backward op with flow ID 42 should keep it
  EXPECT_EQ(bwdFlowIds[0], 42);

  // The non-duplicate backward op (flow ID 99) should keep its ID
  EXPECT_EQ(bwdFlowIds[3], 99);

  // The duplicates should have gotten new unique flow IDs (> max original = 99)
  EXPECT_GT(bwdFlowIds[1], 99);
  EXPECT_GT(bwdFlowIds[2], 99);
  EXPECT_NE(bwdFlowIds[1], bwdFlowIds[2]);
}

TEST_F(GenericActivityProfilerTest, NoDuplicateFlowIdsUnchanged) {
  // Test that when backward ops have unique flow IDs, nothing is changed.

  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler_->configure(*cfg_, start_time);
  profiler_->startTrace(start_time);
  profiler_->stopTrace(start_time + nanoseconds(duration_ns));
  libkineto::get_time_converter() = [](approx_time_t t) { return t; };
  profiler_->recordThreadInfo();

  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);

  // Forward op with flow ID 10
  cpuOps->addOpWithFlow(
      "fwd1",
      start_time_ns + 10,
      start_time_ns + 30,
      1,
      /*flowId=*/10,
      /*flowType=*/kLinkFwdBwd,
      /*flowStart=*/true);

  // Forward op with flow ID 20
  cpuOps->addOpWithFlow(
      "fwd2",
      start_time_ns + 40,
      start_time_ns + 60,
      2,
      /*flowId=*/20,
      /*flowType=*/kLinkFwdBwd,
      /*flowStart=*/true);

  // Backward op with flow ID 10 (unique - no duplicate)
  cpuOps->addOpWithFlow(
      "bwd1",
      start_time_ns + 70,
      start_time_ns + 100,
      3,
      /*flowId=*/10,
      /*flowType=*/kLinkFwdBwd,
      /*flowStart=*/false);

  // Backward op with flow ID 20 (unique - no duplicate)
  cpuOps->addOpWithFlow(
      "bwd2",
      start_time_ns + 110,
      start_time_ns + 150,
      4,
      /*flowId=*/20,
      /*flowType=*/kLinkFwdBwd,
      /*flowStart=*/false);

  profiler_->transferCpuTrace(std::move(cpuOps));

  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler_->processTrace(*logger);
  profiler_->reset();

  const auto* activities = logger->traceActivities();
  ASSERT_NE(activities, nullptr);

  std::vector<int64_t> bwdFlowIds;
  for (auto* activity : *activities) {
    if (activity->flowType() == kLinkFwdBwd && !activity->flowStart()) {
      bwdFlowIds.push_back(activity->flowId());
    }
  }

  ASSERT_EQ(bwdFlowIds.size(), 2);
  // Flow IDs should remain unchanged since there are no duplicates
  EXPECT_EQ(bwdFlowIds[0], 10);
  EXPECT_EQ(bwdFlowIds[1], 20);
}

TEST_F(GenericActivityProfilerTest, NoFlowOpsUnchanged) {
  // Test that ops without flow information are unaffected.

  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler_->configure(*cfg_, start_time);
  profiler_->startTrace(start_time);
  profiler_->stopTrace(start_time + nanoseconds(duration_ns));
  libkineto::get_time_converter() = [](approx_time_t t) { return t; };
  profiler_->recordThreadInfo();

  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);

  // Regular ops without flow
  cpuOps->addOp("op1", start_time_ns + 10, start_time_ns + 50, 1);
  cpuOps->addOp("op2", start_time_ns + 60, start_time_ns + 100, 2);

  profiler_->transferCpuTrace(std::move(cpuOps));

  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler_->processTrace(*logger);
  profiler_->reset();

  const auto* activities = logger->traceActivities();
  ASSERT_NE(activities, nullptr);
  EXPECT_GE(activities->size(), 2);

  // No flow IDs should be set
  for (auto* activity : *activities) {
    if (activity->name() == "op1" || activity->name() == "op2") {
      EXPECT_EQ(activity->flowId(), 0);
    }
  }
}
