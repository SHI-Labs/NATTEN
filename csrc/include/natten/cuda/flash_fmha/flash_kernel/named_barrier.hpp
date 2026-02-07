/******************************************************************************
 * Copyright (c) 2022-2025 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/arch/barrier.h"

namespace natten {
namespace cuda {
namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

// cutlass::arch::NamedBarrier::sync/arrive are only enabled Sm90 even though they work
// for Sm80 as well. We reimplement them here, enabled for both Sm90 and Sm80.

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts

enum class FwdNamedBarriers {
    QueryEmpty = 0,
    WarpSchedulerWG1 = 1,
    WarpSchedulerWG2 = 2,
    WarpSchedulerWG3 = 3,
    AppendKV = 4,
    QueryRotated = 5,
    PFull = 6,
    PEmpty = 7,
};

enum class BwdNamedBarriers {
    KVEmpty = 0,
    PdS = 1,
    dQEmptyWG1 = 2,
    dQEmptyWG2 = 3,
    dQEmptyWG3 = 4,
    dQFullWG1 = 5,
    dQFullWG2 = 6,
    dQFullWG3 = 7,
};

} // namespace flash
} // namespace cuda
} // namespace natten
