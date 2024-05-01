/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
 *
 * Fused Neighborhood Attention kernels are heavily based on the
 * memory-efficient attention kernels from the xFormers project by Meta
 * Platforms, Inc.
 *
 * Copyright (c) Facebook, Inc. and its affiliates
 *
 * BSD 3-Clause License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC
 * Laboratories America and IDIAP Research Institute nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/vector.h>
#include <cutlass/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>

#include <cutlass/epilogue/threadblock/default_epilogue_simt.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/gemm/threadblock/default_mma_core_simt.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm70.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm75.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/platform/platform.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>

#include <natten/cuda/fna/epilogue/epilogue_pipelined.h>
#include <natten/cuda/fna/epilogue/epilogue_rescale_output.h>
#include <natten/cuda/fna/epilogue/predicated_tile_iterator.h>
#include <natten/cuda/fna/gemm/custom_mma.h>
#include <natten/cuda/fna/gemm/find_default_mma.h>
#include <natten/cuda/fna/gemm/mma_from_smem.h>
#include <natten/cuda/fna/gemm_kernel_utils.h>

#include <natten/cuda/fna/na_utils.cuh>

#include <inttypes.h>

namespace natten {
namespace cuda {
namespace fna {

namespace {
template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSmFw() {
  return (
      Arch::kMinComputeCapability >= 80 &&
              !cutlass::platform::is_same<scalar_t, float>::value
          ? 16
          : 12);
}
static CUTLASS_DEVICE float atomicMaxFloat(float* addr, float value) {
  // source: https://stackoverflow.com/a/51549250
  return (value >= 0)
      ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
      : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}

} // namespace

template <
    // NA spatial dimension; can be 1 (NA1d), 2 (NA2d), or 3 (NA3d).
    int NADim_,
    typename CausalMask,
    // The datatype of Q/K/V
    typename scalar_t_,
    // Architecture we are targeting (eg `cutlass::arch::Sm80`)
    typename ArchTag,
    // If Q/K/V are correctly aligned in memory and we can run a fast kernel
    bool isAligned_,
    int kQueriesPerBlock_,
    int kKeysPerBlock_,
    // upperbound on `max(value.shape[-1], query.shape[-1])`
    int kMaxK_ = (int)cutlass::platform::numeric_limits<uint32_t>::max(),
    bool kSupportsRPB_ = false,
    bool kStoresLSE_ = false>
struct FusedNeighborhoodAttentionKernel {
  static constexpr int NADim = NADim_;
  static_assert(NADim >= 1 && NADim < 4, "Only 1D-3D NA are implemented.");
  using Dim = typename GetDim<NADim>::type;
  using NAMask = NeighborhoodAttentionMask<NADim, CausalMask>;

  static constexpr int kKeysPerBlock = kKeysPerBlock_;
  static constexpr int kQueriesPerBlock = kQueriesPerBlock_;
  static constexpr int kMaxK = kMaxK_;

  // static constexpr Dim QueryTileShape = MapTileSizeToNd<kQueriesPerBlock,
  // NADim>::value; static constexpr Dim KeyTileShape =
  // MapTileSizeToNd<kKeysPerBlock, NADim>::value;
  static constexpr bool kHasCausalDims = CausalMask::AnyCausalDims;
  static constexpr bool kSupportsRPB = kSupportsRPB_;
  static constexpr bool kStoresLSE = kStoresLSE_;
  static_assert(
      !kSupportsRPB || !kHasCausalDims,
      "Causal NA does not support RPB yet.");

  using scalar_t = scalar_t_;
  using accum_t = float;
  using lse_scalar_t = float;
  using output_t = scalar_t;
  // Accumulator between 2 iterations
  // Using `accum_t` improves perf on f16 at the cost of
  // numerical errors
  using output_accum_t = accum_t;
  static constexpr bool kIsAligned = isAligned_;
  static constexpr bool kSingleValueIteration = kMaxK <= kKeysPerBlock;
  static constexpr int32_t kAlignLSE = 32; // block size of backward
  static constexpr bool kIsHalf = cutlass::sizeof_bits<scalar_t>::value == 16;
  static constexpr bool kPreloadV =
      ArchTag::kMinComputeCapability >= 80 && kIsHalf;
  static constexpr bool kKeepOutputInRF = kSingleValueIteration;
  static constexpr bool kNeedsOutputAccumulatorBuffer = !kKeepOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  static_assert(kQueriesPerBlock % 32 == 0, "");
  static_assert(kKeysPerBlock % 32 == 0, "");
  static constexpr int kNumWarpsPerBlock =
      kQueriesPerBlock * kKeysPerBlock / (32 * 32);
  static constexpr int kWarpSize = 32;

  // Launch bounds
  static constexpr int kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int kMinBlocksPerSm =
      getWarpsPerSmFw<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  struct Params {
    // Input tensors
    scalar_t* query_ptr = nullptr; // [..., num_heads, head_dim]
    scalar_t* key_ptr = nullptr; // [..., num_heads, head_dim]
    scalar_t* value_ptr = nullptr; // [..., num_heads, head_dim_value]

    scalar_t* rpb_ptr = nullptr; // [num_heads, *(kernel_size * 2 - 1)]

    // Output tensors
    output_t* output_ptr =
        nullptr; // [num_queries_post_partitioning, num_heads, head_dim_value]
    // [num_queries_post_partitioning, num_heads, head_dim_value]
    output_accum_t* output_accum_ptr = nullptr;
    // [num_heads, num_queries_post_partitioning] - can be null
    lse_scalar_t* logsumexp_ptr = nullptr;

    // Sliding window. ignored if == 0
    Dim kernel_size;
    Dim dilation;
    Dim cta_shape_x;

    // Scale
    accum_t scale = 0.0;

    // Dimensions/strides
    int32_t head_dim = 0;
    int32_t head_dim_value = 0;
    Dim num_queries;
    Dim num_queries_post_partitioning;

    Dim query_tile_shape;
    Dim key_tile_shape;

    Dim lse_strideM;
    Dim q_strideM;
    Dim k_strideM;
    Dim v_strideM;
    Dim o_strideM;

    Dim rpb_stride;

    int32_t num_heads = 0;
    int32_t num_batches = 0;

    // Moves pointers to what we should process
    // Returns "false" if there is no work to do
    CUTLASS_DEVICE bool advance_to_block() {
      num_queries_post_partitioning = ceil_div_dim(num_queries, dilation);

      auto batch_id = blockIdx.z;
      auto bidxy = blockIdx.y;
      auto dilation_size = dilation.prod32();
      auto dilation_dim_idx = (int32_t)bidxy % dilation_size;
      auto dilation_idx = map_index_to_coord(dilation_dim_idx, dilation);
      auto head_id = bidxy / dilation_size;

      auto q_strideH = head_dim;
      auto k_strideH = head_dim;
      auto v_strideH = head_dim_value;
      auto o_strideH = head_dim_value;

      auto q_heads_dims = num_heads * q_strideH;
      auto k_heads_dims = num_heads * k_strideH;
      auto v_heads_dims = num_heads * v_strideH;
      auto o_heads_dims = num_heads * o_strideH;

      auto q_stride_dilation = compute_stride(num_queries, q_heads_dims);
      auto k_stride_dilation = compute_stride(num_queries, k_heads_dims);
      auto v_stride_dilation = compute_stride(num_queries, v_heads_dims);
      auto o_stride_dilation = compute_stride(num_queries, o_heads_dims);

      q_strideM = q_stride_dilation * dilation;
      k_strideM = k_stride_dilation * dilation;
      v_strideM = v_stride_dilation * dilation;
      o_strideM = o_stride_dilation * dilation;

      auto q_strideB = num_queries.prod() * q_heads_dims;
      auto k_strideB = num_queries.prod() * k_heads_dims;
      auto v_strideB = num_queries.prod() * v_heads_dims;
      auto o_strideB = num_queries.prod() * o_heads_dims;

      maybe_mask_qk_tiles(
          num_queries_post_partitioning, num_queries, dilation, dilation_idx);

      cta_shape_x =
          ceil_div_dim(num_queries_post_partitioning, query_tile_shape);

      const auto first_query =
          map_index_to_coord((int32_t)blockIdx.x, cta_shape_x) *
          query_tile_shape;

      // Advance to current batch
      query_ptr += batch_id * q_strideB;
      key_ptr += batch_id * k_strideB;
      value_ptr += batch_id * v_strideB;
      output_ptr += batch_id * o_strideB;
      if (output_accum_ptr != nullptr) {
        output_accum_ptr += batch_id * o_strideB;
      }

      // Advance to the current batch / head / first_query
      query_ptr += (first_query * q_strideM).sum() +
          (dilation_idx * q_stride_dilation).sum() + head_id * q_strideH;
      key_ptr += (dilation_idx * k_stride_dilation).sum() + head_id * k_strideH;
      value_ptr +=
          (dilation_idx * v_stride_dilation).sum() + head_id * v_strideH;
      output_ptr += (first_query * o_strideM).sum() +
          (dilation_idx * o_stride_dilation).sum() + head_id * o_strideH;

      if constexpr (kSupportsRPB) {
        if (rpb_ptr != nullptr) {
          auto head_rpb_shape = (kernel_size * 2) - 1;
          rpb_ptr += head_id * head_rpb_shape.prod();
          rpb_stride = compute_stride(head_rpb_shape, 1);
        }
      }

      if (output_accum_ptr != nullptr) {
        output_accum_ptr += (first_query * o_strideM).sum() +
            (dilation_idx * o_stride_dilation).sum() + head_id * o_strideH;
      } else {
        // Accumulate directly in the destination buffer (eg for f32)
        output_accum_ptr = (accum_t*)output_ptr;
      }

      if constexpr (kStoresLSE) {
        if (logsumexp_ptr != nullptr) {
          auto lse_stride_dilation = compute_stride(num_queries, num_heads);
          lse_strideM = lse_stride_dilation * dilation;

          // NOTE(alih): we don't pad for 128-bit alignment like in xFormers,
          // because we have to rewrite the vector iterator in the backward
          // kernel into a gather op.
          // auto lse_dim = cutlass::ceil_div((int32_t)num_queries.prod32(),
          // kAlignLSE) * kAlignLSE;
          // lse[batch_id, query_start, head_id]
          logsumexp_ptr += batch_id * num_queries.prod32() * num_heads +
              ((first_query * lse_strideM).sum() +
               (dilation_idx * lse_stride_dilation).sum()) +
              head_id;
        }
      }

      num_batches = 0; // no longer used after
      // dilation = 0;

      // Make sure the compiler knows these variables are the same on all
      // the threads of the warp.
      // Only worth doing if they could have been modified above.
      query_ptr = gemm_kernel_utils::warp_uniform(query_ptr);
      key_ptr = gemm_kernel_utils::warp_uniform(key_ptr);
      value_ptr = gemm_kernel_utils::warp_uniform(value_ptr);
      output_ptr = gemm_kernel_utils::warp_uniform(output_ptr);
      output_accum_ptr = gemm_kernel_utils::warp_uniform(output_accum_ptr);
      logsumexp_ptr = gemm_kernel_utils::warp_uniform(logsumexp_ptr);
      num_heads = gemm_kernel_utils::warp_uniform(num_heads);
      return true;
    }

    __host__ dim3 getBlocksGrid() const {
      return dim3(
          ceil_div_dim(ceil_div_dim(num_queries, dilation), query_tile_shape)
              .prod32(),
          num_heads * dilation.prod32(),
          num_batches);
    }

    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize, kNumWarpsPerBlock, 1);
    }
  };

  struct MM0 {
    /*
      In this first matmul, we compute a block of `Q @ K.T`.
      While the calculation result is still hot in registers, we update
      `mi`, `m_prime`, `s_prime` in shared-memory, and then store this value
      into a shared-memory ("AccumulatorSharedStorage") that is used later as
      operand A for the second matmul (see MM1)
    */
    using GemmType = gemm_kernel_utils::DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            scalar_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int kAlignmentA =
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment;
    static constexpr int kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
        NADim,
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::ColumnMajor, // LayoutB,
        kAlignmentB,
        accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        OpClass,
        ArchTag, // ArchTag
        ThreadblockShape, // ThreadblockShape
        WarpShape, // WarpShape
        typename GemmType::InstructionShape, // InstructionShape
        ArchTag::kMinComputeCapability >= 80 && kIsHalf
            ? 4
            : DefaultConfig::kStages,
        typename GemmType::Operator // Operator
        >::DefaultMma;
    using MmaCore = typename DefaultMma::MmaCore;
    using IteratorA = typename DefaultMma::IteratorA;
    using IteratorB = typename DefaultMma::IteratorB;
    using DefaultThreadblockMma = typename DefaultMma::ThreadblockMma;
    using Mma = typename cutlass::platform::conditional<
        kSingleValueIteration,
        typename MakeCustomMma<DefaultThreadblockMma, kMaxK>::Mma,
        DefaultThreadblockMma>::type;
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Iterator;
    static_assert(
        MmaCore::WarpCount::kM * MmaCore::WarpCount::kN *
                MmaCore::WarpCount::kK ==
            kNumWarpsPerBlock,
        "");

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        NADim,
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MM1 {
    /**
      Second matmul: perform `attn @ V` where `attn` is the attention (not
      normalized) and stored in shared memory
    */
    using GemmType = gemm_kernel_utils::DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            output_accum_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int kAlignmentA = DefaultConfig::kAlignmentA; // from smem
    static constexpr int kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using LayoutB = cutlass::layout::RowMajor;
    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        LayoutB, // LayoutB,
        kAlignmentB,
        output_accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        ArchTag::kMinComputeCapability >= 80 && kIsHalf
            ? 4
            : DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Policy::Operator::Shape, // WarpShape
            typename DefaultGemm::Mma::Policy::Operator::InstructionShape,
            typename DefaultGemm::Mma::Policy::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy>::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            NADim,
            typename DefaultGemm::Mma,
            MM0::AccumulatorSharedStorage::Shape::kN, // kMaxK
            WarpIteratorA,
            false>; // kScaleOperandA
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;
    static_assert(
        WarpCount::kM * WarpCount::kN * WarpCount::kK == kNumWarpsPerBlock,
        "");

    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::CustomPredicatedTileIterator<
            NADim,
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_t>;
    using OutputTileIteratorAccum =
        typename cutlass::epilogue::threadblock::CustomPredicatedTileIterator<
            NADim,
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_accum_t>;
  };

  static constexpr int64_t kAlignmentQ = MM0::kAlignmentA;
  static constexpr int64_t kAlignmentK = MM0::kAlignmentB;
  static constexpr int64_t kAlignmentV = 1;

  // Shared storage - depends on kernel params
  struct ScalingCoefs {
    cutlass::Array<accum_t, kQueriesPerBlock> m_prime;
    cutlass::Array<accum_t, kQueriesPerBlock> s_prime;
    cutlass::Array<accum_t, kQueriesPerBlock> mi;
    cutlass::Array<accum_t, kQueriesPerBlock> out_rescale;
    cutlass::Array<accum_t, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>
        addition_storage;
  };

  struct SharedStorageEpilogueAtEnd : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      union {
        typename MM0::AccumulatorSharedStorage si;
      };
      typename MM1::Mma::SharedStorage mm1;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return epilogue;
    }
  };

  struct SharedStorageEpilogueInLoop : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      union {
        typename MM0::AccumulatorSharedStorage si;
      };
      typename MM1::Mma::SharedStorage mm1;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return after_mm0.epilogue;
    }
  };

  using SharedStorage = typename cutlass::platform::conditional<
      kSingleValueIteration || kKeepOutputInRF,
      SharedStorageEpilogueAtEnd,
      SharedStorageEpilogueInLoop>::type;

  static bool __host__ check_supported(Params const& p) {
    CHECK_ALIGNED_PTR(p.query_ptr, kAlignmentQ);
    CHECK_ALIGNED_PTR(p.key_ptr, kAlignmentK);
    CHECK_ALIGNED_PTR(p.value_ptr, kAlignmentV);
    NATTEN_CHECK(
        p.head_dim % kAlignmentQ == 0,
        "query is not correctly aligned (strideM)");
    NATTEN_CHECK(
        p.head_dim % kAlignmentK == 0,
        "key is not correctly aligned (strideM)");
    NATTEN_CHECK(
        p.head_dim_value % kAlignmentV == 0,
        "value is not correctly aligned (strideM)");
    return true;
  }

  static void CUTLASS_DEVICE attention_kernel(Params& p) {
    // In this block, we will only ever:
    // - read query[first_query:last_query, :]
    // - write to output[first_query:last_query, :]

    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
    auto& m_prime = shared_storage.m_prime;
    auto& s_prime = shared_storage.s_prime;
    auto& mi = shared_storage.mi;
    auto& out_rescale = shared_storage.out_rescale;
    auto first_query = map_index_to_coord((int32_t)blockIdx.x, p.cta_shape_x) *
        p.query_tile_shape;
    auto problem_size_0_m = fast_min(
        p.query_tile_shape, p.num_queries_post_partitioning - first_query);
    auto last_query = first_query + problem_size_0_m - 1;
    auto problem_size_0_m_int = problem_size_0_m.prod32();

    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    if (thread_id() < kQueriesPerBlock) {
      s_prime[thread_id()] = accum_t(0);
      out_rescale[thread_id()] = accum_t(1.0);
      m_prime[thread_id()] =
          -cutlass::platform::numeric_limits<accum_t>::infinity();
      mi[thread_id()] = -cutlass::platform::numeric_limits<accum_t>::infinity();
    }
    typename MM1::Mma::FragmentC accum_o;
    accum_o.clear();

    auto createOutputIter = [&](int col) -> typename MM1::OutputTileIterator {
      using OutputTileIterator = typename MM1::OutputTileIterator;
      return OutputTileIterator(
          typename OutputTileIterator::Params{p.o_strideM, problem_size_0_m},
          p.output_ptr,
          problem_size_0_m,
          p.head_dim_value,
          thread_id(),
          {0, col});
    };

    auto createOutputAccumIter = [&](int col) ->
        typename MM1::OutputTileIteratorAccum {
          using OutputTileIteratorAccum = typename MM1::OutputTileIteratorAccum;
          return OutputTileIteratorAccum(
              typename OutputTileIteratorAccum::Params{
                  p.o_strideM, problem_size_0_m},
              p.output_accum_ptr,
              problem_size_0_m,
              p.head_dim_value,
              thread_id(),
              {0, col});
        };

    int32_t const& problem_size_0_k_int = p.head_dim;
    int32_t const& problem_size_1_n_int = p.head_dim_value;
    auto gemm_k_iterations =
        (problem_size_0_k_int + MM0::Mma::Shape::kK - 1) / MM0::Mma::Shape::kK;
    const int64_t nBlockN = kSingleValueIteration
        ? 1
        : gemm_kernel_utils::ceil_div(
              (int64_t)problem_size_1_n_int,
              int64_t(MM1::ThreadblockShape::kN));

    auto na_mask = NAMask(p.kernel_size, p.num_queries_post_partitioning);

    auto first_key = na_mask.get_window_start(first_query);
    auto last_key = na_mask.get_window_end_(last_query);

    auto my_warp_id = gemm_kernel_utils::warp_uniform(warp_id());
    auto my_lane_id = lane_id();

    typename MM0::Mma::Operator::IteratorC::TensorCoord iteratorC_tile_offset =
        {/*(tb_tile_offset.m() * MM0::Mma::WarpCount::kM) +*/
         (my_warp_id % MM0::Mma::WarpCount::kM),
         /*(tb_tile_offset.n() * MM0::Mma::WarpCount::kN) +*/
         (my_warp_id / MM0::Mma::WarpCount::kM)};

    auto lane_offset = MM0::AccumLambdaIterator::get_lane_offset(
        my_lane_id, my_warp_id, iteratorC_tile_offset);

    int warp_idx_mn_0 = my_warp_id %
        (MM0::Mma::Base::WarpCount::kM * MM0::Mma::Base::WarpCount::kN);
    auto output_tile_coords = cutlass::MatrixCoord{
        warp_idx_mn_0 % MM0::Mma::Base::WarpCount::kM,
        warp_idx_mn_0 / MM0::Mma::Base::WarpCount::kM};

#ifdef CACHE_NEIGHBORHOOD_COORDINATES
    // NA mask info
    static constexpr auto kNumRows = MM0::AccumLambdaIterator::kRowIters;
    Dim first_cols[kNumRows];
    // Dim last_cols[kNumRows];
    Dim key_bounds[kNumRows];
    Dim rpb_starts[kNumRows];
    MM0::AccumLambdaIterator::iterateRows(
        lane_offset,
        [&](int accum_m) {
          auto row_idx =
              map_index_to_coord((int32_t)accum_m, problem_size_0_m) +
              first_query;
          first_cols[accum_m] = na_mask.get_window_start(row_idx);
          // last_cols[accum_m] = na_mask.get_window_end(row_idx,
          // first_cols[accum_m]);
          key_bounds[accum_m] =
              na_mask.get_window_end(row_idx, first_cols[accum_m]) -
              first_cols[accum_m];
          if constexpr (kSupportsRPB) {
            rpb_starts[accum_m] = na_mask.get_rpb_start(row_idx);
          }
        },
        [&](int accum_m, int accum_n, int idx) {},
        [&](int accum_m) {});
#endif

    // Iterate through keys
    for (auto iter_key_start = first_key;
         is_coord_within_upper_bound(iter_key_start, last_key);
         increment_tile(
             iter_key_start, p.key_tile_shape, first_key, last_key)) {
      const auto problem_size_0_n =
          fast_min(p.key_tile_shape, last_key - iter_key_start);
      auto const& problem_size_1_k = problem_size_0_n;
      auto problem_size_0_n_int = problem_size_0_n.prod32();
      auto& gemm_1_num_iterations = problem_size_0_n_int;

      auto& last_kv_col = gemm_1_num_iterations;
      bool is_first_kv_iter = iter_key_start == first_key;
      bool is_last_kv_iter =
          is_tile_last(iter_key_start, p.key_tile_shape, last_key);

      auto prologueV = [&](int blockN) {
        typename MM1::Mma::IteratorB iterator_V(
            typename MM1::IteratorB::Params{p.v_strideM, problem_size_1_k},
            p.value_ptr + (iter_key_start * p.v_strideM).sum(),
            problem_size_1_k,
            problem_size_1_n_int,
            thread_id(),
            cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
        MM1::Mma::prologue(
            shared_storage.after_mm0.mm1,
            iterator_V,
            thread_id(),
            gemm_1_num_iterations);
      };

      __syncthreads(); // Need to have shared memory initialized, and `m_prime`
                       // updated from end of prev iter
      //
      // MATMUL: Q.K_t
      //
      // Computes the block-matrix product of:
      // (a) query[first_query:last_query, :]
      // with
      // (b) key[iter_key_start:iter_key_start + kKeysPerBlock]
      // and stores that into `shared_storage.si`
      //

      // Construct iterators to A and B operands
      typename MM0::IteratorA iterator_A(
          typename MM0::IteratorA::Params(p.q_strideM, problem_size_0_m),
          p.query_ptr,
          problem_size_0_m,
          problem_size_0_k_int,
          thread_id());

      typename MM0::IteratorB iterator_B(
          typename MM0::IteratorB::Params(p.k_strideM, problem_size_0_n),
          p.key_ptr + (iter_key_start * p.k_strideM).sum(),
          problem_size_0_n,
          problem_size_0_k_int,
          thread_id());

      // Construct thread-scoped matrix multiply
      typename MM0::Mma mma(
          shared_storage.mm0, thread_id(), my_warp_id, my_lane_id);

      typename MM0::Mma::FragmentC accum;

      accum.clear();

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      __syncthreads();

      if (kPreloadV) {
        prologueV(0);
      } else {
        MM1::Mma::drain_cp_asyncs();
      }

      // Neighborhood Attention masking
      Dim first_col, key_bound, row_idx;
      Dim rpb_start;
      MM0::AccumLambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {
            row_idx = map_index_to_coord((int32_t)accum_m, problem_size_0_m) +
                first_query;
            first_col = na_mask.get_window_start(row_idx);
            key_bound = na_mask.get_window_end(row_idx, first_col) - first_col;
            if constexpr (kSupportsRPB) {
              rpb_start = na_mask.get_rpb_start(row_idx);
            }
          },
          [&](int accum_m, int accum_n, int idx) {
            auto col = map_index_to_coord((int32_t)accum_n, problem_size_0_n) +
                iter_key_start;
            if constexpr (!kSupportsRPB) {
              if (
#ifdef _EXTRA_BOUND_CHECKS
                  accum_m >= problem_size_0_m_int ||
                  accum_n >= problem_size_0_n_int ||
#endif
                  !is_coord_within_bounds_nn(col - first_col, key_bound)) {
                accum[idx] =
                    -cutlass::platform::numeric_limits<accum_t>::infinity();
              }
            } else {
              // NOTE: Flipped the condition for kernels with RPB so we can
              // avoid checking whether the rpb offset goes out of bounds.
              col = col - first_col;
              auto rpb_idx = rpb_start + col;
              if (is_coord_within_bounds_nn(col, key_bound)) {
                accum[idx] = p.scale * accum[idx] +
                    (accum_t)(p.rpb_ptr[(rpb_idx * p.rpb_stride).sum()]);
              } else {
                accum[idx] =
                    -cutlass::platform::numeric_limits<accum_t>::infinity();
              }
            }
          },
          [&](int accum_m) {});

      // Update `mi` from accum stored in registers
      // Also does accum[i] <- exp(accum[i] - mi)
      iterative_softmax<typename MM0::Mma::Operator::IteratorC>(
          accum_o,
          accum,
          mi,
          m_prime,
          s_prime,
          out_rescale,
          shared_storage.addition_storage,
          my_lane_id,
          thread_id(),
          my_warp_id,
          last_kv_col,
          is_first_kv_iter,
          iteratorC_tile_offset,
          kSupportsRPB ? 1.0 : p.scale);

      // Output results to shared-memory

      MM0::B2bGemm::accumToSmem(
          shared_storage.after_mm0.si, accum, my_lane_id, output_tile_coords);

      __syncthreads();

      //
      // MATMUL: Attn . V
      // Run the matmul `attn @ V` for a block of attn and V.
      // `attn` is read from shared memory (in `shared_storage_si`)
      // `V` is read from global memory (with iterator_B)
      //

      for (int blockN = 0; blockN < nBlockN; ++blockN) {
        int gemm_k_iterations =
            (gemm_1_num_iterations + MM1::Mma::Shape::kK - 1) /
            MM1::Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add and store it in accum
        // (in registers)
        if (!kPreloadV) {
          __syncthreads(); // we share shmem between mma and epilogue
        }

        typename MM1::Mma::IteratorB iterator_V(
            typename MM1::IteratorB::Params{p.v_strideM, problem_size_1_k},
            p.value_ptr + (iter_key_start * p.v_strideM).sum(),
            problem_size_1_k,
            problem_size_1_n_int,
            thread_id(),
            cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
        typename MM1::Mma mma_pv(
            // operand A: Pij_dropped in shared memory
            shared_storage.after_mm0.si.accum_ref(),
            // operand B: shared memory staging area for Vj, which is loaded
            // from global memory
            shared_storage.after_mm0.mm1.operand_B_ref(),
            (int)thread_id(),
            (int)my_warp_id,
            (int)my_lane_id);
        mma_pv.set_prologue_done(kPreloadV);
        if (!kKeepOutputInRF) {
          accum_o.clear();
        }
        mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o);
        __syncthreads();

        if (kPreloadV && !kSingleValueIteration && blockN + 1 < nBlockN) {
          prologueV(blockN + 1);
        }

        if (!kKeepOutputInRF) {
          MM1::Mma::drain_cp_asyncs();
          DISPATCH_BOOL(
              is_first_kv_iter, kIsFirst, ([&] {
                DISPATCH_BOOL(
                    is_last_kv_iter, kIsLast, ([&] {
                      using DefaultEpilogue = typename MM1::DefaultEpilogue;
                      using DefaultOp =
                          typename MM1::DefaultConfig::EpilogueOutputOp;
                      using ElementCompute = typename DefaultOp::ElementCompute;
                      using EpilogueOutputOp = typename cutlass::epilogue::
                          thread::MemoryEfficientAttentionNormalize<
                              typename cutlass::platform::conditional<
                                  kIsLast,
                                  output_t,
                                  output_accum_t>::type,
                              output_accum_t,
                              DefaultOp::kCount,
                              typename DefaultOp::ElementAccumulator,
                              ElementCompute,
                              kIsFirst,
                              kIsLast,
                              cutlass::Array<ElementCompute, kQueriesPerBlock>>;
                      using Epilogue = typename cutlass::epilogue::threadblock::
                          EpiloguePipelined<
                              typename DefaultEpilogue::Shape,
                              typename MM1::Mma::Operator,
                              DefaultEpilogue::kPartitionsK,
                              typename cutlass::platform::conditional<
                                  kIsLast,
                                  typename MM1::OutputTileIterator,
                                  typename MM1::OutputTileIteratorAccum>::type,
                              typename DefaultEpilogue::
                                  AccumulatorFragmentIterator,
                              typename DefaultEpilogue::WarpTileIterator,
                              typename DefaultEpilogue::SharedLoadIterator,
                              EpilogueOutputOp,
                              typename DefaultEpilogue::Padding,
                              DefaultEpilogue::kFragmentsPerIteration,
                              true, // IterationsUnroll
                              typename MM1::OutputTileIteratorAccum // Read
                                                                    // iterator
                              >;

                      int col = blockN * MM1::Mma::Shape::kN;
                      auto source_iter = createOutputAccumIter(col);
                      auto dest_iter = gemm_kernel_utils::call_conditional<
                          kIsLast,
                          decltype(createOutputIter),
                          decltype(createOutputAccumIter)>::
                          apply(createOutputIter, createOutputAccumIter, col);
                      EpilogueOutputOp rescale(s_prime, out_rescale);
                      Epilogue epilogue(
                          shared_storage.epilogue_shared_storage(),
                          thread_id(),
                          my_warp_id,
                          my_lane_id);
                      epilogue(rescale, dest_iter, accum_o, source_iter);
                    }));
              }));
          if (!kSingleValueIteration) {
            __syncthreads();
          }
        }
      }
      __syncthreads(); // we modify `m_prime` after
    }

    if (kKeepOutputInRF) {
      constexpr bool kIsFirst = true;
      constexpr bool kIsLast = true;
      using DefaultEpilogue = typename MM1::DefaultEpilogue;
      using DefaultOp = typename MM1::DefaultConfig::EpilogueOutputOp;
      using ElementCompute = typename DefaultOp::ElementCompute;
      using EpilogueOutputOp =
          typename cutlass::epilogue::thread::MemoryEfficientAttentionNormalize<
              output_t, // output
              output_accum_t, // source
              DefaultOp::kCount,
              typename DefaultOp::ElementAccumulator, // accum
              output_accum_t, // compute
              kIsFirst,
              kIsLast,
              cutlass::Array<ElementCompute, kQueriesPerBlock>>;
      using Epilogue =
          typename cutlass::epilogue::threadblock::EpiloguePipelined<
              typename DefaultEpilogue::Shape,
              typename MM1::Mma::Operator,
              DefaultEpilogue::kPartitionsK,
              typename MM1::OutputTileIterator, // destination
              typename DefaultEpilogue::AccumulatorFragmentIterator,
              typename DefaultEpilogue::WarpTileIterator,
              typename DefaultEpilogue::SharedLoadIterator,
              EpilogueOutputOp,
              typename DefaultEpilogue::Padding,
              DefaultEpilogue::kFragmentsPerIteration,
              true, // IterationsUnroll
              typename MM1::OutputTileIteratorAccum // source tile
              >;
      auto dest_iter = createOutputIter(0);
      EpilogueOutputOp rescale(s_prime, out_rescale);
      Epilogue epilogue(
          shared_storage.epilogue_shared_storage(),
          thread_id(),
          warp_id(),
          lane_id());
      MM1::Mma::drain_cp_asyncs();
      epilogue(rescale, dest_iter, accum_o);
    }

    if constexpr (kStoresLSE) {
      // 7. Calculate logsumexp
      // To make the backward easier, we pad logsumexp with `inf`
      // this avoids a few bound checks, and is not more expensive during fwd
      static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
      if (p.logsumexp_ptr && thread_id() < kQueriesPerBlock) {
        // auto lse_dim = cutlass::ceil_div((int32_t)p.num_queries.prod32(),
        // kAlignLSE) * kAlignLSE;
        constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
        auto query_idx =
            map_index_to_coord((int32_t)thread_id(), problem_size_0_m);
        auto query_offset = (query_idx * p.lse_strideM).sum();
        if (is_coord_within_upper_bound(query_idx, problem_size_0_m)) {
          p.logsumexp_ptr[query_offset] = accum_t(mi[thread_id()] / kLog2e) +
              cutlass::fast_log(accum_t(s_prime[thread_id()]));
          //} else if (query_offset < lse_dim) {
          //  p.logsumexp_ptr[query_offset] =
          //      cutlass::platform::numeric_limits<accum_t>::infinity();
        }
      }
    }
  }

  template <typename WarpIteratorC>
  CUTLASS_DEVICE static void iterative_softmax(
      typename WarpIteratorC::Fragment& frag_o, // output so far
      typename WarpIteratorC::Fragment& frag,
      cutlass::Array<accum_t, kQueriesPerBlock>& mi,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& s_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& out_rescale,
      cutlass::Array<accum_t, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>&
          addition_storage,
      int8_t lane_id,
      int8_t thread_id,
      int8_t warp_id,
      int max_col,
      bool is_first,
      typename WarpIteratorC::TensorCoord const& tile_offset,
      float scaling) {
    /* Iterates on the accumulator and corresponding position on result matrix

    (1) Update `mi[r]` to the max value of the row `r`
    (2) In a second iteration do the following:
        (a) accum   <- exp(accum - mi)
        (b) m_prime <- exp(m_prime - mi)
        (c) s_prime <- s_prime * m_prime + sum(accum)

    All of this is done on registers, before we store all of this
    on shared memory for the next matmul with Value.
    */
    using Fragment = typename WarpIteratorC::Fragment;
    using LambdaIterator = typename DefaultMmaAccumLambdaIterator<
        WarpIteratorC,
        accum_t,
        kWarpSize>::Iterator;
    // Convert to `accum_t` (rather than double)
    constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E

    static_assert(kQueriesPerBlock % kNumWarpsPerBlock == 0, "");
    static constexpr int kLinesPerWarp = kQueriesPerBlock / kNumWarpsPerBlock;

    frag = cutlass::multiplies<Fragment>()(scaling * kLog2e, frag);

    auto lane_offset =
        LambdaIterator::get_lane_offset(lane_id, warp_id, tile_offset);

    // First update `mi` to the max per-row
    {
      accum_t max;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {
            max = -cutlass::platform::numeric_limits<accum_t>::infinity();
          },
          [&](int accum_m, int accum_n, int idx) {
            if (accum_n < max_col) {
              max = cutlass::fast_max(max, frag[idx]);
            }
          },
          [&](int accum_m) {
            // Having 4x atomicMax seems faster than reduce within warp
            // first...
            atomicMaxFloat(&mi[accum_m], max);
          });
    }

    // Make sure we all share the update values for `mi`
    __syncthreads();

    // Doing this `exp` is quite expensive. Let's
    // split it across the warps
    bool restore_mi_to_minus_inf = false;
    if (lane_id < kLinesPerWarp) {
      int id = warp_id * kLinesPerWarp + lane_id;
      auto m_prime_id = m_prime[id];
      auto mi_id = mi[id];
      bool changed = m_prime_id < mi_id; // `false` if both are -inf
      if (changed) {
        auto m_prime_exp = exp2f(m_prime_id - mi_id);
        out_rescale[id] = m_prime_exp;
        s_prime[id] *= m_prime_exp;
      } else {
        out_rescale[id] = 1.0f;
      }
    }
    __syncthreads(); // Update output fragments
    if (kKeepOutputInRF && !is_first) {
      accum_t line_rescale;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { line_rescale = out_rescale[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag_o[idx] = frag_o[idx] * line_rescale;
          },
          [&](int accum_m) {});
    }
    // Update accum_m, accum_n, ...
    {
      accum_t mi_row, total_row;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { mi_row = mi[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag[idx] =
                (accum_n < max_col) ? exp2f(frag[idx] - mi_row) : accum_t(0.0);
          },
          [&](int accum_m) {});
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { total_row = 0.0; },
          [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
          [&](int accum_m) {
            if (LambdaIterator::reduceSameRow(
                    lane_id, total_row, [](accum_t a, accum_t b) {
                      return a + b;
                    })) {
              // NOTE: we could atomically add `total_row` to `s_prime`, but
              // it's faster (and deterministic) to avoid atomics here
              addition_storage
                  [accum_m + kQueriesPerBlock * tile_offset.column()] =
                      total_row;
            }
          });
    }
    __syncthreads();
    if (lane_id < kLinesPerWarp) {
      int id = warp_id * kLinesPerWarp + lane_id;
      accum_t total_row = s_prime[id];
      if (restore_mi_to_minus_inf) {
        // Restore `mi`, see above when we set `restore_mi_to_minus_inf=true`
        mi[id] = -cutlass::platform::numeric_limits<accum_t>::infinity();
      } else {
        m_prime[id] = mi[id];
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < MM0::MmaCore::WarpCount::kN; ++i) {
        total_row += addition_storage[id + kQueriesPerBlock * i];
      }
      s_prime[id] = total_row;
    }
  }

  static CUTLASS_DEVICE int8_t lane_id() {
    return threadIdx.x;
  }
  static CUTLASS_DEVICE int8_t warp_id() {
    return threadIdx.y;
  }
  static CUTLASS_DEVICE int16_t thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x;
  }
};

} // namespace fna
} // namespace cuda
} // namespace natten
