/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    Example usage:
      $ ./examples/88_hopper_fmha/88_hopper_fmha \
            --b=2048 --h=2048 --d=2048 --q=2048 --k=2048
*/

#include <iostream>
#include <cuda.h>

#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include "cutlass/cutlass.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "reference.hpp"
#include "bwd_reference.hpp"
#include "error.hpp"

#include "flash.h"

#include "flash_fwd_sm80_test.cu"
#include "flash_bwd_sm80_test.cu"

#include "flash_fwd_launch_template.h"
#include "flash_bwd_launch_template.h"

using namespace cute;
using namespace natten::cuda::flash;

///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void print_O_and_LSE(void* block_O, void* block_ref_O,
    void* block_LSE, void* block_ref_LSE){
    if (thread0()){
      print("============= DEBUG ============"); print("\n");
      print("block_O[0]="); print(*((cutlass::half_t*)block_O + 0)); print("\n");
      print("block_O[1]="); print(*((cutlass::half_t*)block_O + 1)); print("\n");
      print("block_O[2]="); print(*((cutlass::half_t*)block_O + 2)); print("\n");
      print("block_ref_O[0]="); print(*((cutlass::half_t*)block_ref_O + 0)); print("\n");
      print("block_ref_O[1]="); print(*((cutlass::half_t*)block_ref_O + 1)); print("\n");
      print("block_ref_O[2]="); print(*((cutlass::half_t*)block_ref_O + 2)); print("\n");
      print("block_LSE[0]="); print(*((float*)block_LSE + 0)); print("\n");
      print("block_LSE[1]="); print(*((float*)block_LSE + 1)); print("\n");
      print("block_LSE[2]="); print(*((float*)block_LSE + 2)); print("\n");
      print("block_ref_LSE[0]="); print(*((float*)block_ref_LSE + 0)); print("\n");
      print("block_ref_LSE[1]="); print(*((float*)block_ref_LSE + 1)); print("\n");
      print("block_ref_LSE[2]="); print(*((float*)block_ref_LSE + 2)); print("\n");
    }
}

__global__ void print_dQ_dK_dV(void* block_dQ, void* block_ref_dQ,
                             void* block_dK, void* block_ref_dK,
                             void* block_dV, void* block_ref_dV) {
    if (thread0()) {
        print("============= DEBUG ============"); print("\n");

        // dQ
        print("block_dQ[0]="); print(*((cutlass::half_t*)block_dQ + 0)); print("\n");
        print("block_dQ[1]="); print(*((cutlass::half_t*)block_dQ + 1)); print("\n");
        print("block_dQ[2]="); print(*((cutlass::half_t*)block_dQ + 2)); print("\n");

        print("block_ref_dQ[0]="); print(*((cutlass::half_t*)block_ref_dQ + 0)); print("\n");
        print("block_ref_dQ[1]="); print(*((cutlass::half_t*)block_ref_dQ + 1)); print("\n");
        print("block_ref_dQ[2]="); print(*((cutlass::half_t*)block_ref_dQ + 2)); print("\n");

        // dK
        print("block_dK[0]="); print(*((cutlass::half_t*)block_dK + 0)); print("\n");
        print("block_dK[1]="); print(*((cutlass::half_t*)block_dK + 1)); print("\n");
        print("block_dK[2]="); print(*((cutlass::half_t*)block_dK + 2)); print("\n");

        print("block_ref_dK[0]="); print(*((cutlass::half_t*)block_ref_dK + 0)); print("\n");
        print("block_ref_dK[1]="); print(*((cutlass::half_t*)block_ref_dK + 1)); print("\n");
        print("block_ref_dK[2]="); print(*((cutlass::half_t*)block_ref_dK + 2)); print("\n");

        // dV
        print("block_dV[0]="); print(*((cutlass::half_t*)block_dV + 0)); print("\n");
        print("block_dV[1]="); print(*((cutlass::half_t*)block_dV + 1)); print("\n");
        print("block_dV[2]="); print(*((cutlass::half_t*)block_dV + 2)); print("\n");

        print("block_ref_dV[0]="); print(*((cutlass::half_t*)block_ref_dV + 0)); print("\n");
        print("block_ref_dV[1]="); print(*((cutlass::half_t*)block_ref_dV + 1)); print("\n");
        print("block_ref_dV[2]="); print(*((cutlass::half_t*)block_ref_dV + 2)); print("\n");
    }
}

__global__ void print_tensors(void** blocks, int num_tensors) {
    if (thread0()) {
        print("============= DEBUG ============"); print("\n");

        for (int i = 0; i < num_tensors; ++i) {
            print("Tensor "); print(i); print(":\n");

            print("  [0]=");
            print(*((cutlass::half_t*)blocks[i] + 0)); print("\n");

            print("  [1]=");
            print(*((cutlass::half_t*)blocks[i] + 1)); print("\n");

            print("  [2]=");
            print(*((cutlass::half_t*)blocks[i] + 2)); print("\n");
        }
    }
}

/// Command line options parsing
struct Options {

  bool help;
  bool error;

  int b, h, q, k, d;
  int iterations;
  bool verify;
  bool verbose;
  bool causal;
  bool residual;
  bool bwd;

  Options():
    help(false),
    error(false),
    b(16), h(16), q(1024), k(1024), d(128),
    iterations(20), verify(true),
    causal(false), residual(false), bwd(false), verbose(false)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    Options defaults;

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("d", d, defaults.d);
    cmd.get_cmd_line_argument("h", h, -1);
    if (h == -1) h = 2048 / d;

    cmd.get_cmd_line_argument("q", q, -1);
    cmd.get_cmd_line_argument("k", k, -1);
    if (q == -1) q = k;
    if (k == -1) k = q;
    if (q == -1 && k == -1) q = k = defaults.q;

    cmd.get_cmd_line_argument("b", b, -1);
    if (b == -1) b = 16384 / k;
    if (b == 0) b = 1;

    cmd.get_cmd_line_argument("iterations", iterations, defaults.iterations);
    verify = true;
    verbose = cmd.check_cmd_line_flag("verbose");

    std::string mask;
    cmd.get_cmd_line_argument<std::string>("mask", mask, "");
    if (mask == "no" || mask == "") {
      causal = residual = false;
    }
    else if (mask == "causal") {
      residual = false;
      causal = true;
    }
    else if (mask == "residual") {
      residual = true;
      causal = false;
    }

    bwd = cmd.check_cmd_line_flag("bwd");
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "88_hopper_fmha\n\n"
      << "  This example showcases the use of CUTLASS's collective operation builders to easily construct\n"
      << "  fused multi-head attention forward-pass kernels targeting NVIDIA's Hopper architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --b=<int>                   Sets the B extent\n"
      << "  --h=<int>                   Sets the H extent\n"
      << "  --q=<int>                   Sets the Q extent\n"
      << "  --k=<int>                   Sets the K extent\n"
      << "  --d=<int>                   Sets the D extent\n"
      << "  --iterations=<int>          Benchmarking iterations\n"
      << "  --verify                    Verify results\n"
      << "  --verbose                   Print smem and execution time per kernel\n"
      << "  --mask=<no|residual|causal> Enables masking\n"
      << "  --bwd                       Runs the backwards pass\n"
      << "\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023, bool init_one=false) {

  if (init_one) {
    cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, (Element) 1, (Element) 1);
  } else {
    cutlass::reference::device::BlockFillRandomGaussian(
      block.get(), block.size(), seed, (Element) 0, (Element) 1);
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct ExampleResult {
  bool passed = false;
  bool verified = false;
  float runtime_ms = 0;
  double tflops_s = 0;
  size_t smem_size = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// #if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

struct DefaultFusion {
  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return ceil_div(get<3>(problem_size), get<1>(tile_shape));
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_masked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return 0;
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void before_softmax(
    AccQK& acc_qk,
    IndexQK const& index_qk,
    ProblemSize const& problem_size

  ) {
    return;
  }
};


template<
  class TileShape,
  class... KernelOptions
>
struct FwdRunner {

#ifdef FP8
  using Element = cutlass::float_e4m3_t;
  using ElementAccumulatorQK = find_option_t<Tag::kAccQK, float, KernelOptions...>;
#else
  using Element = cutlass::half_t;
  using ElementAccumulatorQK = float;
#endif

  using ElementAccumulatorPV = float;

  // B H Q K D
  using ProblemShapeType = cute::tuple<int, int, int, int, int>;

  
  using StrideQ = cute::tuple<int, _1, cute::tuple<int, int>>;  // Q D (B H)
  using StrideK = cute::tuple<int, _1, cute::tuple<int, int>>;  // K D (B H)
  using StrideV = std::conditional_t<sizeof(Element) == 1,
    cute::tuple<_1, int, cute::tuple<int, int>>,
    cute::tuple<int, _1, cute::tuple<int, int>>>;  // K D (B H)
  using StrideO = cute::tuple<int, _1, cute::tuple<int, int>>; // Q D (B H)
  using StrideLSE = cute::tuple<_1, cute::tuple<int, int>>; // Q (B H)

  // using Operation = cutlass::device::Universal<
  //   typename cutlass::fmha::kernel::FmhaBuilder<
  //     Element, ElementAccumulatorQK, ElementAccumulatorPV,
  //     TileShape, StrideQ, StrideK, StrideV,
  //     ActiveFusion, DispatchPolicy, KernelOptions...
  //   >::Kernel>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<Element> block_Q;
  cutlass::DeviceAllocation<Element> block_K;
  cutlass::DeviceAllocation<Element> block_V;
  cutlass::DeviceAllocation<Element> block_O;
  cutlass::DeviceAllocation<ElementAccumulatorPV> block_LSE;
  cutlass::DeviceAllocation<Element> block_ref_O;
  cutlass::DeviceAllocation<ElementAccumulatorPV> block_ref_LSE;

  //
  // Methods
  //
  bool verify(const ProblemShapeType& problem_size) {
    auto [B, H, Q, K, D] = problem_size;

    Tensor mQ = make_tensor(make_gmem_ptr(block_Q.get()),
      make_shape(Q, D, make_shape(B, H)),
      stride_Q);

    Tensor mK = make_tensor(make_gmem_ptr(block_K.get()),
      make_shape(K, D, make_shape(B, H)),
      stride_K);

    Tensor mV = make_tensor(make_gmem_ptr(block_V.get()),
      make_shape(K, D, make_shape(B, H)),
      stride_V);

    Tensor mO = make_tensor(make_gmem_ptr(block_ref_O.get()),
      make_shape(Q, D, make_shape(B, H)),
      stride_O);

    Tensor mLSE = make_tensor(make_gmem_ptr(block_ref_LSE.get()),
      make_shape(Q, make_shape(B, H)),
      stride_LSE);

    fmha_reference(problem_size, mQ, mK, mV, mO, mLSE, DefaultFusion{});
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    const double kMaxDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-2;
    const double kMeanDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-3;

    dim3 blk(1, 1, 1);
    dim3 grd(1, 1, 1);
    print_O_and_LSE<<<grd, blk>>>(block_O.get(), block_ref_O.get(), block_LSE.get(),
        block_ref_LSE.get());
    // Check if output from CUTLASS kernel and reference kernel are equal or not
    double max_diff = 0;
    double mean_diff = 0;
    reference_abs_diff(block_O, block_ref_O, max_diff, mean_diff);
    bool passed_O = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_O) {
      std::cerr << "failed O: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(block_LSE, block_ref_LSE, max_diff, mean_diff);
    bool passed_LSE = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if ( ! passed_LSE) {
      std::cerr << "failed LSE: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    std::cout << "passed_O: " << passed_O << std::endl;
    std::cout << "passed_LSE: " << passed_LSE << std::endl;

    // if (!(passed_O && passed_LSE)){
    // }

    return passed_O && passed_LSE;
  }


  void initialize_stride(cute::tuple<int, int, int> const& shape, cute::tuple<_1, cute::tuple<int, int>>& stride) {
    auto [B, H, Q] = shape;
    stride = make_stride(_1{}, make_stride(H*Q, Q));
  }

  void initialize_stride(cute::tuple<int, int, int, int> const& shape, cute::tuple<int, _1, cute::tuple<int, int>>& stride) {
    auto [B, H, Q, D] = shape;
    stride = make_stride(D, _1{}, make_stride(H*Q*D, Q*D));
  }

  void initialize_stride(cute::tuple<int, int, int, int> const& shape, cute::tuple<_1, int, cute::tuple<int, int>>& stride) {
    auto [B, H, Q, D] = shape;
    stride = make_stride(_1{}, Q, make_stride(H*Q*D, Q*D));
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto [B, H, Q, K, D] = problem_size;
    D = cutlass::round_up(D, 8); // Alignment

    auto shape_QO = cute::make_shape(B, H, Q, D);
    auto shape_KV = cute::make_shape(B, H, K, D);
    auto shape_LSE = cute::make_shape(B, H, Q);

    initialize_stride(shape_QO, stride_Q);
    initialize_stride(shape_KV, stride_K);
    initialize_stride(shape_KV, stride_V);
    initialize_stride(shape_QO, stride_O);
    initialize_stride(shape_LSE, stride_LSE);

    block_Q.reset(size(shape_QO));
    block_K.reset(size(shape_KV));
    block_V.reset(size(shape_KV));
    block_O.reset(size(shape_QO));
    block_LSE.reset(size(shape_LSE));
    block_ref_O.reset(size(shape_QO));
    block_ref_LSE.reset(size(shape_LSE));

    initialize_block(block_Q, seed + 2023, false);
    initialize_block(block_K, seed + 2022, false);
    initialize_block(block_V, seed + 2021, false);
  }

  // ExampleResult run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
  ExampleResult run(const Options& options) {
    ProblemShapeType problem_size = ProblemShapeType{options.b, options.h, options.q, options.k, options.d};

    initialize(problem_size);

    Flash_fwd_params params = set_flash_fwd_params_for_testing(
      problem_size,
      block_Q.get(),
      block_K.get(),
      block_V.get(),
      block_O.get(),
      block_LSE.get(),
      stride_Q,
      stride_K,
      stride_V,
      stride_O,
      stride_LSE,
      1.0 / sqrt(double(options.d)) /* softmax_scale */ 
    );
    // print_strides(params);

    ExampleResult example_result;
    example_result.smem_size = 0; // Operation::Kernel::SharedStorageSize;

    cudaError_t result = cudaDeviceSynchronize();
    cutlass::Status status;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamDefault);


    // What needs to be done here:
    // Initialize and run a flash attention kernel and check if it correctly achieves the same throughput

    // Test scope:
    // Arch: SM80
    // Dtype: FP16
    // HeadDim: 128

    auto flash_fwd_fn = run_flash_fwd<
      /* Arch= */ 80,
      /* kHeadDim= */ 128,
      /* kHeadDimV= */ 128,
      /* kBlockM= */ 128,
      /* kBlockN= */ 112,
      /* T= */ cutlass::half_t,
      /* T_out= */ cutlass::half_t,
      /* PackGQA= */ false, 
      /* V_colmajor= */ false
      // /* Is_causal= */ false, 
      // /* Is_local= */ false,
      // /* Has_softcap= */ false, 
      // /* Varlen= */ false,
      // /* AppendKV= */ false,
      // /* PagedKVNonTMA= */ false, 
      // /* HasQv= */ false,
      // /* Split= */ false, 
    >;

    cudaEvent_t events[2];

    for (auto & event : events) {
      result = cudaEventCreate(&event);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result) << std::endl;
        return example_result;
      }
    }

    // Record an event at the start of a series of GEMMs
    result = cudaEventRecord(events[0]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    for (int i = 0; i < options.iterations; i++) {
      // status = op.run();
      flash_fwd_fn(params, stream);
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return example_result;
      }
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMMs are complete
    result = cudaEventRecord(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Wait for work on the device to complete.
    result = cudaEventSynchronize(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    runtime_ms /= static_cast<float>(options.iterations);

    double flops = 4.0;
    flops *= static_cast<double>(get<0>(problem_size));
    flops *= static_cast<double>(get<1>(problem_size));
    flops *= static_cast<double>(get<2>(problem_size));
    flops *= static_cast<double>(get<3>(problem_size));
    flops *= static_cast<double>(get<4>(problem_size));
    double tflops_s = flops * 1e-12 /*tera*/ / (runtime_ms * 1e-3 /*ms*/);
    example_result.tflops_s = tflops_s;
    example_result.runtime_ms = runtime_ms;

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Verify that the result is correct
    bool passed = true;
    if (options.verify) {
      passed = verify(problem_size);
      if (passed) example_result.verified = true;
    }
    
    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
      return example_result;
    }

    example_result.passed = true;

    return example_result;
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class TileShape,
  class... KernelOptions
>
struct BwdRunner {

  using Element = cutlass::half_t;
  using ElementAccumulator = float;

  // B H Q K D
  using ProblemShapeType = cute::tuple<int, int, int, int, int>;

  // using Operation = cutlass::fmha::device::FmhaBwd<Element, ElementAccumulator, TileShape, ActiveFusion, KernelOptions...>;
  
  // Just like forward
  using StrideQ = cute::tuple<int, int, int, _1>; // B H Q D
  using StrideK = cute::tuple<int, int, int, _1>; // B H K D
  using StrideV = cute::tuple<int, int, int, _1>; // B H K D
  using StrideO = cute::tuple<int, int, int, _1>; // B H Q D
  using StrideLSE = cute::tuple<int, int, _1>; // B H Q

  // Backwards specific
  using StrideDQ = cute::tuple<int, int, int, _1>; // B H Q D
  using StrideDK = cute::tuple<int, int, int, _1>; // B H K D
  using StrideDV = cute::tuple<int, int, int, _1>; // B H K D
  using StrideDO = cute::tuple<int, int, int, _1>; // B H Q D

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;

  StrideDQ stride_dQ;
  StrideDK stride_dK;
  StrideDV stride_dV;
  StrideDO stride_dO;

  uint64_t seed = 0;

  cutlass::DeviceAllocation<Element> block_Q;
  cutlass::DeviceAllocation<Element> block_K;
  cutlass::DeviceAllocation<Element> block_V;
  cutlass::DeviceAllocation<Element> block_O;
  cutlass::DeviceAllocation<ElementAccumulator> block_LSE;
  cutlass::DeviceAllocation<int> block_dQ_semaphore;

  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 128;

  // (aditya) NOTE: Specifically for Flash Attention
  cutlass::DeviceAllocation<ElementAccumulator> block_LSE_log2;
  cutlass::DeviceAllocation<ElementAccumulator> block_dsoftmax_sum;
  cutlass::DeviceAllocation<ElementAccumulator> block_dQ_accum;

  cutlass::DeviceAllocation<Element> block_dQ;
  cutlass::DeviceAllocation<Element> block_dK;
  cutlass::DeviceAllocation<Element> block_dV;
  cutlass::DeviceAllocation<Element> block_dO;

  cutlass::DeviceAllocation<Element> block_ref_dQ;
  cutlass::DeviceAllocation<Element> block_ref_dK;
  cutlass::DeviceAllocation<Element> block_ref_dV;

  //
  // Methods
  //
  bool verify(const ProblemShapeType& problem_size) {
    auto [B, H, Q, K, D] = problem_size;

    Tensor mQ = make_tensor(make_gmem_ptr(block_Q.get()),
      make_shape(Q, D, make_shape(B, H)),
      make_stride(get<2>(stride_Q), get<3>(stride_Q), make_stride(get<0>(stride_Q), get<1>(stride_Q))));

    Tensor mK = make_tensor(make_gmem_ptr(block_K.get()),
      make_shape(K, D, make_shape(B, H)),
      make_stride(get<2>(stride_K), get<3>(stride_K), make_stride(get<0>(stride_K), get<1>(stride_K))));

    Tensor mV = make_tensor(make_gmem_ptr(block_V.get()),
      make_shape(K, D, make_shape(B, H)),
      make_stride(get<2>(stride_V), get<3>(stride_V), make_stride(get<0>(stride_V), get<1>(stride_V))));

    Tensor mO = make_tensor(make_gmem_ptr(block_O.get()),
      make_shape(Q, D, make_shape(B, H)),
      make_stride(get<2>(stride_O), get<3>(stride_O), make_stride(get<0>(stride_O), get<1>(stride_O))));

    Tensor mLSE = make_tensor(make_gmem_ptr(block_LSE.get()),
      make_shape(Q, make_shape(B, H)),
      make_stride(get<2>(stride_LSE), make_stride(get<0>(stride_LSE), get<1>(stride_LSE))));

    Tensor mDQ = make_tensor(make_gmem_ptr(block_ref_dQ.get()),
      make_shape(Q, D, make_shape(B, H)),
      make_stride(get<2>(stride_dQ), get<3>(stride_dQ), make_stride(get<0>(stride_dQ), get<1>(stride_dQ))));

    Tensor mDK = make_tensor(make_gmem_ptr(block_ref_dK.get()),
      make_shape(K, D, make_shape(B, H)),
      make_stride(get<2>(stride_dK), get<3>(stride_dK), make_stride(get<0>(stride_dK), get<1>(stride_dK))));

    Tensor mDV = make_tensor(make_gmem_ptr(block_ref_dV.get()),
      make_shape(K, D, make_shape(B, H)),
      make_stride(get<2>(stride_dV), get<3>(stride_dV), make_stride(get<0>(stride_dV), get<1>(stride_dV))));

    Tensor mDO = make_tensor(make_gmem_ptr(block_dO.get()),
      make_shape(Q, D, make_shape(B, H)),
      make_stride(get<2>(stride_dO), get<3>(stride_dO), make_stride(get<0>(stride_dO), get<1>(stride_dO))));

    fmha_bwd_reference(problem_size, mQ, mK, mV, mO, mLSE, mDO, mDQ, mDK, mDV, DefaultFusion{});

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }
    // dim3 blk(1, 1, 1);
    // dim3 grd(1, 1, 1);
    // print_dQ_dK_dV<<<grd, blk>>>(block_dQ.get(), block_ref_dQ.get(),
    //                              block_dK.get(), block_ref_dK.get(),
    //                              block_dV.get(), block_ref_dV.get());
    // 
    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    double max_diff = 0;
    double mean_diff = 0;
    reference_abs_diff(block_dQ, block_ref_dQ, max_diff, mean_diff);
    std::cout << "dQ max_diff=" << max_diff << " mean_diff=" << mean_diff << std::endl;
    bool passed_dQ = (max_diff < 1e-2) && (mean_diff < 1e-3);
    if (! passed_dQ) {
      std::cerr << "failed dQ: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(block_dK, block_ref_dK, max_diff, mean_diff);
    std::cout << "dK max_diff=" << max_diff << " mean_diff=" << mean_diff << std::endl;
    bool passed_dK = (max_diff < 1e-2) && (mean_diff < 1e-3);
    if (! passed_dK) {
      std::cerr << "failed dK: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(block_dV, block_ref_dV, max_diff, mean_diff);
    std::cout << "dV max_diff=" << max_diff << " mean_diff=" << mean_diff << std::endl;
    bool passed_dV = (max_diff < 1e-2) && (mean_diff < 1e-3);
    if (! passed_dV) {
      std::cerr << "failed dV: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    std::cout << "passed_dQ: " << passed_dQ << std::endl;
    std::cout << "passed_dK: " << passed_dK << std::endl;
    std::cout << "passed_dV: " << passed_dV << std::endl;

    if (!(passed_dQ && passed_dK && passed_dV)) {
      dim3 blk(1, 1, 1);
      dim3 grd(1, 1, 1);
      print_dQ_dK_dV<<<grd, blk>>>(block_dQ.get(), block_ref_dQ.get(),
                                 block_dK.get(), block_ref_dK.get(),
                                 block_dV.get(), block_ref_dV.get());
    }



    return passed_dQ && passed_dK && passed_dV;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto [B, H, Q, K, D] = problem_size;
    // Q = cutlass::round_up(Q, 8);  // Alignment
    auto Q_rounded = cutlass::round_up(Q, kBlockM);

    auto shape_QO = cute::make_shape(B, H, Q, D);
    auto shape_KV = cute::make_shape(B, H, K, D);
    auto shape_LSE = cute::make_shape(B, H, Q);
    auto shape_LSE_rounded = cute::make_shape(B, H, Q_rounded);
    auto shape_dQ_accum = cute::make_shape(B, H, Q_rounded * D);
    auto shape_dQ_semaphore = cute::make_shape((Q + kBlockM - 1) / kBlockM, B, H);

    stride_Q = cute::compact_row_major(shape_QO);
    stride_K = cute::compact_row_major(shape_KV);
    stride_V = cute::compact_row_major(shape_KV);
    stride_O = cute::compact_row_major(shape_QO);
    stride_LSE = cute::compact_row_major(shape_LSE);

    stride_dQ = stride_Q;
    stride_dK = stride_K;
    stride_dV = stride_V;
    stride_dO = stride_O;

    block_Q.reset(size(shape_QO));
    block_K.reset(size(shape_KV));
    block_V.reset(size(shape_KV));
    block_O.reset(size(shape_QO));
    block_LSE.reset(size(shape_LSE));
    block_LSE_log2.reset(size(shape_LSE_rounded));
    block_dsoftmax_sum.reset(size(shape_LSE_rounded));
    block_dQ_accum.reset(size(shape_dQ_accum));
    block_dQ_semaphore.reset(size(shape_dQ_semaphore));

    block_dQ.reset(size(shape_QO));
    block_dK.reset(size(shape_KV));
    block_dV.reset(size(shape_KV));
    block_dO.reset(size(shape_QO));

    block_ref_dQ.reset(size(shape_QO));
    block_ref_dK.reset(size(shape_KV));
    block_ref_dV.reset(size(shape_KV));

    initialize_block(block_Q, seed + 2023, false);
    initialize_block(block_K, seed + 2022, false);
    initialize_block(block_V, seed + 2021, false);
    initialize_block(block_dO, seed + 2020, false);

    Tensor mQ = make_tensor(make_gmem_ptr(block_Q.get()),
      make_shape(Q, D, make_shape(B, H)),
      make_stride(get<2>(stride_Q), get<3>(stride_Q), make_stride(get<0>(stride_Q), get<1>(stride_Q))));

    Tensor mK = make_tensor(make_gmem_ptr(block_K.get()),
      make_shape(K, D, make_shape(B, H)),
      make_stride(get<2>(stride_K), get<3>(stride_K), make_stride(get<0>(stride_K), get<1>(stride_K))));

    Tensor mV = make_tensor(make_gmem_ptr(block_V.get()),
      make_shape(K, D, make_shape(B, H)),
      make_stride(get<2>(stride_V), get<3>(stride_V), make_stride(get<0>(stride_V), get<1>(stride_V))));

    Tensor mO = make_tensor(make_gmem_ptr(block_O.get()),
      make_shape(Q, D, make_shape(B, H)),
      make_stride(get<2>(stride_O), get<3>(stride_O), make_stride(get<0>(stride_O), get<1>(stride_O))));

    Tensor mLSE = make_tensor(make_gmem_ptr(block_LSE.get()),
      make_shape(Q, make_shape(B, H)),
      make_stride(get<2>(stride_LSE), make_stride(get<0>(stride_LSE), get<1>(stride_LSE))));

    fmha_reference(problem_size, mQ, mK, mV, mO, mLSE, DefaultFusion{});
  }

  ExampleResult run(const Options& options) {
    ProblemShapeType problem_size = ProblemShapeType{options.b, options.h, options.q, options.k, options.d};

    initialize(problem_size);

    // Set params here
    Flash_bwd_params params = set_flash_bwd_params_for_testing(
      problem_size,
      block_Q.get(),
      block_K.get(),
      block_V.get(),
      block_O.get(),
      block_LSE.get(),
      block_dQ.get(),
      block_dK.get(),
      block_dV.get(),
      block_dO.get(),
      block_dsoftmax_sum.get(),
      block_LSE_log2.get(),
      block_dQ_accum.get(),
      block_dQ_semaphore.get(),
      stride_Q,
      stride_K,
      stride_V,
      stride_O,
      stride_LSE,
      (float) 1.0 / sqrt(double(options.d)) /* softmax_scale */ 
    );

    ExampleResult example_result;

    example_result.smem_size = 0; //Operation::Operation::Kernel::SharedStorageSize;

    cudaError_t result = cudaDeviceSynchronize();
    cutlass::Status status;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamDefault);

    // auto flash_bwd_fn = run_flash_bwd<
    //   /* Arch= */ 80,
    //   /* kHeadDim= */ 128,
    //   /* kBlockM= */ kBlockM,
    //   /* kBlockN= */ kBlockN,
    //   /* Element= */ cutlass::half_t,
    //   /* Is_causal= */ false,
    //   /* Is_local= */ false,
    //   /* Has_softcap= */ false,
    //   /* Varlen= */ false,
    //   /* Deterministic= */ false,
    //   /* GQA= */ false,
    //   /* Stages_dO= */ 2,
    //   /* Stages_dS_or_QSm80= */ 2,
    //   /* SdP_swapAB= */ true,
    //   /* dKV_swapAB= */ false,
    //   /* dQ_swapAB= */ false,
    //   /* NumMmaWarpGroups= */ 2,
    //   /* AtomLayoutMSdP= */ 2,
    //   /* AtomLayoutNdKV= */ 2,
    //   /* AtomLayoutMdQ= */ 2,
    //   /* V_in_regs= */ false
    // >;

    auto flash_bwd_fn = run_flash_bwd<
      /* Arch= */ 80,
      /* kHeadDim= */ 128,
      /* kBlockM= */ kBlockM,
      /* kBlockN= */ kBlockN,
      /* Element= */ cutlass::half_t,
      /* Deterministic= */ false,
      /* GQA= */ false,
      /* Stages_dO= */ 2,
      /* Stages_dS_or_QSm80= */ 2,
      /* SdP_swapAB= */ true,
      /* dKV_swapAB= */ false,
      /* dQ_swapAB= */ false,
      /* NumMmaWarpGroups= */ 2,
      /* AtomLayoutMSdP= */ 2,
      /* AtomLayoutNdKV= */ 2,
      /* AtomLayoutMdQ= */ 2,
      /* V_in_regs= */ false
    >;

    cudaMemset(block_dQ.get(), 0, block_dQ.size() * sizeof(Element));
    cudaMemset(block_dQ_accum.get(), 0, block_dQ_accum.size() * sizeof(ElementAccumulator));
    cudaMemset(block_dK.get(), 0, block_dK.size() * sizeof(Element));
    cudaMemset(block_dV.get(), 0, block_dV.size() * sizeof(Element));

    cudaEvent_t events[2];

    for (auto & event : events) {
      result = cudaEventCreate(&event);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result) << std::endl;
        return example_result;
      }
    }

    // Record an event at the start of a series of GEMMs
    result = cudaEventRecord(events[0]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    for (int i = 0; i < options.iterations; i++) {
      flash_bwd_fn(params, stream);
      // verify(problem_size);
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return example_result;
      }
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMMs are complete
    result = cudaEventRecord(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Wait for work on the device to complete.
    result = cudaEventSynchronize(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    cudaMemset(block_dQ.get(), 0, block_dQ.size() * sizeof(Element));
    cudaMemset(block_dQ_accum.get(), 0, block_dQ.size() * sizeof(ElementAccumulator));
    cudaMemset(block_dK.get(), 0, block_dK.size() * sizeof(Element));
    cudaMemset(block_dV.get(), 0, block_dV.size() * sizeof(Element));
    flash_bwd_fn(params, stream);
    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "cudaDeviceSynchronize() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }
    

    // Measure elapsed runtime
    float runtime_ms = 0;
    result = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    runtime_ms /= static_cast<float>(options.iterations);

    double flops = 10.0;
    flops *= static_cast<double>(get<0>(problem_size));
    flops *= static_cast<double>(get<1>(problem_size));
    flops *= static_cast<double>(get<2>(problem_size));
    flops *= static_cast<double>(get<3>(problem_size));
    flops *= static_cast<double>(get<4>(problem_size));
    double tflops_s = flops * 1e-12 /*tera*/ / (runtime_ms * 1e-3 /*ms*/);
    example_result.tflops_s = tflops_s;
    example_result.runtime_ms = runtime_ms;

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Verify that the result is correct
    bool passed = true;
    if (options.verify) {
      passed = verify(problem_size);
      if (passed) example_result.verified = true;
    }
    
    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
      return example_result;
    }

    example_result.passed = true;

    return example_result;
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, ExampleResult result, bool verbose) {
  std::ios fmt(nullptr);
  fmt.copyfmt(std::cout);
  std::cout << (result.passed ? (result.verified ? " [OK]  " : " [--] ") : "[FAIL] ");
  std::cout << std::setw(32) << std::left << description;
  std::cout.copyfmt(fmt);
  std::cout <<std::endl;
  std::cout << " \t " << "Throughput: " << result.tflops_s << " TFLOPS/s" << std::endl;
  std::cout << " \t " << "Runtime:    " << result.runtime_ms << " ms" << std::endl;
  std::cout << " \t "<<  "SMEM size:  " << result.smem_size << "b" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// using KernelTma = cutlass::gemm::KernelTma;
// using KernelCooperative = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
// using KernelPingpong = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

///////////////////////////////////////////////////////////////////////////////////////////////////

void run_fwd_32(Options const & options) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    FwdRunner<decltype(shape), decltype(kernel_options)...> runner;
    auto result = runner.run(options);
    print_result(name, result, options.verbose);
  };

  using HeadDim = _32;

  run(Shape< _64, _128, HeadDim>{}, "tma 64x128x32");
  run(Shape< _128, _64, HeadDim>{}, "tma ws cooperative 128x64x32");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void run_fwd_64(Options const & options) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    FwdRunner<decltype(shape), decltype(kernel_options)...> runner;
    auto result = runner.run(options);
    print_result(name, result, options.verbose);
  };

  using HeadDim = _64;

  run(Shape< _64, _128, HeadDim>{}, "tma 64x128x64");
  run(Shape< _128, _64, HeadDim>{}, "tma ws cooperative 128x64x64");
  run(Shape< _128, _64, HeadDim>{}, "tma ws ping-pong 128x64x64");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void run_fwd_128(Options const & options) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    FwdRunner<decltype(shape), decltype(kernel_options)...> runner;
    auto result = runner.run(options);
    print_result(name, result, options.verbose);
  };

  using HeadDim = _128;

  run(Shape<_128, _128, HeadDim>{}, "tma ws cooperative 128x128x128");
  // run(Shape<_128, _128, HeadDim>{}, "tma ws ping-pong 128x128x128");
#ifdef FP8
  // run(Shape<_128, _256, HeadDim>{}, "tma ws cooperative 128x256x128 acc fp16", Option<Tag::kAccQK, cutlass::half_t>{});
  // run(Shape<_128, _256, HeadDim>{}, "tma ws cooperative 128x256x128 acc fp32");
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void run_fwd_256(Options const & options) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    FwdRunner<decltype(shape), decltype(kernel_options)...> runner;
    auto result = runner.run(options);
    print_result(name, result, options.verbose);
  };

  using HeadDim = _256;

#ifdef FP8
  run(Shape<_128, _128, HeadDim>{}, "tma ws cooperative 128x128x256");
  run(Shape<_128, _128, HeadDim>{}, "tma ws ping-pong 128x128x256");
#else
  run(Shape<_128, _64, HeadDim>{}, "tma ws cooperative 128x64x256");
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void run_bwd_32(Options const & options) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    BwdRunner<decltype(shape), decltype(kernel_options)...> runner;
    auto result = runner.run(options);
    print_result(name, result, options.verbose);
  };

  using HeadDim = _32;

  run(Shape< _64, _128, HeadDim>{}, "tma ws cooperative 64x128x32");
  run(Shape<_128, _128, HeadDim>{}, "tma ws cooperative 128x128x32");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void run_bwd_64(Options const & options) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    BwdRunner<decltype(shape), decltype(kernel_options)...> runner;
    auto result = runner.run(options);
    print_result(name, result, options.verbose);
  };

  using HeadDim = _64;

  run(Shape< _64, _128, HeadDim>{}, "tma ws cooperative 64x128x64");
  run(Shape<_128, _128, HeadDim>{}, "tma ws cooperative 128x128x64");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void run_bwd_128(Options const & options) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    BwdRunner<decltype(shape), decltype(kernel_options)...> runner;
    auto result = runner.run(options);
    print_result(name, result, options.verbose);
  };

  using HeadDim = _128;

  run(Shape<_64, _128, HeadDim>{}, "tma ws cooperative 64x128x128");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// #endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main_single(int argc, char const **args) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

// #if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  //
  // Run examples
  //

  std::cout << "###### B " << options.b << " H " << options.h << " Q " << options.q << " K " << options.k << " D " << options.d << " ";
  std::cout << (options.bwd ? "Backward" : "Forward") << " " << (options.causal ? "Causal" : "Full") << " ";

  // with_fusion([&](auto fusion) {
  if (options.bwd) {
#ifndef FP8
    if (options.d <= 32) {
      run_bwd_32(options);
    } else if (options.d <= 64) {
      run_bwd_64(options);
    } else if (options.d <= 128) {
      run_bwd_128(options);
    } else
#endif
   {
#ifdef FP8
      std::cout << "Backward is not implemented for FP8." << std::endl;
#else
      std::cout << "No backward kernel instantiated for d=" << options.d << std::endl;
#endif
    }
  } else {
#ifndef FP8
    if (options.d <= 32) {
      run_fwd_32(options);
    } else
    if (options.d <= 64) {
      run_fwd_64(options);
    } else
#endif
    if (options.d <= 128) {
      run_fwd_128(options);
    } else
    if (options.d <= 256) {
      run_fwd_256(options);
    }
    else {
      std::cout << "No forward kernel instantiated for d=" << options.d << std::endl;
    }
  }
// };
  //);
// #endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  std::vector<std::string> full_arguments(args, args + argc);

  int result = 0;

  bool recursed = false;
  for (size_t i = 1; i < full_arguments.size(); i++) {
    if (full_arguments[i].find(',') != std::string::npos) {
      auto arg = full_arguments[i];
      size_t eq_pos = arg.find('=');
      std::string prefix = eq_pos == std::string::npos ? "" : arg.substr(0, eq_pos+1);
      std::string rest = eq_pos == std::string::npos ? arg : arg.substr(eq_pos+1);
      for (;;) {
        size_t comma_pos = rest.find(',');
        std::string current = rest.substr(0, comma_pos);
        full_arguments[i] = prefix + current;
        std::vector<const char*> next_args;
        for (auto& elem : full_arguments) { next_args.push_back(elem.data()); }
        main(argc, next_args.data());
        if (comma_pos == std::string::npos) break;
        rest = rest.substr(comma_pos+1);
      }
      recursed = true;
      break;
    }
  }

  if (! recursed) {
    main_single(argc, args);
  }

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
