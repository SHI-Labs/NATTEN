#include <cuda_runtime.h>
#include <natten/dtypes.cuh>
#include <natten/gemm_argpack.cuh>
#include <natten/config.h>
#include <natten/cuda/gemm/na2d.cuh>
namespace natten { 
namespace cuda { 
namespace gemm { 

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks21_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 12, 1, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks23_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks23_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks23_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks25_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 13, 1, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks25_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 13, 1, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks25_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 13, 1, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks27_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks27_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks27_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks29_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks29_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks29_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks31_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks31_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

} 
} 
} 

