#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################


from typing import Any, Callable, Dict, Optional, Tuple

# from natten.libnatten import (  # type: ignore
#    na1d_av_backward as na1d_av_backward,
#    na1d_av_forward as na1d_av,
#    na1d_backward as na1d_fused_backward,
#    na1d_forward as na1d_fused,
#    na1d_qk_backward as na1d_qk_backward,
#    na1d_qk_forward as na1d_qk,
#    na2d_av_backward as na2d_av_backward,
#    na2d_av_forward as na2d_av,
#    na2d_backward as na2d_fused_backward,
#    na2d_forward as na2d_fused,
#    na2d_qk_backward as na2d_qk_backward,
#    na2d_qk_forward as na2d_qk,
#    na3d_av_backward as na3d_av_backward,
#    na3d_av_forward as na3d_av,
#    na3d_backward as na3d_fused_backward,
#    na3d_forward as na3d_fused,
#    na3d_qk_backward as na3d_qk_backward,
#    na3d_qk_forward as na3d_qk,
# )
from natten.functional import (
    na1d,
    na1d_av,
    na1d_qk,
    na2d,
    na2d_av,
    na2d_qk,
    na3d,
    na3d_av,
    na3d_qk,
)

from torch import Tensor

from .ops import NAOp

try:
    from xformers.ops.fmha import (  # type: ignore
        attn_bias,
        memory_efficient_attention,
        MemoryEfficientAttentionCutlassOp,
        MemoryEfficientAttentionFlashAttentionOp,
    )

    def fmha(
        q: Tensor, k: Tensor, v: Tensor, window_size: Optional[int] = None
    ) -> Tensor:
        bias = (
            None
            if window_size is None or window_size < 1
            else attn_bias.LowerTriangularFromBottomRightLocalAttentionMask(window_size)
        )
        return memory_efficient_attention(
            q, k, v, op=MemoryEfficientAttentionCutlassOp, attn_bias=bias
        )

    def fav2(
        q: Tensor, k: Tensor, v: Tensor, window_size: Optional[int] = None
    ) -> Tensor:
        bias = (
            None
            if window_size is None or window_size < 1
            else attn_bias.LowerTriangularFromBottomRightLocalAttentionMask(window_size)
        )
        return memory_efficient_attention(
            q, k, v, op=MemoryEfficientAttentionFlashAttentionOp, attn_bias=bias
        )

except:

    def fmha(
        q: Tensor, k: Tensor, v: Tensor, window_size: Optional[int] = None
    ) -> Tensor:
        raise RuntimeError("Benchmarking FMHA and FAv2 requires xFormers.")

    def fav2(
        q: Tensor, k: Tensor, v: Tensor, window_size: Optional[int] = None
    ) -> Tensor:
        raise RuntimeError("Benchmarking FMHA and FAv2 requires xFormers.")


_NA_1D_C_KEYWORDS = {
    NAOp.FusedForward: ["natten::cuda::fna::FusedNeighborhoodAttentionKernel<1,"],
    NAOp.FusedBackward: [
        "natten::cuda::fna::FusedNeighborhoodAttentionBackwardKernel<1,"
    ],
    NAOp.PN: ["NA1dPN", "OperatorE0", "gemm::PointwiseNeighborhood1D"],
    NAOp.NN: ["NA1dNN", "OperatorE1", "gemm::NeighborhoodNeighborhood1D"],
    NAOp.IN: ["NA1dIN", "OperatorE2", "gemm::InverseNeighborhood1D"],
    NAOp.RPB: ["rpb_forward"],
    NAOp.RPBGRAD: [
        "rpb_backward",
        "rpb_grad",
        "natten1drpb_cuda_backward",
        "na1d_rpb_cuda_backward",
        "rel_pos_bias_gradient_1d",
        "naive::RelPosBiasGradient1D",
    ],
    NAOp.QKRPB: ["na1d_qkrpb_cuda_forward", "natten1dqkrpb_cuda_forward"],
    NAOp.AV: ["na1d_av_cuda_forward", "natten1dav_cuda_forward"],
    NAOp.QGRAD: ["na1d_q_cuda_backward", "natten1dq_cuda_backward"],
    NAOp.KGRAD: ["na1d_k_cuda_backward", "natten1dk_cuda_backward"],
    NAOp.VGRAD: ["na1d_v_cuda_backward", "natten1dv_cuda_backward"],
    NAOp.AGRAD: ["na1d_a_cuda_backward", "natten1da_cuda_backward"],
    NAOp.LegacyPN: [
        "pointwise_neighborhood_1d",
        "naive::PointwiseNeighborhood1D",
    ],
    NAOp.LegacyNN: [
        "neighborhood_neighborhood_1d",
        "naive::NeighborhoodNeighborhood1D",
    ],
    NAOp.LegacyIN: ["inverse_neighborhood_1d", "naive::InverseNeighborhood1D"],
}

_NA_2D_C_KEYWORDS = {
    NAOp.FusedForward: ["natten::cuda::fna::FusedNeighborhoodAttentionKernel<2,"],
    NAOp.FusedBackward: [
        "natten::cuda::fna::FusedNeighborhoodAttentionBackwardKernel<2,"
    ],
    NAOp.PN: ["NA2dPN", "OperatorE0", "gemm::PointwiseNeighborhood2D"],
    NAOp.NN: ["NA2dNN", "OperatorE1", "gemm::NeighborhoodNeighborhood2D"],
    NAOp.IN: ["NA2dIN", "OperatorE2", "gemm::InverseNeighborhood2D"],
    NAOp.RPB: ["rpb_forward"],
    NAOp.RPBGRAD: [
        "rpb_backward",
        "rpb_grad",
        "natten2drpb_cuda_backward",
        "na2d_rpb_cuda_backward",
        "rel_pos_bias_gradient_2d",
        "naive::RelPosBiasGradient2D",
    ],
    NAOp.QKRPB: ["na2d_qkrpb_cuda_forward", "natten2dqkrpb_cuda_forward"],
    NAOp.AV: ["na2d_av_cuda_forward", "natten2dav_cuda_forward"],
    NAOp.QGRAD: ["na2d_q_cuda_backward", "natten2dq_cuda_backward"],
    NAOp.KGRAD: ["na2d_k_cuda_backward", "natten2dk_cuda_backward"],
    NAOp.VGRAD: ["na2d_v_cuda_backward", "natten2dv_cuda_backward"],
    NAOp.AGRAD: ["na2d_a_cuda_backward", "natten2da_cuda_backward"],
    NAOp.LegacyPN: [
        "pointwise_neighborhood_2d",
        "naive::PointwiseNeighborhood2D",
    ],
    NAOp.LegacyNN: [
        "neighborhood_neighborhood_2d",
        "naive::NeighborhoodNeighborhood2D",
    ],
    NAOp.LegacyIN: ["inverse_neighborhood_2d", "naive::InverseNeighborhood2D"],
}


_NA_3D_C_KEYWORDS = {
    NAOp.FusedForward: ["natten::cuda::fna::FusedNeighborhoodAttentionKernel<3,"],
    NAOp.FusedBackward: [
        "natten::cuda::fna::FusedNeighborhoodAttentionBackwardKernel<3,"
    ],
    NAOp.RPB: ["rpb_forward"],
    NAOp.RPBGRAD: [
        "rpb_backward",
        "rpb_grad",
        "natten3drpb_cuda_backward",
        "na3d_rpb_cuda_backward",
        "rel_pos_bias_gradient_3d",
        "naive::RelPosBiasGradient3D",
    ],
    NAOp.QKRPB: ["na3d_qkrpb_cuda_forward", "natten3dqkrpb_cuda_forward"],
    NAOp.AV: ["na3d_av_cuda_forward", "natten3dav_cuda_forward"],
    NAOp.QGRAD: ["na3d_q_cuda_backward", "natten3dq_cuda_backward"],
    NAOp.KGRAD: ["na3d_k_cuda_backward", "natten3dk_cuda_backward"],
    NAOp.VGRAD: ["na3d_v_cuda_backward", "natten3dv_cuda_backward"],
    NAOp.AGRAD: ["na3d_a_cuda_backward", "natten3da_cuda_backward"],
    NAOp.LegacyPN: [
        "pointwise_neighborhood_3d",
        "naive::PointwiseNeighborhood3D",
    ],
    NAOp.LegacyNN: [
        "neighborhood_neighborhood_3d",
        "naive::NeighborhoodNeighborhood3D",
    ],
    NAOp.LegacyIN: ["inverse_neighborhood_3d", "naive::InverseNeighborhood3D"],
}


def get_kernel_map(na_dim: int) -> Dict[NAOp, Any]:
    if na_dim == 1:
        return _NA_1D_C_KEYWORDS
    if na_dim == 2:
        return _NA_2D_C_KEYWORDS
    if na_dim == 3:
        return _NA_3D_C_KEYWORDS
    raise ValueError(f"Expected NA1d, 2d, or 3d, got {na_dim=}.")


# TODO add fused backward kernel
def get_ops(
    na_dim: int,
) -> Tuple[Callable, Callable, Callable]:
    # ) -> Tuple[Callable, Callable, Callable, Callable, Callable, Optional[Callable]]:
    if na_dim == 1:
        return (
            na1d_qk,
            na1d_av,
            na1d,
            # na1d_qk,
            # na1d_qk_backward,
            # na1d_av,
            # na1d_av_backward,
            # na1d_fused,
            # na1d_fused_backward,
        )
    if na_dim == 2:
        return (
            na2d_qk,
            na2d_av,
            na2d,
            # na2d_qk,
            # na2d_qk_backward,
            # na2d_av,
            # na2d_av_backward,
            # na2d_fused,
            # na2d_fused_backward,
        )
    if na_dim == 3:
        return (
            na3d_qk,
            na3d_av,
            na3d,
            # na3d_qk,
            # na3d_qk_backward,
            # na3d_av,
            # na3d_av_backward,
            # na3d_fused,
            # na3d_fused_backward,
        )
    raise ValueError(f"Expected NA1d, 2d, or 3d, got {na_dim=}.")
