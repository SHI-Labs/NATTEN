#################################################################################################
# Copyright (c) 2022-2025 Ali Hassani.
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

import math
import unittest

import torch
from natten.token_permute import (
    generate_tokperm_varlen_metadata,
    token_permute_operation,
    token_permute_varlen_operation,
    token_unpermute_operation,
    token_unpermute_varlen_operation,
)
from natten.utils import log
from natten.utils.testing import skip_if_cuda_is_not_supported, supports_float16


logger = log.get_logger(__name__)


class TokenPermuteTest(unittest.TestCase):
    def _test_permute_unpermute_torch(
        self, B, H, S, D, tile_shape, dilation, flip_dims, eps, dtype, device="cuda"
    ):
        with torch.no_grad():
            tensor = torch.randn((B, *S, H, D), device=device, dtype=dtype)

            logger.debug(
                f"Testing token permute: {tensor.shape=}, {tile_shape=}, {dilation=}, {flip_dims=}, {dtype=}."
            )

            tensor_copy = tensor.clone()
            tensor_permuted, qkv_shape, _ = token_permute_operation(
                tensor_copy,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_dims,
                use_torch=True,
            )
            tensor_out = token_unpermute_operation(
                tensor_permuted,
                token_layout_shape=qkv_shape,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_dims,
                use_torch=True,
            )

            torch.testing.assert_close(tensor, tensor_out, atol=eps, rtol=0)

    def test_permute_torch_cpu(self):
        problem_sizes = [
            # 1D
            (1, 1, (80,), 32, (8,), None),
            (1, 1, (16,), 32, (2,), None),
            (1, 1, (16,), 32, (8,), None),
            (1, 1, (8,), 32, (8,), None),
            (1, 1, (80,), 128, (20,), None),
            (1, 1, (80,), 32, (8,), (7,)),
            (1, 1, (16,), 32, (2,), (3,)),
            (1, 1, (16,), 32, (8,), (3,)),
            (1, 1, (8,), 32, (8,), (2,)),
            (1, 1, (8,), 32, (8,), (4,)),
            (1, 1, (80,), 128, (40,), (13,)),
            # 2D
            (1, 1, (48, 80), 32, (8, 8), None),
            (1, 1, (16, 16), 32, (8, 2), None),
            (1, 1, (16, 16), 32, (8, 2), None),
            (1, 1, (44, 8), 32, (22, 8), None),
            (1, 1, (48, 80), 128, (4, 16), None),
            (1, 1, (48, 80), 32, (8, 8), (2, 8)),
            (1, 1, (16, 16), 32, (8, 2), (2, 4)),
            (1, 1, (16, 16), 32, (8, 2), (2, 1)),
            (1, 1, (44, 8), 32, (22, 8), (4, 8)),
            # 3D
            (1, 1, (30, 48, 80), 32, (4, 8, 8), None),
            (1, 1, (16, 16, 16), 32, (4, 8, 2), None),
            (1, 1, (32, 16, 16), 32, (4, 8, 2), None),
            (1, 1, (32, 44, 8), 32, (16, 22, 8), None),
            (1, 1, (30, 48, 80), 128, (2, 4, 16), None),
            (1, 1, (30, 48, 80), 32, (4, 8, 8), (2, 5, 4)),
            (1, 1, (16, 16, 16), 32, (4, 8, 2), (1, 9, 3)),
        ]
        for B, H, S, D, tile_shape, dilation in problem_sizes:
            for flip_dims in [True, False]:
                self._test_permute_unpermute_torch(
                    B=B,
                    H=H,
                    S=S,
                    D=D,
                    tile_shape=tile_shape,
                    dilation=dilation,
                    flip_dims=flip_dims,
                    eps=1e-3,
                    dtype=torch.float32,
                    device="cpu",
                )

    @skip_if_cuda_is_not_supported()
    def test_permute_torch_cuda(self):
        torch.set_default_device("cuda")
        problem_sizes = [
            # 1D
            (1, 1, (128,), 1, (2,), None),
            (1, 1, (80,), 32, (8,), None),
            (1, 1, (16,), 32, (2,), None),
            (1, 1, (16,), 32, (8,), None),
            (1, 1, (8,), 32, (8,), None),
            (1, 1, (80,), 128, (20,), None),
            (1, 1, (80,), 128, (40,), None),
            (1, 1, (80,), 32, (8,), (7,)),
            (1, 1, (16,), 32, (2,), (3,)),
            (1, 1, (16,), 32, (8,), (3,)),
            (1, 1, (8,), 32, (8,), (2,)),
            (1, 1, (8,), 32, (8,), (4,)),
            (1, 1, (80,), 128, (20,), (13,)),
            (1, 1, (80,), 128, (40,), (13,)),
            # 2D
            (1, 1, (48, 80), 32, (8, 8), None),
            (1, 1, (16, 16), 32, (8, 2), None),
            (1, 1, (16, 16), 32, (8, 2), None),
            (1, 1, (44, 8), 32, (22, 8), None),
            (1, 1, (48, 80), 128, (4, 16), None),
            (1, 1, (48, 80), 128, (4, 16), None),
            (1, 1, (48, 80), 32, (8, 8), (2, 8)),
            (1, 1, (16, 16), 32, (8, 2), (2, 4)),
            (1, 1, (16, 16), 32, (8, 2), (2, 1)),
            (1, 1, (44, 8), 32, (22, 8), (4, 8)),
            (1, 1, (48, 80), 128, (4, 16), (11, 7)),
            (1, 1, (48, 80), 128, (4, 16), (5, 11)),
            # 3D
            (1, 1, (30, 48, 80), 32, (4, 8, 8), None),
            (1, 1, (16, 16, 16), 32, (4, 8, 2), None),
            (1, 1, (32, 16, 16), 32, (4, 8, 2), None),
            (1, 1, (32, 44, 8), 32, (16, 22, 8), None),
            (1, 1, (30, 48, 80), 128, (2, 4, 16), None),
            (1, 1, (30, 48, 80), 32, (4, 8, 8), (2, 5, 4)),
            (1, 1, (16, 16, 16), 32, (4, 8, 2), (1, 9, 3)),
            (1, 1, (32, 16, 16), 32, (4, 8, 2), (7, 7, 7)),
            (1, 1, (32, 44, 8), 32, (16, 22, 8), (13, 2, 6)),
            (1, 1, (30, 48, 80), 128, (2, 4, 16), (7, 3, 1)),
            (1, 1, (57, 32, 32), 128, (8, 4, 8), (2, 1, 1)),
            (1, 1, (57, 32, 32), 128, (4, 8, 8), (2, 1, 1)),
            (1, 1, (57, 32, 32), 128, (2, 8, 16), (2, 1, 1)),
            (1, 1, (57, 32, 32), 128, (4, 4, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (8, 4, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 8, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (2, 8, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 4, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (8, 4, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 8, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (2, 8, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 4, 16), (2, 1, 1)),
        ]
        for B, H, S, D, tile_shape, dilation in problem_sizes:
            for flip_dims in [True, False]:
                self._test_permute_unpermute_torch(
                    B=B,
                    H=H,
                    S=S,
                    D=D,
                    tile_shape=tile_shape,
                    dilation=dilation,
                    flip_dims=flip_dims,
                    eps=1e-3,
                    dtype=(
                        torch.float16
                        if supports_float16(torch.get_default_device())
                        else torch.float32
                    ),
                    device="cuda",
                )

    @skip_if_cuda_is_not_supported()
    def test_permute_kernel_cuda(self):
        torch.set_default_device("cuda")
        problem_sizes = [
            # 1D
            (1, 1, (7,), 1, (3,), None),
            (1, 1, (7,), 4, (3,), None),
            (1, 1, (19,), 4, (3,), None),
            (1, 1, (19,), 4, (3,), (7,)),
            (1, 1, (128,), 4, (2,), None),
            (1, 1, (80,), 32, (8,), None),
            (1, 1, (16,), 32, (2,), None),
            (1, 1, (16,), 32, (8,), None),
            (1, 1, (8,), 32, (8,), None),
            (1, 1, (80,), 128, (20,), None),
            (1, 1, (80,), 128, (40,), None),
            (1, 1, (80,), 32, (8,), (7,)),
            (1, 1, (16,), 32, (2,), (3,)),
            (1, 1, (16,), 32, (8,), (3,)),
            (1, 1, (8,), 32, (8,), (2,)),
            (1, 1, (8,), 32, (8,), (4,)),
            (1, 1, (80,), 128, (20,), (13,)),
            (1, 1, (80,), 128, (40,), (13,)),
            # 2D
            (1, 1, (48, 80), 32, (8, 8), None),
            (1, 1, (16, 16), 32, (8, 2), None),
            (1, 1, (16, 16), 32, (8, 2), None),
            (1, 1, (44, 8), 32, (22, 8), None),
            (1, 1, (48, 80), 128, (4, 16), None),
            (1, 1, (48, 80), 128, (4, 16), None),
            (1, 1, (48, 80), 32, (8, 8), (2, 8)),
            (1, 1, (16, 16), 32, (8, 2), (2, 4)),
            (1, 1, (16, 16), 32, (8, 2), (2, 1)),
            (1, 1, (44, 8), 32, (22, 8), (4, 8)),
            (1, 1, (48, 80), 128, (4, 16), (11, 7)),
            (1, 1, (48, 80), 128, (4, 16), (5, 11)),
            # 3D
            (1, 1, (30, 48, 80), 32, (4, 8, 8), None),
            (1, 1, (16, 16, 16), 32, (4, 8, 2), None),
            (1, 1, (32, 16, 16), 32, (4, 8, 2), None),
            (1, 1, (32, 44, 8), 32, (16, 22, 8), None),
            (1, 1, (30, 48, 80), 128, (2, 4, 16), None),
            (1, 1, (30, 48, 80), 32, (4, 8, 8), (2, 5, 4)),
            (1, 1, (16, 16, 16), 32, (4, 8, 2), (1, 9, 3)),
            (1, 1, (32, 16, 16), 32, (4, 8, 2), (7, 7, 7)),
            (1, 1, (32, 44, 8), 32, (16, 22, 8), (13, 2, 6)),
            (1, 1, (30, 48, 80), 128, (2, 4, 16), (7, 3, 1)),
            (1, 1, (57, 32, 32), 128, (8, 4, 8), (2, 1, 1)),
            (1, 1, (57, 32, 32), 128, (4, 8, 8), (2, 1, 1)),
            (1, 1, (57, 32, 32), 128, (2, 8, 16), (2, 1, 1)),
            (1, 1, (57, 32, 32), 128, (4, 4, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (8, 4, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 8, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (2, 8, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 4, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (8, 4, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 8, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (2, 8, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 4, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 1, (4, 4, 16), (2, 1, 1)),
        ]
        for B, H, S, D, tile_shape, dilation in problem_sizes:
            for flip_dims in [True, False]:
                for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    self._test_permute_cuda_kernel(
                        B=B,
                        H=H,
                        S=S,
                        D=D,
                        tile_shape=tile_shape,
                        dilation=dilation,
                        flip_dims=flip_dims,
                        eps=1e-4,
                        dtype=dtype,
                        device="cuda",
                    )

    def _test_permute_cuda_kernel(
        self, B, H, S, D, tile_shape, dilation, flip_dims, eps, dtype, device="cuda"
    ):
        with torch.no_grad():
            tensor = torch.randn((B, *S, H, D), device=device, dtype=dtype)

            logger.debug(
                f"Testing token permute kernel against torch reference: {tensor.shape=}, {tile_shape=}, {dilation=}, {flip_dims=}, {dtype=}."
            )

            tensor_copy = tensor.clone()
            tensor_permuted_ref, qkv_shape_ref, _ = token_permute_operation(
                tensor_copy,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_dims,
                use_torch=True,
            )
            tensor_out_ref = token_unpermute_operation(
                tensor_permuted_ref,
                token_layout_shape=qkv_shape_ref,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_dims,
                use_torch=True,
            )

            tensor_copy_2 = tensor.clone()
            tensor_permuted, qkv_shape, _ = token_permute_operation(
                tensor_copy_2,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_dims,
                use_torch=False,
            )
            tensor_out = token_unpermute_operation(
                tensor_permuted,
                token_layout_shape=qkv_shape,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_dims,
                use_torch=False,
            )

            torch.testing.assert_close(
                tensor_permuted, tensor_permuted_ref, atol=eps, rtol=0
            )
            torch.testing.assert_close(tensor_out, tensor_out_ref, atol=eps, rtol=0)
            torch.testing.assert_close(tensor_out, tensor, atol=eps, rtol=0)


class TokenPermuteVarlenTest(unittest.TestCase):
    @skip_if_cuda_is_not_supported()
    def test_permute_varlen_kernel_cuda(self):
        torch.set_default_device("cuda")
        problem_sizes = [
            ([(47, 2), (56, 2)], 3, 32, (8, 16), (2, 1)),
            ([(47, 2), (56, 2)], 3, 32, (8, 32), (2, 1)),
            (
                [
                    (37, 21),
                    (35, 75),
                    (72, 37),
                    (78, 58),
                    (78, 55),
                    (50, 32),
                    (21, 69),
                    (67, 15),
                ],
                3,
                32,
                (8, 32),
                (6, 1),
            ),
            (
                [
                    (37, 21),
                    (35, 75),
                    (72, 37),
                    (78, 58),
                    (78, 55),
                    (50, 32),
                    (21, 69),
                    (67, 15),
                ],
                3,
                32,
                (8, 16),
                (6, 1),
            ),
            ([(3, 5), (4, 4)], 1, 1, (2, 2), (1, 2)),
            ([(5, 10), (8, 8)], 1, 1, (4, 4), (1, 2)),
            ([(48, 80), (8, 8)], 1, 1, (4, 4), (1, 2)),
            ([(48, 80), (16, 16)], 1, 1, (4, 4), (1, 2)),
            ([(48, 80), (16, 16), (7, 4)], 1, 1, (4, 4), (1, 2)),
            ([(48, 80), (16, 16), (7, 4), (44, 8), (48, 23)], 1, 1, (4, 4), (1, 2)),
            (
                [(48, 80), (16, 16), (7, 4), (44, 8), (48, 23), (48, 80)],
                4,
                8,
                (5, 7),
                (1, 2),
            ),
            (
                [(48, 80), (16, 16), (7, 4), (44, 8), (48, 23), (48, 80)],
                4,
                8,
                (5, 7),
                (2, 2),
            ),
            (
                [(48, 80), (16, 16), (7, 4), (44, 8), (48, 23), (48, 80)],
                4,
                8,
                (5, 7),
                (2, 3),
            ),
            # Trivial single-batch problems
            ([(29,)], 2, 4, (8,), None),
            ([(16, 7)], 2, 4, (2, 4), None),
            ([(16, 20, 19)], 2, 4, (4, 2, 4), None),
            # 1D
            ([(3,), (6,)], 1, 1, (4,), (2,)),
            ([(3,), (6,)], 1, 1, (4,), None),
            ([(3,), (6,), (2,)], 1, 1, (4,), None),
            ([(29,), (37,)], 1, 4, (8,), None),
            ([(29,), (37,)], 2, 4, (8,), None),
            ([(29,), (37,), (4,)], 2, 4, (8,), None),
            ([(29,), (37,), (127,)], 2, 4, (8,), None),
            ([(29,), (37,), (127,), (33,)], 2, 4, (8,), None),
            # 2D
            ([(18, 9), (37, 15), (16, 16), (17, 4)], 2, 4, (4, 3), None),
            #
            (
                [(48, 80), (16, 16), (7, 4), (44, 8), (48, 23), (48, 80)],
                4,
                8,
                (5, 7),
                (2, 3),
            ),
            # 3D
            ([(18, 9, 7), (37, 2, 15), (16, 1, 16), (17, 4, 8)], 2, 4, (4, 3, 5), None),
            (
                [(18, 9, 7), (37, 2, 15), (16, 1, 16), (17, 4, 8)],
                2,
                4,
                (4, 3, 5),
                (5, 1, 3),
            ),
        ]
        for token_layout_list, heads, head_dim, tile_shape, dilation in problem_sizes:
            for flip_dims in [False, True]:
                for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    self._test_permute_varlen_cuda_kernel(
                        heads=heads,
                        head_dim=head_dim,
                        token_layout_list=token_layout_list,
                        tile_shape=tile_shape,
                        dilation=dilation,
                        flip_dims=flip_dims,
                        eps=1e-6,
                        dtype=dtype,
                        device="cuda",
                    )

    def _test_permute_varlen_cuda_kernel(
        self,
        token_layout_list,
        heads,
        head_dim,
        tile_shape,
        dilation,
        flip_dims,
        eps,
        dtype,
        device="cuda",
    ):
        na_dim = len(tile_shape)
        batch = len(token_layout_list)

        metadata = generate_tokperm_varlen_metadata(
            token_layout_list=token_layout_list,
            tile_shape=tile_shape,
            device=device,
            dilation=dilation,
            flip_tiled_dims=flip_dims,
        )

        total_seqlen_pre_permute = metadata["total_seqlen_pre_permute"]
        total_seqlen_post_permute = metadata["total_seqlen_post_permute"]

        with torch.no_grad():
            tensor = torch.randn(
                (1, total_seqlen_pre_permute, heads, head_dim),
                device=device,
                dtype=dtype,
            )

            tensor_ref = tensor.clone()
            tensor_kernel = tensor.clone()

            logger.debug(
                f"Testing token permute varlen kernel against torch reference: \n{token_layout_list=}\n{tensor.shape=}, {tile_shape=}, {dilation=}, {flip_dims=}, {dtype=}."
            )

            # reference impl
            permuted_ref = []
            unpermuted_ref = []
            for i in range(batch):
                seq_start = sum([math.prod(x) for x in token_layout_list[:i]])
                seq_end = seq_start + math.prod(token_layout_list[i])
                tensor_ref_batch = tensor_ref[:, seq_start:seq_end, :, :]
                tensor_ref_batch = tensor_ref_batch.reshape(
                    1, *token_layout_list[i], heads, head_dim
                )

                tensor_permuted_ref, qkv_shape_ref, _ = token_permute_operation(
                    tensor_ref_batch,
                    tile_shape=tile_shape,
                    dilation=dilation,
                    flip_tiled_dims=flip_dims,
                    use_torch=False,
                )
                tensor_out_ref = token_unpermute_operation(
                    tensor_permuted_ref,
                    token_layout_shape=qkv_shape_ref,
                    tile_shape=tile_shape,
                    dilation=dilation,
                    flip_tiled_dims=flip_dims,
                    use_torch=False,
                )
                tensor_permuted_ref = tensor_permuted_ref.reshape(
                    1, -1, heads, head_dim
                )
                permuted_ref.append(tensor_permuted_ref)
                unpermuted_ref.append(tensor_out_ref.flatten(1, na_dim))

            tensor_permuted_ref = torch.cat(permuted_ref, dim=1)
            tensor_unpermuted_ref = torch.cat(unpermuted_ref, dim=1)

            assert tensor_permuted_ref.shape[1] == total_seqlen_post_permute
            assert tensor_unpermuted_ref.shape[1] == total_seqlen_pre_permute

            tensor_permuted = token_permute_varlen_operation(
                tensor_kernel,
                metadata=metadata,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_dims,
                use_torch=False,
            )
            tensor_unpermuted = token_unpermute_varlen_operation(
                tensor_permuted,
                metadata=metadata,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_dims,
                use_torch=False,
            )

            torch.testing.assert_close(
                tensor_permuted, tensor_permuted_ref, atol=eps, rtol=0
            )
            torch.testing.assert_close(
                tensor_unpermuted, tensor_unpermuted_ref, atol=eps, rtol=0
            )


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()
