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

"""
Comprehensive tests for the NATTEN Metal (MPS) backend.
Covers forward correctness, backward correctness, FMHA, GQA, causal, dilation, stride, dtypes.
"""

import unittest

import torch

HAS_MPS = torch.backends.mps.is_available()

try:
    import natten
    from natten.backends.metal import HAS_METAL_NATTEN
except ImportError:
    HAS_METAL_NATTEN = False


def skip_if_no_mps(fn):
    return unittest.skipUnless(HAS_MPS, "MPS device not available")(fn)


def skip_if_no_metal(fn):
    return unittest.skipUnless(HAS_MPS and HAS_METAL_NATTEN, "Metal NATTEN not available")(fn)


def _get_win_start_1d(index, window_left, window_right, stride, length, is_causal):
    """Compute window start for 1D NA, matching mask.hpp exactly."""
    if is_causal:
        leader = min((index // stride) * stride + stride - 1, length - 1)
        return max(leader - window_left - window_right, 0)
    else:
        leader = min((index // stride) * stride + (stride // 2), length - 1)
        return max(leader - window_left, 0) + (
            (leader + window_right >= length) * (length - window_right - leader - 1)
        )


def _get_win_end_1d(index, start, window_size, length, is_causal):
    """Compute window end for 1D NA, matching mask.hpp exactly."""
    if is_causal:
        return min(index + 1, length)
    else:
        return start + window_size


def _qkv_fix_dilation(qkv_shape, dilation, dilation_group):
    """Correct effective sequence length for a dilation group."""
    padding = 1 - ((dilation_group + (dilation - (qkv_shape % dilation))) // dilation)
    return (qkv_shape // dilation) + padding


def naive_na1d_forward(query, key, value, kernel_size, is_causal=False, scale=None,
                       stride=1, dilation=1):
    """Pure PyTorch reference for NA1D forward with stride, dilation, GQA support."""
    B, S, H, D = query.shape
    DV = value.shape[-1]
    HKV = key.shape[2]
    scale = scale or D ** -0.5

    wl = kernel_size // 2
    wr = (kernel_size // 2) + ((kernel_size % 2) - 1)

    output = torch.zeros(B, S, H, DV, dtype=query.dtype, device=query.device)
    lse = torch.zeros(B, S, H, dtype=torch.float32, device=query.device)

    for b in range(B):
        for i in range(S):
            for h in range(H):
                h_kv = h // (H // HKV) if HKV != H else h

                # Dilation: decompose into dilation group coordinates
                di_group = i % dilation
                q_coord = i // dilation

                # Corrected shape for this dilation group
                eff_len = _qkv_fix_dilation(S, dilation, di_group)

                start = _get_win_start_1d(q_coord, wl, wr, stride, eff_len, is_causal)
                end = _get_win_end_1d(q_coord, start, kernel_size, eff_len, is_causal)

                q_vec = query[b, i, h, :].float()
                scores = []
                indices = []
                for j_inner in range(start, end):
                    # Map back to global index
                    j_global = j_inner * dilation + di_group
                    if j_global >= S:
                        continue
                    k_vec = key[b, j_global, h_kv, :].float()
                    s = (q_vec * k_vec).sum() * scale
                    scores.append(s)
                    indices.append(j_global)

                if len(scores) == 0:
                    continue

                scores_t = torch.stack(scores)
                max_s = scores_t.max()
                exp_s = torch.exp(scores_t - max_s)
                sum_s = exp_s.sum()

                lse[b, i, h] = torch.log(sum_s) + max_s

                weights = exp_s / sum_s
                acc = torch.zeros(DV, dtype=torch.float32, device=query.device)
                for w, j in zip(weights, indices):
                    acc += w * value[b, j, h_kv, :].float()

                output[b, i, h, :] = acc.to(query.dtype)

    return output, lse


def naive_na2d_forward(query, key, value, kernel_size, is_causal=(False, False),
                       scale=None, stride=(1, 1), dilation=(1, 1)):
    """Pure PyTorch reference for NA2D forward with stride, dilation, GQA support.
    query: [B, Hsp, Wsp, H, D], key: [B, Hsp, Wsp, HKV, D], value: [B, Hsp, Wsp, HKV, DV]
    kernel_size, stride, dilation: 2-tuples
    """
    B, Hsp, Wsp, H, D = query.shape
    DV = value.shape[-1]
    HKV = key.shape[3]
    scale = scale or D ** -0.5
    ks_h, ks_w = kernel_size
    str_h, str_w = stride
    dil_h, dil_w = dilation

    wl_h, wl_w = ks_h // 2, ks_w // 2
    wr_h = (ks_h // 2) + ((ks_h % 2) - 1)
    wr_w = (ks_w // 2) + ((ks_w % 2) - 1)

    output = torch.zeros(B, Hsp, Wsp, H, DV, dtype=query.dtype, device=query.device)

    for b in range(B):
        for ih in range(Hsp):
            for iw in range(Wsp):
                for h in range(H):
                    h_kv = h // (H // HKV) if HKV != H else h

                    # Dilation decomposition for each dimension
                    di_h = ih % dil_h
                    di_w = iw % dil_w
                    qc_h = ih // dil_h
                    qc_w = iw // dil_w

                    eff_h = _qkv_fix_dilation(Hsp, dil_h, di_h)
                    eff_w = _qkv_fix_dilation(Wsp, dil_w, di_w)

                    start_h = _get_win_start_1d(qc_h, wl_h, wr_h, str_h, eff_h, is_causal[0])
                    end_h = _get_win_end_1d(qc_h, start_h, ks_h, eff_h, is_causal[0])
                    start_w = _get_win_start_1d(qc_w, wl_w, wr_w, str_w, eff_w, is_causal[1])
                    end_w = _get_win_end_1d(qc_w, start_w, ks_w, eff_w, is_causal[1])

                    q_vec = query[b, ih, iw, h, :].float()
                    scores = []
                    kv_indices = []
                    for jh_inner in range(start_h, end_h):
                        for jw_inner in range(start_w, end_w):
                            jh = jh_inner * dil_h + di_h
                            jw = jw_inner * dil_w + di_w
                            if jh >= Hsp or jw >= Wsp:
                                continue
                            k_vec = key[b, jh, jw, h_kv, :].float()
                            s = (q_vec * k_vec).sum() * scale
                            scores.append(s)
                            kv_indices.append((jh, jw))

                    if len(scores) == 0:
                        continue

                    scores_t = torch.stack(scores)
                    max_s = scores_t.max()
                    exp_s = torch.exp(scores_t - max_s)
                    sum_s = exp_s.sum()
                    weights = exp_s / sum_s

                    acc = torch.zeros(DV, dtype=torch.float32, device=query.device)
                    for w, (jh, jw) in zip(weights, kv_indices):
                        acc += w * value[b, jh, jw, h_kv, :].float()

                    output[b, ih, iw, h, :] = acc.to(query.dtype)

    return output


def naive_fmha_forward(query, key, value, is_causal=False, scale=None):
    """Full self-attention reference."""
    return naive_na1d_forward(query, key, value, query.shape[1], is_causal=is_causal, scale=scale)


class MetalFNAForwardTest(unittest.TestCase):
    """Forward correctness tests for Metal FNA."""

    @skip_if_no_metal
    def test_na1d_basic_fp32(self):
        """NA1D forward, kernel_size=5, FP32."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=5)

        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na1d_basic_fp16(self):
        """NA1D forward, kernel_size=5, FP16."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float16)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float16)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float16)

        out = natten.na1d(q, k, v, kernel_size=5)

        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5)
        torch.testing.assert_close(out, ref_out, atol=2e-3, rtol=2e-3)

    @skip_if_no_metal
    def test_na1d_basic_bf16(self):
        """NA1D forward, kernel_size=5, BF16."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.bfloat16)

        out = natten.na1d(q, k, v, kernel_size=5)

        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5)
        torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)

    @skip_if_no_metal
    def test_na1d_even_kernel(self):
        """NA1D with even kernel_size=6."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=6)

        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=6)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na1d_causal(self):
        """NA1D causal forward."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=5, is_causal=True)

        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5, is_causal=True)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na1d_gqa(self):
        """NA1D with GQA (heads_q=4, heads_kv=2)."""
        torch.manual_seed(42)
        B, S, D = 1, 16, 32
        HQ, HKV = 4, 2
        q = torch.randn(B, S, HQ, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, HKV, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, HKV, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=5)

        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na1d_full_window(self):
        """NA1D with kernel_size=seqlen (equivalent to full attention)."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=S)

        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=S)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na1d_large_batch(self):
        """NA1D with batch > 1."""
        torch.manual_seed(42)
        B, S, H, D = 4, 32, 4, 64
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=7)

        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=7)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)


class MetalFMHAForwardTest(unittest.TestCase):
    """Forward correctness tests for Metal FMHA."""

    @skip_if_no_metal
    def test_fmha_non_causal_fp32(self):
        """FMHA non-causal FP32."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        from natten.backends.metal import metal_fmha
        out = metal_fmha(q, k, v, is_causal=False)

        ref_out, _ = naive_fmha_forward(q, k, v, is_causal=False)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_fmha_causal_fp32(self):
        """FMHA causal FP32."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        from natten.backends.metal import metal_fmha
        out = metal_fmha(q, k, v, is_causal=True)

        ref_out, _ = naive_fmha_forward(q, k, v, is_causal=True)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_fmha_non_causal_fp16(self):
        """FMHA non-causal FP16."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float16)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float16)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float16)

        from natten.backends.metal import metal_fmha
        out = metal_fmha(q, k, v, is_causal=False)

        ref_out, _ = naive_fmha_forward(q, k, v, is_causal=False)
        torch.testing.assert_close(out, ref_out, atol=2e-3, rtol=2e-3)

    @skip_if_no_metal
    def test_fmha_even_seqlen(self):
        """FMHA with even sequence length (regression test for even kernel_size)."""
        torch.manual_seed(42)
        B, S, H, D = 1, 20, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        from natten.backends.metal import metal_fmha
        out = metal_fmha(q, k, v, is_causal=False)

        ref_out, _ = naive_fmha_forward(q, k, v, is_causal=False)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_fmha_odd_seqlen(self):
        """FMHA with odd sequence length."""
        torch.manual_seed(42)
        B, S, H, D = 1, 17, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        from natten.backends.metal import metal_fmha
        out = metal_fmha(q, k, v, is_causal=False)

        ref_out, _ = naive_fmha_forward(q, k, v, is_causal=False)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_fmha_gqa(self):
        """FMHA with GQA."""
        torch.manual_seed(42)
        B, S, D = 1, 16, 32
        HQ, HKV = 4, 2
        q = torch.randn(B, S, HQ, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, HKV, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, HKV, D, device="mps", dtype=torch.float32)

        from natten.backends.metal import metal_fmha
        out = metal_fmha(q, k, v, is_causal=False)

        ref_out, _ = naive_fmha_forward(q, k, v, is_causal=False)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)


class MetalBackwardTest(unittest.TestCase):
    """Backward correctness tests for Metal backend."""

    def _check_backward_na1d(self, B, S, H, D, kernel_size, is_causal=False,
                              dtype=torch.float32, atol=1e-4, rtol=1e-4,
                              heads_kv=None):
        """Compare Metal backward against PyTorch autograd through naive reference."""
        torch.manual_seed(42)
        HKV = heads_kv or H

        # Metal path
        q_m = torch.randn(B, S, H, D, device="mps", dtype=dtype, requires_grad=True)
        k_m = torch.randn(B, S, HKV, D, device="mps", dtype=dtype, requires_grad=True)
        v_m = torch.randn(B, S, HKV, D, device="mps", dtype=dtype, requires_grad=True)

        out_m = natten.na1d(q_m, k_m, v_m, kernel_size=kernel_size, is_causal=is_causal)
        loss_m = out_m.sum()
        loss_m.backward()

        dq_m = q_m.grad.clone()
        dk_m = k_m.grad.clone()
        dv_m = v_m.grad.clone()

        # CPU reference path (using naive implementation + autograd)
        q_r = q_m.detach().cpu().float().requires_grad_(True)
        k_r = k_m.detach().cpu().float().requires_grad_(True)
        v_r = v_m.detach().cpu().float().requires_grad_(True)

        out_r, _ = naive_na1d_forward(q_r, k_r, v_r, kernel_size=kernel_size, is_causal=is_causal)
        loss_r = out_r.sum()
        loss_r.backward()

        dq_r = q_r.grad.to(device="mps", dtype=dtype)
        dk_r = k_r.grad.to(device="mps", dtype=dtype)
        dv_r = v_r.grad.to(device="mps", dtype=dtype)

        torch.testing.assert_close(dq_m, dq_r, atol=atol, rtol=rtol)
        torch.testing.assert_close(dk_m, dk_r, atol=atol, rtol=rtol)
        torch.testing.assert_close(dv_m, dv_r, atol=atol, rtol=rtol)

    @skip_if_no_metal
    def test_backward_na1d_basic(self):
        """NA1D backward, kernel_size=5, FP32."""
        self._check_backward_na1d(1, 16, 2, 32, kernel_size=5)

    @skip_if_no_metal
    def test_backward_na1d_even_kernel(self):
        """NA1D backward, kernel_size=6."""
        self._check_backward_na1d(1, 16, 2, 32, kernel_size=6)

    @skip_if_no_metal
    def test_backward_na1d_causal(self):
        """NA1D backward, causal."""
        self._check_backward_na1d(1, 16, 2, 32, kernel_size=5, is_causal=True)

    @skip_if_no_metal
    def test_backward_na1d_full_window(self):
        """NA1D backward, kernel_size=seqlen (full attention backward)."""
        self._check_backward_na1d(1, 16, 2, 32, kernel_size=16)

    @skip_if_no_metal
    def test_backward_na1d_large(self):
        """NA1D backward with larger dims."""
        self._check_backward_na1d(2, 32, 4, 64, kernel_size=7)

    @skip_if_no_metal
    def test_backward_na1d_gqa(self):
        """NA1D backward with GQA."""
        self._check_backward_na1d(1, 16, 4, 32, kernel_size=5, heads_kv=2)

    @skip_if_no_metal
    def test_backward_fmha_non_causal(self):
        """FMHA backward, non-causal."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32

        q_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        k_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        v_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)

        from natten.backends.metal import metal_fmha
        out_m = metal_fmha(q_m, k_m, v_m, is_causal=False)
        loss_m = out_m.sum()
        loss_m.backward()

        dq_m = q_m.grad.clone()
        dk_m = k_m.grad.clone()
        dv_m = v_m.grad.clone()

        q_r = q_m.detach().cpu().float().requires_grad_(True)
        k_r = k_m.detach().cpu().float().requires_grad_(True)
        v_r = v_m.detach().cpu().float().requires_grad_(True)

        out_r, _ = naive_fmha_forward(q_r, k_r, v_r, is_causal=False)
        loss_r = out_r.sum()
        loss_r.backward()

        dq_r = q_r.grad.to(device="mps")
        dk_r = k_r.grad.to(device="mps")
        dv_r = v_r.grad.to(device="mps")

        torch.testing.assert_close(dq_m, dq_r, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dk_m, dk_r, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dv_m, dv_r, atol=1e-4, rtol=1e-4)

    @skip_if_no_metal
    def test_backward_fmha_causal(self):
        """FMHA backward, causal."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32

        q_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        k_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        v_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)

        from natten.backends.metal import metal_fmha
        out_m = metal_fmha(q_m, k_m, v_m, is_causal=True)
        loss_m = out_m.sum()
        loss_m.backward()

        dq_m = q_m.grad.clone()
        dk_m = k_m.grad.clone()
        dv_m = v_m.grad.clone()

        q_r = q_m.detach().cpu().float().requires_grad_(True)
        k_r = k_m.detach().cpu().float().requires_grad_(True)
        v_r = v_m.detach().cpu().float().requires_grad_(True)

        out_r, _ = naive_fmha_forward(q_r, k_r, v_r, is_causal=True)
        loss_r = out_r.sum()
        loss_r.backward()

        dq_r = q_r.grad.to(device="mps")
        dk_r = k_r.grad.to(device="mps")
        dv_r = v_r.grad.to(device="mps")

        torch.testing.assert_close(dq_m, dq_r, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dk_m, dk_r, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dv_m, dv_r, atol=1e-4, rtol=1e-4)

    @skip_if_no_metal
    def test_backward_fmha_even_seqlen(self):
        """FMHA backward with even seqlen (regression for even kernel_size bug)."""
        torch.manual_seed(42)
        B, S, H, D = 1, 20, 2, 32

        q_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        k_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        v_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)

        from natten.backends.metal import metal_fmha
        out_m = metal_fmha(q_m, k_m, v_m, is_causal=False)
        loss_m = out_m.sum()
        loss_m.backward()

        dq_m = q_m.grad.clone()
        dk_m = k_m.grad.clone()
        dv_m = v_m.grad.clone()

        q_r = q_m.detach().cpu().float().requires_grad_(True)
        k_r = k_m.detach().cpu().float().requires_grad_(True)
        v_r = v_m.detach().cpu().float().requires_grad_(True)

        out_r, _ = naive_fmha_forward(q_r, k_r, v_r, is_causal=False)
        loss_r = out_r.sum()
        loss_r.backward()

        dq_r = q_r.grad.to(device="mps")
        dk_r = k_r.grad.to(device="mps")
        dv_r = v_r.grad.to(device="mps")

        torch.testing.assert_close(dq_m, dq_r, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dk_m, dk_r, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dv_m, dv_r, atol=1e-4, rtol=1e-4)


class MetalNA2DTest(unittest.TestCase):
    """Forward and backward tests for NA2D."""

    @skip_if_no_metal
    def test_na2d_forward_fp32(self):
        """NA2D forward with full window (should match full attention)."""
        torch.manual_seed(42)
        B, Hsp, Wsp, H, D = 1, 4, 4, 2, 32
        q = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)

        # Full window = full attention
        out_full = natten.na2d(q, k, v, kernel_size=(Hsp, Wsp))

        # Compare with flattened FMHA
        from natten.backends.metal import metal_fmha
        S = Hsp * Wsp
        q_flat = q.reshape(B, S, H, D)
        k_flat = k.reshape(B, S, H, D)
        v_flat = v.reshape(B, S, H, D)
        out_fmha = metal_fmha(q_flat, k_flat, v_flat, is_causal=False)
        out_fmha = out_fmha.reshape(B, Hsp, Wsp, H, D)

        torch.testing.assert_close(out_full, out_fmha, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na2d_forward_small_window(self):
        """NA2D forward with kernel_size=(3,3) — compare against naive 2D reference."""
        torch.manual_seed(42)
        B, Hsp, Wsp, H, D = 1, 8, 8, 2, 32
        q = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)

        out = natten.na2d(q, k, v, kernel_size=(3, 3))
        ref_out = naive_na2d_forward(q, k, v, kernel_size=(3, 3))
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na2d_backward(self):
        """NA2D backward with full window — compare against FMHA backward."""
        torch.manual_seed(42)
        B, Hsp, Wsp, H, D = 1, 4, 4, 2, 32
        S = Hsp * Wsp

        # Metal NA2D path (full window = full attention)
        q_2d = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        k_2d = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        v_2d = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32, requires_grad=True)

        out_2d = natten.na2d(q_2d, k_2d, v_2d, kernel_size=(Hsp, Wsp))
        out_2d.sum().backward()

        # Reference: CPU naive FMHA on flattened tensors
        q_r = q_2d.detach().reshape(B, S, H, D).cpu().float().requires_grad_(True)
        k_r = k_2d.detach().reshape(B, S, H, D).cpu().float().requires_grad_(True)
        v_r = v_2d.detach().reshape(B, S, H, D).cpu().float().requires_grad_(True)

        out_r, _ = naive_fmha_forward(q_r, k_r, v_r, is_causal=False)
        out_r.sum().backward()

        torch.testing.assert_close(q_2d.grad, q_r.grad.reshape(B, Hsp, Wsp, H, D).to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_2d.grad, k_r.grad.reshape(B, Hsp, Wsp, H, D).to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v_2d.grad, v_r.grad.reshape(B, Hsp, Wsp, H, D).to("mps"), atol=1e-4, rtol=1e-4)

    @skip_if_no_metal
    def test_na2d_backward_small_window(self):
        """NA2D backward with small kernel_size=(3,3) — compare against naive 2D reference autograd."""
        torch.manual_seed(42)
        B, Hsp, Wsp, H, D = 1, 6, 6, 2, 32
        ks = (3, 3)

        # Metal NA2D path
        q_m = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        k_m = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        v_m = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32, requires_grad=True)

        out_m = natten.na2d(q_m, k_m, v_m, kernel_size=ks)
        out_m.sum().backward()

        # CPU naive 2D reference
        q_r = q_m.detach().cpu().float().requires_grad_(True)
        k_r = k_m.detach().cpu().float().requires_grad_(True)
        v_r = v_m.detach().cpu().float().requires_grad_(True)

        ref_out = naive_na2d_forward(q_r, k_r, v_r, kernel_size=ks)
        ref_out.sum().backward()

        torch.testing.assert_close(q_m.grad, q_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_m.grad, k_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v_m.grad, v_r.grad.to("mps"), atol=1e-4, rtol=1e-4)

    @skip_if_no_metal
    def test_na2d_forward_causal(self):
        """NA2D forward with causal=(True, True)."""
        torch.manual_seed(42)
        B, Hsp, Wsp, H, D = 1, 8, 8, 2, 32
        q = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)

        out = natten.na2d(q, k, v, kernel_size=(5, 5), is_causal=(True, True))
        ref_out = naive_na2d_forward(q, k, v, kernel_size=(5, 5), is_causal=(True, True))
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na2d_backward_causal(self):
        """NA2D backward with causal=(True, True)."""
        torch.manual_seed(42)
        B, Hsp, Wsp, H, D = 1, 6, 6, 2, 32
        ks = (5, 5)

        q_m = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        k_m = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        v_m = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32, requires_grad=True)

        out_m = natten.na2d(q_m, k_m, v_m, kernel_size=ks, is_causal=(True, True))
        out_m.sum().backward()

        q_r = q_m.detach().cpu().float().requires_grad_(True)
        k_r = k_m.detach().cpu().float().requires_grad_(True)
        v_r = v_m.detach().cpu().float().requires_grad_(True)

        ref_out = naive_na2d_forward(q_r, k_r, v_r, kernel_size=ks, is_causal=(True, True))
        ref_out.sum().backward()

        torch.testing.assert_close(q_m.grad, q_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_m.grad, k_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v_m.grad, v_r.grad.to("mps"), atol=1e-4, rtol=1e-4)

    @skip_if_no_metal
    def test_na2d_forward_gqa(self):
        """NA2D forward with GQA (heads_q=4, heads_kv=2)."""
        torch.manual_seed(42)
        B, Hsp, Wsp, D = 1, 8, 8, 32
        HQ, HKV = 4, 2
        q = torch.randn(B, Hsp, Wsp, HQ, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, Hsp, Wsp, HKV, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, Hsp, Wsp, HKV, D, device="mps", dtype=torch.float32)

        out = natten.na2d(q, k, v, kernel_size=(3, 3))
        ref_out = naive_na2d_forward(q, k, v, kernel_size=(3, 3))
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na2d_backward_gqa(self):
        """NA2D backward with GQA (heads_q=4, heads_kv=2)."""
        torch.manual_seed(42)
        B, Hsp, Wsp, D = 1, 6, 6, 32
        HQ, HKV = 4, 2

        q_m = torch.randn(B, Hsp, Wsp, HQ, D, device="mps", dtype=torch.float32, requires_grad=True)
        k_m = torch.randn(B, Hsp, Wsp, HKV, D, device="mps", dtype=torch.float32, requires_grad=True)
        v_m = torch.randn(B, Hsp, Wsp, HKV, D, device="mps", dtype=torch.float32, requires_grad=True)

        out_m = natten.na2d(q_m, k_m, v_m, kernel_size=(3, 3))
        out_m.sum().backward()

        q_r = q_m.detach().cpu().float().requires_grad_(True)
        k_r = k_m.detach().cpu().float().requires_grad_(True)
        v_r = v_m.detach().cpu().float().requires_grad_(True)

        ref_out = naive_na2d_forward(q_r, k_r, v_r, kernel_size=(3, 3))
        ref_out.sum().backward()

        torch.testing.assert_close(q_m.grad, q_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_m.grad, k_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v_m.grad, v_r.grad.to("mps"), atol=1e-4, rtol=1e-4)

    @skip_if_no_metal
    def test_na2d_forward_stride(self):
        """NA2D forward with stride=(2,2)."""
        torch.manual_seed(42)
        B, Hsp, Wsp, H, D = 1, 8, 8, 2, 32
        q = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)

        out = natten.na2d(q, k, v, kernel_size=(3, 3), stride=(2, 2))
        ref_out = naive_na2d_forward(q, k, v, kernel_size=(3, 3), stride=(2, 2))
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)


class MetalNA3DTest(unittest.TestCase):
    """Forward and backward tests for NA3D."""

    @skip_if_no_metal
    def test_na3d_forward_fp32(self):
        """NA3D forward with full window."""
        torch.manual_seed(42)
        B, D1, D2, D3, H, D = 1, 3, 3, 3, 2, 32
        q = torch.randn(B, D1, D2, D3, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, D1, D2, D3, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, D1, D2, D3, H, D, device="mps", dtype=torch.float32)

        # Full window = full attention
        out_full = natten.na3d(q, k, v, kernel_size=(D1, D2, D3))

        from natten.backends.metal import metal_fmha
        S = D1 * D2 * D3
        q_flat = q.reshape(B, S, H, D)
        k_flat = k.reshape(B, S, H, D)
        v_flat = v.reshape(B, S, H, D)
        out_fmha = metal_fmha(q_flat, k_flat, v_flat, is_causal=False)
        out_fmha = out_fmha.reshape(B, D1, D2, D3, H, D)

        torch.testing.assert_close(out_full, out_fmha, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na3d_forward_small_window(self):
        """NA3D forward with small kernel_size=(3,3,3) — exercises actual neighborhood masking."""
        torch.manual_seed(42)
        B, D1, D2, D3, H, D = 1, 5, 5, 5, 2, 32
        q = torch.randn(B, D1, D2, D3, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, D1, D2, D3, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, D1, D2, D3, H, D, device="mps", dtype=torch.float32)

        out_small = natten.na3d(q, k, v, kernel_size=(3, 3, 3))
        out_full = natten.na3d(q, k, v, kernel_size=(D1, D2, D3))

        # Small window should NOT match full window (neighborhood masking is active)
        self.assertFalse(torch.allclose(out_small, out_full, atol=1e-5))

        # Verify shape
        self.assertEqual(out_small.shape, (B, D1, D2, D3, H, D))

    @skip_if_no_metal
    def test_na3d_backward(self):
        """NA3D backward with full window — compare against FMHA backward."""
        torch.manual_seed(42)
        B, D1, D2, D3, H, D = 1, 3, 3, 3, 2, 32
        S = D1 * D2 * D3

        q_3d = torch.randn(B, D1, D2, D3, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        k_3d = torch.randn(B, D1, D2, D3, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        v_3d = torch.randn(B, D1, D2, D3, H, D, device="mps", dtype=torch.float32, requires_grad=True)

        out_3d = natten.na3d(q_3d, k_3d, v_3d, kernel_size=(D1, D2, D3))
        out_3d.sum().backward()

        # Reference: CPU naive FMHA on flattened tensors
        q_r = q_3d.detach().reshape(B, S, H, D).cpu().float().requires_grad_(True)
        k_r = k_3d.detach().reshape(B, S, H, D).cpu().float().requires_grad_(True)
        v_r = v_3d.detach().reshape(B, S, H, D).cpu().float().requires_grad_(True)

        out_r, _ = naive_fmha_forward(q_r, k_r, v_r, is_causal=False)
        out_r.sum().backward()

        torch.testing.assert_close(q_3d.grad, q_r.grad.reshape(B, D1, D2, D3, H, D).to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_3d.grad, k_r.grad.reshape(B, D1, D2, D3, H, D).to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v_3d.grad, v_r.grad.reshape(B, D1, D2, D3, H, D).to("mps"), atol=1e-4, rtol=1e-4)


class MetalStrideAndDilationTest(unittest.TestCase):
    """Tests for stride > 1 and dilation > 1."""

    @skip_if_no_metal
    def test_na1d_stride(self):
        """NA1D forward with stride=2 — compare against naive reference."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=5, stride=2)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5, stride=2)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na1d_dilation(self):
        """NA1D forward with dilation=2 — compare against naive reference."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=5, dilation=2)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5, dilation=2)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_na1d_stride_backward(self):
        """NA1D backward with stride=2 — compare against naive reference autograd."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32

        q_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        k_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        v_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)

        out_m = natten.na1d(q_m, k_m, v_m, kernel_size=5, stride=2)
        out_m.sum().backward()

        q_r = q_m.detach().cpu().float().requires_grad_(True)
        k_r = k_m.detach().cpu().float().requires_grad_(True)
        v_r = v_m.detach().cpu().float().requires_grad_(True)

        ref_out, _ = naive_na1d_forward(q_r, k_r, v_r, kernel_size=5, stride=2)
        ref_out.sum().backward()

        torch.testing.assert_close(q_m.grad, q_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_m.grad, k_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v_m.grad, v_r.grad.to("mps"), atol=1e-4, rtol=1e-4)

    @skip_if_no_metal
    def test_na1d_dilation_backward(self):
        """NA1D backward with dilation=2 — compare against naive reference autograd."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32

        q_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        k_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        v_m = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)

        out_m = natten.na1d(q_m, k_m, v_m, kernel_size=5, dilation=2)
        out_m.sum().backward()

        q_r = q_m.detach().cpu().float().requires_grad_(True)
        k_r = k_m.detach().cpu().float().requires_grad_(True)
        v_r = v_m.detach().cpu().float().requires_grad_(True)

        ref_out, _ = naive_na1d_forward(q_r, k_r, v_r, kernel_size=5, dilation=2)
        ref_out.sum().backward()

        torch.testing.assert_close(q_m.grad, q_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_m.grad, k_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v_m.grad, v_r.grad.to("mps"), atol=1e-4, rtol=1e-4)

    @skip_if_no_metal
    def test_na2d_dilation(self):
        """NA2D forward with dilation=(2,2) — compare against naive 2D reference."""
        torch.manual_seed(42)
        B, Hsp, Wsp, H, D = 1, 8, 8, 2, 32
        q = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)

        out = natten.na2d(q, k, v, kernel_size=(3, 3), dilation=(2, 2))
        ref_out = naive_na2d_forward(q, k, v, kernel_size=(3, 3), dilation=(2, 2))
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)


class MetalAdditionalKVTest(unittest.TestCase):
    """Tests for additional KV tokens (cross-attention)."""

    @skip_if_no_metal
    def test_na1d_additional_kv_forward(self):
        """NA1D forward with additional_keys/values."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        num_extra = 4

        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        ak = torch.randn(B, num_extra, H, D, device="mps", dtype=torch.float32)
        av = torch.randn(B, num_extra, H, D, device="mps", dtype=torch.float32)

        from natten.backends.metal import metal_fna_generic
        out = metal_fna_generic(
            q, k, v,
            kernel_size=5,
            additional_keys=ak,
            additional_values=av,
        )
        assert out.shape == (B, S, H, D)

    @skip_if_no_metal
    def test_na1d_additional_kv_backward(self):
        """NA1D backward with additional_keys/values — determinism + non-zero + gradient flow."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 32
        num_extra = 4

        q1 = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        k1 = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        v1 = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        ak1 = torch.randn(B, num_extra, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        av1 = torch.randn(B, num_extra, H, D, device="mps", dtype=torch.float32, requires_grad=True)

        from natten.backends.metal import metal_fna_generic
        out1 = metal_fna_generic(q1, k1, v1, kernel_size=5, additional_keys=ak1, additional_values=av1)
        out1.sum().backward()

        # Second run with same data
        q2 = q1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        v2 = v1.detach().clone().requires_grad_(True)
        ak2 = ak1.detach().clone().requires_grad_(True)
        av2 = av1.detach().clone().requires_grad_(True)

        out2 = metal_fna_generic(q2, k2, v2, kernel_size=5, additional_keys=ak2, additional_values=av2)
        out2.sum().backward()

        # Determinism
        torch.testing.assert_close(q1.grad, q2.grad, atol=0, rtol=0)
        torch.testing.assert_close(k1.grad, k2.grad, atol=0, rtol=0)
        torch.testing.assert_close(v1.grad, v2.grad, atol=0, rtol=0)
        torch.testing.assert_close(ak1.grad, ak2.grad, atol=0, rtol=0)
        torch.testing.assert_close(av1.grad, av2.grad, atol=0, rtol=0)

        # Non-trivial gradients (all 5 tensors should receive gradient flow)
        assert q1.grad.abs().sum() > 0
        assert k1.grad.abs().sum() > 0
        assert v1.grad.abs().sum() > 0
        assert ak1.grad.abs().sum() > 0, "additional_keys should receive gradients"
        assert av1.grad.abs().sum() > 0, "additional_values should receive gradients"

        # Verify shapes
        assert q1.grad.shape == q1.shape
        assert ak1.grad.shape == ak1.shape
        assert av1.grad.shape == av1.shape


class MetalSmokeTest(unittest.TestCase):
    """Quick integration smoke tests."""

    @skip_if_no_metal
    def test_na1d_forward_backward_smoke(self):
        """End-to-end: na1d forward + backward on MPS."""
        q = torch.randn(1, 16, 4, 64, device="mps", requires_grad=True)
        k = torch.randn(1, 16, 4, 64, device="mps", requires_grad=True)
        v = torch.randn(1, 16, 4, 64, device="mps", requires_grad=True)
        out = natten.na1d(q, k, v, kernel_size=5)
        out.sum().backward()
        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(k.grad.shape, k.shape)
        self.assertEqual(v.grad.shape, v.shape)

    @skip_if_no_metal
    def test_fmha_forward_backward_smoke(self):
        """End-to-end: FMHA forward + backward on MPS."""
        from natten.backends.metal import metal_fmha
        q = torch.randn(1, 16, 4, 64, device="mps", requires_grad=True)
        k = torch.randn(1, 16, 4, 64, device="mps", requires_grad=True)
        v = torch.randn(1, 16, 4, 64, device="mps", requires_grad=True)
        out = metal_fmha(q, k, v)
        out.sum().backward()
        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(k.grad.shape, k.shape)
        self.assertEqual(v.grad.shape, v.shape)

    @skip_if_no_metal
    def test_requires_grad_accepted(self):
        """Verify that requires_grad=True tensors are accepted by Metal backend."""
        from natten.backends.configs.checks import can_run_metal_fna, can_run_metal_fmha
        q = torch.randn(1, 16, 4, 64, device="mps", requires_grad=True)
        k = torch.randn(1, 16, 4, 64, device="mps")
        v = torch.randn(1, 16, 4, 64, device="mps")
        self.assertTrue(can_run_metal_fna(q, k, v))
        self.assertTrue(can_run_metal_fmha(q, k, v, is_causal=False, is_varlen=False))


class MetalTiledForwardTest(unittest.TestCase):
    """Tests that the tiled flash-attention forward kernel matches the naive reference.
    The tiled kernel is used for D<=128; these tests cover NA1D, NA2D, various kernel sizes,
    D=32, D=64, and D=128, plus FP16 and causal modes.
    """

    @skip_if_no_metal
    def test_tiled_na1d_d32_ks5(self):
        """NA1D D=32, kernel_size=5 — exercises Br=32 path."""
        torch.manual_seed(42)
        B, S, H, D = 1, 64, 4, 32
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=5)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_tiled_na1d_d64_ks7(self):
        """NA1D D=64, kernel_size=7 — exercises Br=32 path."""
        torch.manual_seed(42)
        B, S, H, D = 2, 128, 4, 64
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=7)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=7)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_tiled_na1d_d128_ks5(self):
        """NA1D D=128, kernel_size=5 — exercises Br=16 path."""
        torch.manual_seed(42)
        B, S, H, D = 1, 64, 4, 128
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=5)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_tiled_na1d_full_window(self):
        """NA1D with kernel_size=seqlen (full attention) via tiled kernel."""
        torch.manual_seed(42)
        B, S, H, D = 1, 64, 2, 64
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=S)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=S)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_tiled_na1d_even_kernel(self):
        """NA1D with even kernel_size=6 via tiled kernel."""
        torch.manual_seed(42)
        B, S, H, D = 1, 64, 2, 64
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=6)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=6)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_tiled_na1d_causal(self):
        """NA1D causal via tiled kernel."""
        torch.manual_seed(42)
        B, S, H, D = 1, 64, 2, 64
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=7, is_causal=True)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=7, is_causal=True)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_tiled_na1d_fp16(self):
        """NA1D FP16 via tiled kernel."""
        torch.manual_seed(42)
        B, S, H, D = 1, 64, 2, 64
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float16)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float16)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float16)

        out = natten.na1d(q, k, v, kernel_size=7)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=7)
        torch.testing.assert_close(out, ref_out, atol=2e-3, rtol=2e-3)

    @skip_if_no_metal
    def test_tiled_na1d_gqa(self):
        """NA1D with GQA via tiled kernel."""
        torch.manual_seed(42)
        B, S, D = 1, 64, 64
        HQ, HKV = 8, 2
        q = torch.randn(B, S, HQ, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, HKV, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, HKV, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=7)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=7)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_tiled_na2d_small_window(self):
        """NA2D k=(3,3) via tiled kernel — the main performance target."""
        torch.manual_seed(42)
        B, Hsp, Wsp, H, D = 1, 16, 16, 2, 64
        q = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)

        out = natten.na2d(q, k, v, kernel_size=(3, 3))
        ref_out = naive_na2d_forward(q, k, v, kernel_size=(3, 3))
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_tiled_na2d_k13(self):
        """NA2D k=(13,13) — the benchmark scenario from the plan."""
        torch.manual_seed(42)
        B, Hsp, Wsp, H, D = 1, 16, 16, 2, 64
        q = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)

        out = natten.na2d(q, k, v, kernel_size=(13, 13))
        ref_out = naive_na2d_forward(q, k, v, kernel_size=(13, 13))
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_tiled_na2d_full_window(self):
        """NA2D full window via tiled kernel."""
        torch.manual_seed(42)
        B, Hsp, Wsp, H, D = 1, 8, 8, 2, 64
        q = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, Hsp, Wsp, H, D, device="mps", dtype=torch.float32)

        out = natten.na2d(q, k, v, kernel_size=(Hsp, Wsp))
        ref_out = naive_na2d_forward(q, k, v, kernel_size=(Hsp, Wsp))
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_tiled_na1d_dilation(self):
        """NA1D with dilation=2 via tiled kernel."""
        torch.manual_seed(42)
        B, S, H, D = 1, 64, 2, 64
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=5, dilation=2)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5, dilation=2)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    @skip_if_no_metal
    def test_tiled_na1d_stride(self):
        """NA1D with stride=2 via tiled kernel."""
        torch.manual_seed(42)
        B, S, H, D = 1, 64, 2, 64
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32)

        out = natten.na1d(q, k, v, kernel_size=5, stride=2)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5, stride=2)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)


class MetalTiledBackwardTest(unittest.TestCase):
    """Tests for tiled backward kernels (flash-attention style)."""

    def _compare_backward(self, B, S, H, D, KS, dtype=torch.float32,
                          is_causal=False, heads_kv=None, atol=1e-4, rtol=1e-3):
        """Run metal backward and compare against naive reference backward."""
        torch.manual_seed(42)
        HKV = heads_kv or H
        q = torch.randn(B, S, H, D, device="mps", dtype=dtype, requires_grad=True)
        k = torch.randn(B, S, HKV, D, device="mps", dtype=dtype, requires_grad=True)
        v = torch.randn(B, S, HKV, D, device="mps", dtype=dtype, requires_grad=True)

        # Metal backward
        o = natten.na1d(q, k, v, kernel_size=KS, is_causal=is_causal)
        o.sum().backward()
        torch.mps.synchronize()
        dq_m, dk_m, dv_m = q.grad.clone(), k.grad.clone(), v.grad.clone()

        # Naive reference backward via autograd
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        ref_out, _ = naive_na1d_forward(q2, k2, v2, kernel_size=KS, is_causal=is_causal)
        ref_out.sum().backward()
        torch.mps.synchronize()

        torch.testing.assert_close(dq_m.float(), q2.grad.float(), atol=atol, rtol=rtol)
        torch.testing.assert_close(dk_m.float(), k2.grad.float(), atol=atol, rtol=rtol)
        torch.testing.assert_close(dv_m.float(), v2.grad.float(), atol=atol, rtol=rtol)

    @skip_if_no_metal
    def test_tiled_bwd_d32_ks5(self):
        self._compare_backward(1, 32, 2, 32, 5)

    @skip_if_no_metal
    def test_tiled_bwd_d64_ks7(self):
        self._compare_backward(1, 64, 2, 64, 7)

    @skip_if_no_metal
    def test_tiled_bwd_d128_ks5(self):
        self._compare_backward(1, 32, 2, 128, 5)

    @skip_if_no_metal
    def test_tiled_bwd_even_kernel(self):
        self._compare_backward(1, 32, 2, 32, 6)

    @skip_if_no_metal
    def test_tiled_bwd_full_window(self):
        """FMHA case: kernel_size = seqlen."""
        self._compare_backward(1, 16, 2, 32, 16)

    @skip_if_no_metal
    def test_tiled_bwd_causal(self):
        self._compare_backward(1, 32, 2, 32, 7, is_causal=True)

    @skip_if_no_metal
    def test_tiled_bwd_gqa(self):
        self._compare_backward(1, 32, 4, 32, 5, heads_kv=2)

    @skip_if_no_metal
    def test_tiled_bwd_fp16(self):
        self._compare_backward(1, 32, 2, 64, 5, dtype=torch.float16, atol=5e-2, rtol=5e-2)

    @skip_if_no_metal
    def test_tiled_bwd_bf16(self):
        """Backward with BF16."""
        self._compare_backward(1, 32, 2, 32, 5, dtype=torch.bfloat16, atol=5e-2, rtol=5e-2)

    @skip_if_no_metal
    def test_tiled_bwd_na2d(self):
        """NA2D backward — numerical correctness against naive 2D reference."""
        torch.manual_seed(42)
        B, Sx, Sy, H, D = 1, 8, 8, 2, 32
        ks = (5, 5)

        q_m = torch.randn(B, Sx, Sy, H, D, device="mps", requires_grad=True)
        k_m = torch.randn(B, Sx, Sy, H, D, device="mps", requires_grad=True)
        v_m = torch.randn(B, Sx, Sy, H, D, device="mps", requires_grad=True)

        out_m = natten.na2d(q_m, k_m, v_m, kernel_size=ks)
        out_m.sum().backward()
        torch.mps.synchronize()

        q_r = q_m.detach().cpu().float().requires_grad_(True)
        k_r = k_m.detach().cpu().float().requires_grad_(True)
        v_r = v_m.detach().cpu().float().requires_grad_(True)

        ref_out = naive_na2d_forward(q_r, k_r, v_r, kernel_size=ks)
        ref_out.sum().backward()

        torch.testing.assert_close(q_m.grad, q_r.grad.to("mps"), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(k_m.grad, k_r.grad.to("mps"), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(v_m.grad, v_r.grad.to("mps"), atol=1e-4, rtol=1e-3)

    @skip_if_no_metal
    def test_reference_fallback_d256(self):
        """D=256 falls back to reference kernel (tiled only supports D<=128)."""
        torch.manual_seed(42)
        B, S, H, D = 1, 16, 2, 256
        q = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, S, H, D, device="mps", dtype=torch.float32, requires_grad=True)

        out = natten.na1d(q, k, v, kernel_size=5)
        ref_out, _ = naive_na1d_forward(q, k, v, kernel_size=5)
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

        out.sum().backward()
        torch.mps.synchronize()

        q_r = q.detach().cpu().float().requires_grad_(True)
        k_r = k.detach().cpu().float().requires_grad_(True)
        v_r = v.detach().cpu().float().requires_grad_(True)
        ref_out2, _ = naive_na1d_forward(q_r, k_r, v_r, kernel_size=5)
        ref_out2.sum().backward()

        torch.testing.assert_close(q.grad, q_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.grad, k_r.grad.to("mps"), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v.grad, v_r.grad.to("mps"), atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
