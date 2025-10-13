import click

import torch


from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention

HAS_FLASH_ATTN = False
try:
    from natten import libnatten  # noqa: 
    flash_attn_func = libnatten.flash_fmha_forward
    print("Using Flash Attention 2 for SM80")
    HAS_FLASH_ATTN = True
except ImportError:
    raise ImportError("Could not import flash_fmha_forward")


class Problem:
    def __init__(
        self,
        b: int = 1,
        h: int = 24,
        q: int = 1019,
        k: int = 1024,
        d: int = 128,
        seed: int = 42,
        dtype: torch.dtype = torch.float16,
        ref_backend: str = "cudnn",
    ):
        self.b = b
        self.h = h
        self.q = q
        self.k = k
        self.d = d
        self.seed = seed
        self.dtype = dtype
        self.ref_backend = ref_backend

    def __repr__(self):
        return (
            f"Problem(b={self.b}, h={self.h}, q={self.q}, k={self.k}, d={self.d}, "
            f"dtype={self.dtype}, ref_backend='{self.ref_backend}', seed='{self.seed}')"
        )


def test_fa2_allclose(ps: Problem):

    q_shape = (ps.b, ps.h, ps.q, ps.d)
    kv_shape = (ps.b, ps.h, ps.k, ps.d)
    pt_generator = torch.Generator(device="cuda").manual_seed(ps.seed)

    torch.cuda.synchronize()

    q_ref = torch.randn( q_shape, dtype=ps.dtype, device="cuda", generator=pt_generator)
    k_ref = torch.randn(kv_shape, dtype=ps.dtype, device="cuda", generator=pt_generator)
    v_ref = torch.randn(kv_shape, dtype=ps.dtype, device="cuda", generator=pt_generator)
    do_ref = torch.randn(q_shape, dtype=ps.dtype, device="cuda", generator=pt_generator)

    q_pt = q_ref.clone()
    k_pt = k_ref.clone()
    v_pt = v_ref.clone()
    do_pt = do_ref.clone()

    q_fa = q_ref.clone()
    k_fa = k_ref.clone()
    v_fa = v_ref.clone()
    do_fa = do_ref.clone()

    q_fa = q_fa.transpose(1, 2).contiguous()
    k_fa = k_fa.transpose(1, 2).contiguous()
    v_fa = v_fa.transpose(1, 2).contiguous()
    do_fa = do_fa.transpose(1, 2).contiguous()

    torch.cuda.synchronize()

    q_pt.requires_grad_(True)
    k_pt.requires_grad_(True)
    v_pt.requires_grad_(True)
    do_pt.requires_grad_(True)

    q_fa.requires_grad_(True)
    k_fa.requires_grad_(True)
    v_fa.requires_grad_(True)
    do_fa.requires_grad_(True)

    torch.cuda.synchronize()

    if ps.ref_backend == "xformers":
        backends = [SDPBackend.EFFICIENT_ATTENTION]

    elif ps.ref_backend == "cudnn":
        backends = [SDPBackend.CUDNN_ATTENTION]

    else:
        raise ValueError(f"`ref_backend` must be one of `xformers` or `cudnn`, got {ref_backend}.")

    with sdpa_kernel(backends=backends):
        o_pt = scaled_dot_product_attention(q_pt, k_pt, v_pt)
        torch.cuda.synchronize()
        o_pt.backward(gradient=do_pt)
        torch.cuda.synchronize()

        assert q_pt.grad is not None
        assert k_pt.grad is not None
        assert v_pt.grad is not None

    o_fa = torch.empty_like(q_fa)
    lse_fa = torch.empty((ps.b, ps.q, ps.h), dtype=torch.float32, device="cuda")
    flash_attn_func(o_fa, q_fa, k_fa, v_fa, lse_fa, 1/(ps.d ** 0.5), 64, 128)
    torch.cuda.synchronize()
    # o_fa.backward(gradient=do_fa)
    torch.cuda.synchronize()

    torch.testing.assert_close(o_fa.transpose(1, 2), o_pt, atol=1e-3, rtol=0.)
    # torch.testing.assert_close(q_fa.grad.transpose(1, 2), q_pt.grad, atol=1e-3, rtol=0.)
    # torch.testing.assert_close(k_fa.grad.transpose(1, 2), k_pt.grad, atol=1e-3, rtol=0.)
    # torch.testing.assert_close(v_fa.grad.transpose(1, 2), v_pt.grad, atol=1e-3, rtol=0.)


@click.command()
@click.option('--b', default=3, type=int, help="Batch size")
@click.option('--h', default=49, type=int, help="Number of heads")
@click.option('--q', default=1019, type=int, help="Query length")
@click.option('--k', default=1019, type=int, help="Key length")
@click.option('--d', default=128, type=int, help="Head dimension")
@click.option(
    '--dtype',
    default='float16',
    type=click.Choice(['float16', 'float32', 'bfloat16']),
    show_default=True,
    help="Torch dtype"
)
@click.option(
    '--ref-backend',
    default='cudnn',
    type=click.Choice(['cudnn', 'xformers']),
    show_default=True,
    help="Reference backend"
)
def main(b, h, q, k, d, dtype, ref_backend):
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[dtype]

    print("====================== Testing allclose ====================")
    problem = Problem(b=b, h=h, q=q, k=k, d=d, dtype=dtype, ref_backend=ref_backend)
    click.echo(problem)

    test_fa2_allclose(problem)


if __name__ == "__main__":
    main()
