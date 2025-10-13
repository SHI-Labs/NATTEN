import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F

from importlib.metadata import version


try:
    import flash_attn
    from flash_attn import flash_attn_func
    from flash_attn import cute
    print(version("flash_attn"))
    print(f"Using FAv2")
except:
    # from flash_attn_3 import flash_attn_interface
    import flash_attn_3.flash_attn_interface as faint
    print(dir(flash_attn_3))
    print(flash_attn_3.__name__)
    print(flash_attn_3.__path__)
    print(flash_attn_3.__spec__)
    print(version("flash_attn_3"))
    # print(flash_attn_3.__version__) # Errors out
    # print(flash_attn_interface.__version__) # Errors out

    import flash_attn_3 # Does not do anything
    from flash_attn_interface import flash_attn_func # This is the actual FAv3 function
    print(version("flash_attn_interface"))


    # print(dir(flash_attn_interface.__init__.__qualname__))
    # print(flash_attn_interface.__name__)
    # print(flash_attn_interface.__spec__)
    # print(flash_attn_interface.__package__)
    print(f"Using FAv3")


B = 1
H = 8
L = 1024
D = 64
tensor_shape = (B, L, H, D)

q_ref = torch.randn(tensor_shape, dtype=torch.bfloat16, device="cuda:1")
k_ref = torch.randn(tensor_shape, dtype=torch.bfloat16, device="cuda:1")
v_ref = torch.randn(tensor_shape, dtype=torch.bfloat16, device="cuda:1")

q_cpp = q_ref.clone()
k_cpp = k_ref.clone()
v_cpp = v_ref.clone()

# q_cute = q_ref.clone()
# k_cute = k_ref.clone()
# v_cute = v_ref.clone()

cpp_result = flash_attn_func(q_cpp, k_cpp, v_cpp)

with sdpa_kernel(backends=[SDPBackend.MATH]):
    ref_result = F.scaled_dot_product_attention(q_ref.transpose(1, 2), k_ref.transpose(1, 2),
                                                v_ref.transpose(1, 2))

# cute_result = cute.interface.flash_attn_func(q_cute, k_cute, v_cute)

torch.testing.assert_close(cpp_result, ref_result.transpose(1, 2), atol=1e-2, rtol=0)
