# Profiler

We offer a profiling toolkit in NATTEN, designed to allow easy measurement of NATTEN's performance
on your device and given your desired use case. You can also use it to compare against baselines
available in PyTorch's SDPA (cuDNN Attention and Flash Attention v2), as well as
[self attention operators in NATTEN][natten.attention].

It directly uses the [PyTorch profiler API](https://docs.pytorch.org/docs/stable/profiler.html) to
extract the trace of operations, maps known symbol names to human readable operation names, and
filters out the less relevant and mutual operations (i.e. device synchronize).

It also provides an interface for exploring [backends](backends.md), their
[configurations](#dry-run), and automatically [searching](#optimize) for the best configuration.

## Getting Started

You can access the profiler with `python -m natten.profiler`.

??? question "Dependencies"
    We recommend installing `rich` and `tqdm` (1) for the best visual experience, but they are not
    required.
    { .annotate }

    1. 
        ```shell
        pip install rich tqdm
        ```

For example, let's say we want to profile a 3D use case, where our token layout (feature map) shape
is 8 x 16 x 24, and we have head dim 128:

```bash
python -m natten.profiler \
    -i 8 16 24 \ #(1)!
    -d 128 #(2)!
```

1. Token layout (feature map) shape: `(8, 16, 24)`
2. Head dim 128

This will report the default [self attention][natten.attention] time.
Now let's say we want to profile this with window size 2 x 4 x 3.
We can use `-w` to specify `kernel_size`:

```bash
python -m natten.profiler \
    -i 8 16 24 \
    -d 128 \
    -w 2 4 3
```

There's so many more options available. 
Not only can you specify all the neighborhood attention parameters (`kernel_size`, `dilation`,
`stride`, `is_causal`), you can also toggle backward pass, play around with different
[backends](backends.md), data types, backend configurations, and also find out what backends and
configurations are available on your GPU and for your specific use case.

Refer to [Arguments](#arguments) for a detailed list of the options, and our examples below
highlighting [dry run mode](#dry-run), which lists out all available backends and their
configurations, and [optimize mode](#optimize), which picks a backend and configuration for you by
running reasonable choices and finding the fastest one.

There's also some [examples](#hopper-and-blackwell-examples) highlighting the performance of our
new Hopper FNA and Blackwell FNA kernels.

## Arguments

+-----------------------------+--------------------------------------------------------------------+
| Option                      | Description                                                        |
+=============================+====================================================================+
| `-h`, `#!bash --help`       | Show help message and exit.                                        |
+-----------------------------+--------------------------------------------------------------------+
| `-i`,                       | **Required** QKV token layout shape (i.e. sequence length in 1-D,  |
| `#!bash --input-size`       | height and width in 2-D, depth, height, and width in 3-D).         |
|                             | ???+ info                                                          |
|                             |     `-i 16` is interpreted as 1-D (neighborhood) attention with    |
|                             |     sequence length `16`.                                          |
|                             |                                                                    |
|                             |     `-i 16 32` is interpreted as 2-D (neighborhood) attention with |
|                             |     token layout shape (feature map shape) `(16, 32)`.             |
|                             |                                                                    |
|                             |     `-i 16 32 24` is interpreted as 3-D (neighborhood) attention   |
|                             |     with token layout shape (feature map shape) `(16, 32, 24)`.    |
+-----------------------------+--------------------------------------------------------------------+
| `-b`, `--batch-size`        | QKV batch size. Default: `1`.                                      |
+-----------------------------+--------------------------------------------------------------------+
| `-n`, `--heads`             | QKV number of heads (GQA/MQA are not supported in NATTEN at this   |
|                             | time). Default: `1`.                                               |
+-----------------------------+--------------------------------------------------------------------+
| `-d`, `--dim`               | QK (and optionally V) head dim. Default: `64`.                     |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --dim-value`        | Head dim for the V tensor, if different from Q and K.              |
|                             | Defaults to the value of `-d`/`--dim`.                             |
+-----------------------------+--------------------------------------------------------------------+
| `-w`,                       | Neighborhood attention window size (shape), also referred to as    |
| `#!bash --window-size`      | `#!bash kernel_size`. This must be a tuple with the same number of |
|                             | elements as in `#!bash --input-size`. Defaults to                  |
|                             | `#!bash --input-size` (self attention).                            |
+-----------------------------+--------------------------------------------------------------------+
| `-s`, `--stride`            | Neighborhood attention stride values. This must be a tuple with    |
|                             | the same number of elements as in `#!bash --input-size`. Defaults  |
|                             | to 1s (standard sliding window).                                   |
+-----------------------------+--------------------------------------------------------------------+
| `--dilation`                | Neighborhood attention dilation values. This must be a tuple with  |
|                             | the same number of elements as in `#!bash --input-size`. Defaults  |
|                             | to 1s (standard sliding window).                                   |
+-----------------------------+--------------------------------------------------------------------+
| `-c`, `--causal`            | Causal masking values. This must be a boolean tuple with the same  |
|                             | number of elements as in `#!bash --input-size`. Defaults to        |
|                             | `False`s (bi-directional in all dimensions).                       |
+-----------------------------+--------------------------------------------------------------------+
| `--dtype`                   | Element (data) type. Choices: `fp32`, `bf16`, `fp16`, `e4m3`,      |
|                             | `e5m2`.                                                            |
+-----------------------------+--------------------------------------------------------------------+
| `--backprop`                | Profile backward pass as well as forward pass.                     |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --add-kv`           | Number of additional KV tokens, if desired. Defaults to 0.         |
+-----------------------------+--------------------------------------------------------------------+
| `--backend`                 | Backend / kernel to run.                                           |
|                             |                                                                    |
|                             |   Choices:                                                         |
|                             |                                                                    |
|                             |   **NATTEN backends**: `cutlass-fna`, `blackwell-fna`,             |
|                             |   `hopper-fna`, `flex-fna`.                                        |
|                             |                                                                    |
|                             |   **PyTorch SDPA backends** (can only perform self attention):     |
|                             |   `xformers`, `cudnn`, `fav2`.                                     |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --fmha-backend`     | Backend / kernel for cross-attention (additional KV) and fast-path |
|                             | self attention in NATTEN.                                          |
|                             |                                                                    |
|                             |   Choices:                                                         |
|                             |   `cutlass-fmha`, `blackwell-fmha`, `hopper-fmha`, `flex-fna`.     |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --q-tile`           | Q tile shape in the kernel (varies between different backends).    |
|                             | Run with `#!bash --dry-run` to see available tile shapes for your  |
|                             | use case.                                                          |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --kv-tile`          | KV tile shape in the kernel (varies between different backends).   |
|                             | Run with `#!bash --dry-run` to see available tile shapes for your  |
|                             | use case.                                                          |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --backward-q-tile`  | Q tile shape in the **backward pass** kernel (not respected by     |
|                             | the `flex-fna`/`flex-fmha` backends yet).                          |
|                             | Run with `#!bash --dry-run` to see available tile shapes for your  |
|                             | use case.                                                          |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --backward-kv-tile` | KV tile shape in the **backward pass** kernel (not respected by    |
|                             | the `flex-fna`/`flex-fmha` backends yet).                          |
|                             | Run with `#!bash --dry-run` to see available tile shapes for your  |
|                             | use case.                                                          |
+-----------------------------+--------------------------------------------------------------------+
| `--schedule`                | Kernel schedule (`hopper-fna` and `hopper-fmha` backends only).    |
|                             | This is only an option for forward pass.                           |
|                             | Choices: `non` (non-persistent), `coop` (cooperative), `pp`        |
|                             | (ping-pong).                                                       |
+-----------------------------+--------------------------------------------------------------------+
| `--persistent`              | Use persistent tile scheduler in `blackwell-fna` `blackwell-fmha`  |
|                             | backends. This is only an option for forward pass.                 |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --compile`          | Enables compiling Flex Attention block sparse mask and kernel in   |
|                             | `flex-fna` and `flex-fmha` backends.                               |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --warmup-steps`     | Number of profiling warmup steps. Default: `10`.                   |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --dry-run`          | Display valid forward and backward pass configurations for this    |
|                             | use case and exit.                                                 |
|                             |                                                                    |
|                             | [Learn more about this feature](#dry-run)                          |
|                             |                                                                    |
|                             | ???+ note                                                          |
|                             |     your default CUDA device (or CPU if CUDA is not available)     |
|                             |     will be a determining factor in the tile shapes /              |
|                             |     configurations shown. For instance, some tile shapes may only  |
|                             |     be available for specific GPU architectures, or if your GPU    |
|                             |     architecture does not support a specific backend, you will see |
|                             |     an empty list.                                                 |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --max-configs`      | Maximum number of tile configurations to display. Default: `10`.   |
|                             | If set to `0` shows all valid configurations.                      |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash --optimize`         | Find the best configuration (and backend if unspecified) for your  |
|                             | use case, by profiling all available choices, and selecting the    | 
|                             | fastest one.                                                       |
|                             |                                                                    |
|                             | ???+ warning "Experimental feature"                                |
|                             |     This feature is experimental, and may change in future         |
|                             |     releases. We hope to eventually integrate / replace this with  |
|                             |     **[NATTEN Simulator](https://arxiv.org/abs/2504.16922)**.      |
|                             |                                                                    |
|                             |     [Learn more about this feature](#optimize)                     |
+-----------------------------+--------------------------------------------------------------------+
| `#!bash                     | Number of warmup steps for optimizer. Defaults to `5`.             |
| --optimize-warmup-steps`    |                                                                    |
+-----------------------------+--------------------------------------------------------------------+

## Dry run
*Figuring out tile sizes for your use case.*

We offer many different [backends](backends.md), and each backend can offer various configurations,
mainly tile sizes / shapes, for each unique use case.
Factors that determine those include, but are not limited to:

1. GPU architecture
2. Element type (`float16`/`bfloat16` vs `float32`)
3. Attention head dim

Due to this, we highly recommend first trying to understand what options you have, by using the
profiler's *dry run* mode:

```shell title="Dry run for bf16 3D problem, with head dim 128"
python -m natten.profiler \
    --dry-run \
    --dtype bf16 \
    -i 16 16 16 \ #(1)!
    -d 128   #(2)!
```

1. Sample feature map shape `(16, 16, 16)`
2. 128 head dim

Since a backend was not specified, profiler will first detect all compatible backends with the
specified options for your default GPU, and print out compatible tile shapes for each available
backend in a separate table.

=== "Hopper"

    ??? hopper-example "Sample output from running on H100"
        ```
        --8<-- "docs/sample-outputs/profiler-dry-run-h100.txt"
        ```

=== "Blackwell"

    ??? blackwell-example "Sample output from running on B200"
        ```
        --8<-- "docs/sample-outputs/profiler-dry-run-b200.txt"
        ```

Note that some backends offer many combinations, and you may want to use `#!bash --max-configs` to
change the default limit from `10`, or set it to `0` to display everything.

!!! tip
    You can also do this directly in your code by using our `get[_bwd]configs_for_{BACKEND}` APIs.
    All you need to do is pass one of your query, key, or value tensors, and they'll return a list
    of valid options. Read more about them in [backends](backends.md).

## Optimize
You can use the profiler to also search and find the fastest backend configurations for your use
case. This can sometimes bring you very significant speedups. However, since running this on some
backends may be time consuming, it is a good idea to run it ahead of your actual program, and then
hard-code your chosen configuration (this is why we removed our
[autotuner feature](https://github.com/SHI-Labs/NATTEN/blob/v0.17.0/docs/fna/autotuner.md)).

As an example, here we demonstrate a 3-D use case on Ampere, where the `cutlass-fna` backend
specifically is the best choice, but also comes with over 70 forward pass and 300 backward pass
configurations just for this use case.

```shell title="Profiler without optimizer"
python -m natten.profiler \
    --backprop \
    -b 1 \
    -n 24 \
    -d 128 \
    -i 30 48 80 \
    -w 18 24 24 \
    -s 16 8 8
```

<div class="result" markdown>

??? ampere-example "Sample output from running on A100"
    ```
    --8<-- "docs/sample-outputs/profiler-optimize-off-cutlass-fna-a100.txt"

    ```
</div>

```shell title="Profiler with optimizer"
python -m natten.profiler \
    --backprop \
    -b 1 \
    -n 24 \
    -d 128 \
    -i 30 48 80 \
    -w 18 24 24 \
    -s 16 8 8 \
    --optimize
```

<div class="result" markdown>

??? ampere-example "Sample output from running on A100"
    ```
    --8<-- "docs/sample-outputs/profiler-optimize-on-cutlass-fna-a100.txt"

    ```
</div>

!!! warning
    Running this took over an hour, since there are so many unique configurations for
    backward pass, and that, well, it's a pretty big use case.
    But remember, this only needs to be done once.

    Alternatively, you can always try to set batch and heads to 1 (when there's at least a few
    thousand tokens), run more quickly, and then use the same config from the `batch=1` `heads=1`
    case. It can get you most of the way, especially on newer architectures that have persistent
    scheduling.

In this case, the default case's runtime was approximately 2 seconds, while the optimized case was
at approximately 1.4 seconds, which is **1.45X speedup**.
If we look at just forward pass, it's ~ 557 ms vs ~ 222 ms, which is **2.5X speedup**!

However, we plan to eventually integrate our new
**[NATTEN Simulator](https://arxiv.org/abs/2504.16922)** with this feature, so that we can optimize
faster by ruling out obviously terrible configurations, and only search a fraction of the total per
use case. In addition, the Simulator can also make other recommendations, like what `stride` and
`dilation` values give you the best speedups!


## Hopper and Blackwell Examples

### 1-D use case

In this example, we try to measure the runtime for self attention, two forms of neighborhood
attention, and blocked attention (also implemented by NATTEN), on a 32K 1-D sequence.


=== "Hopper"

    ```shell title="Baseline (cuDNN)"
    python -m natten.profiler \
        --backend cudnn \
        -d 128 \
        -i 32768
    ```

    <div class="result" markdown>

    ??? hopper-example "Sample output from running on H100"
        ```
        --8<-- "docs/sample-outputs/profiler-1d-32k-cudnn-h100.txt"
        ```

    </div>

=== "Blackwell"

    ```shell title="Baseline (cuDNN)"
    python -m natten.profiler \
        --backend cudnn \
        -d 128 \
        -i 32768
    ```

    <div class="result" markdown>

    ??? blackwell-example "Sample output from running on B200"
        ```
        --8<-- "docs/sample-outputs/profiler-1d-32k-cudnn-b200.txt"
        ```

    </div>




Now let's try standard neighborhood attention, with a 2K window size, which gives us 93.75%
sparsity.


=== "Hopper"

    ```shell title="Neighborhood attention (Hopper FNA) w/ 2K sliding window"
    python -m natten.profiler \
        --backend hopper-fna \
        -d 128 \
        -i 32768 \
        -w 2048
    ```

    <div class="result" markdown>

    ??? hopper-example "Sample output from running on H100"
        ```
        --8<-- "docs/sample-outputs/profiler-1d-32k-w2k-hopper-fna-h100.txt"
        ```

        *FLOP-wise speedup (upper bound) with 93.75% sparsity: 16X*

        Speedup: 787.392 us / 94.848 us = **8.3X**

        !!! note
            1-D non-dilated neighborhood attention does not require token permute, hence the only
            runtime is the kernel runtime.

    </div>


=== "Blackwell"

    ```shell title="Neighborhood attention (Blackwell FNA) w/ 2K sliding window"
    python -m natten.profiler \
        --backend blackwell-fna \
        -d 128 \
        -i 32768 \
        -w 2048
    ```

    <div class="result" markdown>

    ??? blackwell-example "Sample output from running on B200"
        ```
        --8<-- "docs/sample-outputs/profiler-1d-32k-w2k-blackwell-fna-b200.txt"
        ```

        *FLOP-wise speedup (upper bound) with 93.75% sparsity: 16X*

        Speedup: 424.766 us / 58.111 us = **7.3X**

        !!! note
            1-D non-dilated neighborhood attention does not require token permute, hence the only
            runtime is the kernel runtime.

    </div>






Let's try a [strided](https://arxiv.org/abs/2504.16922) variant of neighborhood attention.
We ran the NATTEN Simulator (coming soon), and found that `stride=256` is the minimum stride that
results in a fully block-sparse mask, for both Hopper and Blackwell tile sizes.

=== "Hopper"

    ```shell title="Strided neighborhood attention (Hopper FNA) w/ 2K sliding window and stride 256"
    python -m natten.profiler \
        --backend hopper-fna \
        -d 128 \
        -i 32768 \
        -w 2048 \
        -s 256
    ```

    <div class="result" markdown>

    ??? hopper-example "Sample output from running on H100"
        ```
        --8<-- "docs/sample-outputs/profiler-1d-32k-w2k-s256-hopper-fna-h100.txt"
        ```

        *FLOP-wise speedup (upper bound) with 93.75% sparsity: 16X*

        Speedup: 787.392 us / 62.687 us = **12.6X**

        !!! note
            1-D non-dilated neighborhood attention does not require token permute, hence the only
            runtime is the kernel runtime.

    </div>

=== "Blackwell"

    ```shell title="Strided neighborhood attention (Blakwell FNA) w/ 2K sliding window and stride 256"
    python -m natten.profiler \
        --backend blackwell-fna \
        -d 128 \
        -i 32768 \
        -w 2048 \
        -s 256
    ```

    <div class="result" markdown>

    ??? blackwell-example "Sample output from running on B200"
        ```
        --8<-- "docs/sample-outputs/profiler-1d-32k-w2k-s256-blackwell-fna-b200.txt"
        ```

        *FLOP-wise speedup (upper bound) with 93.75% sparsity: 16X*

        Speedup: 424.766 us / 37.888 us = **11.4X**

        !!! note
            1-D non-dilated neighborhood attention does not require token permute, hence the only
            runtime is the kernel runtime.

    </div>




Finally, we know that when `stride == kernel_size`, neighborhood attention numerically matches
blocked attention. In this use case, blocked attention is also fully block-sparse, and given that it
is using the same window size as the neighborhood attention case, we should expect identical
performance (excluding runtime variance).


=== "Hopper"

    ```shell title="Blocked attention (Hopper FNA) w/ 2K block size"
    python -m natten.profiler \
        --backend hopper-fna \
        -d 128 \
        -i 32768 \
        -w 2048 \
        -s 2048
    ```

    <div class="result" markdown>

    ??? hopper-example "Sample output from running on H100"
        ```
        --8<-- "docs/sample-outputs/profiler-1d-32k-w2k-s2k-hopper-fna-h100.txt"
        ```

        *FLOP-wise speedup (upper bound) with 93.75% sparsity: 16X*

        Speedup: 787.392 us / 59.264 us = **13.3X**

        !!! note
            1-D non-dilated neighborhood attention does not require token permute, hence the only
            runtime is the kernel runtime.

    </div>

=== "Blackwell"

    ```shell title="Blocked attention (Blackwell FNA) w/ 2K block size"
    python -m natten.profiler \
        --backend blackwell-fna \
        -d 128 \
        -i 32768 \
        -w 2048 \
        -s 2048
    ```

    <div class="result" markdown>

    ??? blackwell-example "Sample output from running on B200"
        ```
        --8<-- "docs/sample-outputs/profiler-1d-32k-w2k-s2k-blackwell-fna-b200.txt"
        ```

        *FLOP-wise speedup (upper bound) with 93.75% sparsity: 16X*

        Speedup: 424.766 us / 38.240 us = **11.1X**

        !!! note
            1-D non-dilated neighborhood attention does not require token permute, hence the only
            runtime is the kernel runtime.

    </div>




### 2-D use case (FLUX)

In this example we take the problem size from Flux.1-dev (4K), with 24 attention heads, and a
`(256, 256)` token layout (feature map) shape.


=== "Hopper"

    ```shell title="Baseline (cuDNN)"
    python -m natten.profiler \
        --backend cudnn \
        -n 24 \
        -d 128 \
        -i 256 256
    ```

    <div class="result" markdown>

    ??? hopper-example "Sample output from running on H100"
        ```
        --8<-- "docs/sample-outputs/profiler-2d-flux-cudnn-h100.txt"
        ```

    </div>

=== "Blackwell"

    ```shell title="Baseline (cuDNN)"
    python -m natten.profiler \
        --backend cudnn \
        -n 24 \
        -d 128 \
        -i 256 256
    ```

    <div class="result" markdown>

    ??? blackwell-example "Sample output from running on B200"
        ```
        --8<-- "docs/sample-outputs/profiler-2d-flux-cudnn-b200.txt"
        ```

    </div>




Now let's try standard neighborhood attention, with a window size of `(80, 80)`, which is
approximately 90% sparsity (the same configuration as in the
[GNA paper](https://arxiv.org/abs/2504.16922)).

=== "Hopper"

    We explicitly set the query and KV tile shapes to `(16, 8)`, which is a combination that
    requires not additional input padding, and given the window size can become fully block-sparse
    when using stride (next up). Q and KV tile shapes `(16, 8)` is a combination supported by our
    Hopper FNA kernel.

    ```shell title="Neighborhood attention (Hopper FNA)"
    python -m natten.profiler \
        --backend hopper-fna \
        --q-tile 16 8 \
        --kv-tile 16 8 \
        -n 24 \
        -d 128 \
        -i 256 256 \
        -w 80 80
    ```

    <div class="result" markdown>

    ??? hopper-example "Sample output from running on H100"
        ```
        --8<-- "docs/sample-outputs/profiler-2d-flux-na-hopper-fna-h100.txt"
        ```

        *FLOP-wise speedup (upper bound) with ~90% sparsity: 10.2X*

        Speedup w/o token permute time: 97.435 ms / 16.201 ms = **6.0X**

        Speedup w/ token permute time: 97.435 ms / 18.603 ms = **5.2X**

    </div>

=== "Blackwell"

    We explicitly set the query and KV tile shapes to `(16, 16)` and `(16, 8)` respectively.
    This is because the Blackwell forward pass kernel presently only supports a Q tile size of 256
    and KV tile size of 128. Query padding is not avoidable in this case, but KV padding is avoided,
    which allows the mask to become fully block-sparse when using stride (next up).

    ```shell title="Neighborhood attention (Blackwell FNA)"
    python -m natten.profiler \
        --backend blackwell-fna \
        --q-tile 16 16 \
        --kv-tile 16 8 \
        -n 24 \
        -d 128 \
        -i 256 256 \
        -w 80 80
    ```

    <div class="result" markdown>

    ??? blackwell-example "Sample output from running on B200"
        ```
        --8<-- "docs/sample-outputs/profiler-2d-flux-na-blackwell-fna-b200.txt"
        ```

        *FLOP-wise speedup (upper bound) with ~90% sparsity: 10.2X*

        Speedup w/o token permute time: 43.313 ms / 8.692 ms = **5.0X**

        Speedup w/ token permute time: 43.313 ms / 10.931 ms = **4.0X**

    </div>




Finally let's try strided neighborhood attention, with stride `(16, 16)` (the same configuration
as in the [GNA paper](https://arxiv.org/abs/2504.16922)).
Given the use case, and Q and KV tile shapes, this stride results in a fully block-sparse mask.



=== "Hopper"

    ```shell title="Neighborhood attention (Hopper FNA) w/ stride"
    python -m natten.profiler \
        --backend hopper-fna \
        --q-tile 16 8 \
        --kv-tile 16 8 \
        -n 24 \
        -d 128 \
        -i 256 256 \
        -w 80 80 \
        -s 16 16
    ```

    <div class="result" markdown>

    ??? hopper-example "Sample output from running on H100"
        ```
        --8<-- "docs/sample-outputs/profiler-2d-flux-gna-hopper-fna-h100.txt"

        ```

        *FLOP-wise speedup (upper bound) with ~90% sparsity: 10.2X*

        Speedup w/o token permute time: 97.435 ms / 7.914 ms = **12.3X**

        Speedup w/ token permute time: 97.435 ms / 10.601 ms = **9.2X**

        !!! note
            The baseline is cuDNN Attention, which in this particlar case is slightly slower than
            CUTLASS FMHA, which is why the speedup without token permute exceeds the 10.2X FLOP-wise
            limit. Additionally, runtime variance and other factors may affect observable speedups.

    </div>

=== "Blackwell"

    ```shell title="Neighborhood attention (Blackwell FNA) w/ stride"
    python -m natten.profiler \
        --backend blackwell-fna \
        --q-tile 16 16 \
        --kv-tile 16 8 \
        -n 24 \
        -d 128 \
        -i 256 256 \
        -w 80 80 \
        -s 16 16
    ```

    <div class="result" markdown>

    ??? blackwell-example "Sample output from running on B200"
        ```
        --8<-- "docs/sample-outputs/profiler-2d-flux-gna-blackwell-fna-b200.txt"

        ```

        *FLOP-wise speedup (upper bound) with ~90% sparsity: 10.2X*

        Speedup w/o token permute time: 43.313 ms / 4.114 ms = **10.5X**

        Speedup w/ token permute time: 43.313 ms / 6.389 ms = **6.8X**

        !!! note
            Speedup without token permute only appears to exceed the 10.2X FLOP-wise due to runtime
            variance and other factors that may affect observable speedups.


    </div>




### 3-D use case (Hunyuan Video)

In this example we take the problem size from Hunyuan Video, with 24 attention heads, and a
`(30, 48, 80)` token layout (feature map) shape.


=== "Hopper"

    ```shell title="Baseline (cuDNN)"
    python -m natten.profiler \
        --backend cudnn \
        -n 24 \
        -d 128 \
        -i 30 48 80
    ```

    <div class="result" markdown>

    ??? hopper-example "Sample output from running on H100"
        ```
        --8<-- "docs/sample-outputs/profiler-3d-hunyuan-cudnn-h100.txt"
        ```

    </div>

=== "Blackwell"

    ```shell title="Baseline (cuDNN)"
    python -m natten.profiler \
        --backend cudnn \
        -n 24 \
        -d 128 \
        -i 30 48 80
    ```

    <div class="result" markdown>

    ??? blackwell-example "Sample output from running on B200"
        ```
        --8<-- "docs/sample-outputs/profiler-3d-hunyuan-cudnn-b200.txt"
        ```

    </div>




Now let's try standard neighborhood attention, with a window size of `(18, 24, 24)`, which is
approximately 90% sparsity (the same configuration as in the
[GNA paper](https://arxiv.org/abs/2504.16922)).

=== "Hopper"

    We explicitly set the query and KV tile shapes to `(2, 8, 8)`, which is a combination that
    requires not additional input padding, and given the window size can become fully block-sparse
    when using stride (next up). Q and KV tile shapes `(2, 8, 8)` is a combination supported by our
    Hopper FNA kernel.

    ```shell title="Neighborhood attention (Hopper FNA)"
    python -m natten.profiler \
        --backend hopper-fna \
        --q-tile 2 8 8 \
        --kv-tile 2 8 8 \
        -n 24 \
        -d 128 \
        -i 30 48 80 \
        -w 18 24 24
    ```

    <div class="result" markdown>

    ??? hopper-example "Sample output from running on H100"
        ```
        --8<-- "docs/sample-outputs/profiler-3d-hunyuan-na-hopper-fna-h100.txt"
        ```

        *FLOP-wise speedup (upper bound) with ~91% sparsity: 11.1X*

        Speedup w/o token permute time: 283.327 ms / 79.043 ms = **3.6X**

        Speedup w/ token permute time: 283.327 ms / 84.417 ms = **3.4X**

    </div>

=== "Blackwell"

    We explicitly set the query and KV tile shapes to `(4, 8, 8)` and `(2, 8, 8)` respectively.
    This is because the Blackwell forward pass kernel presently only supports a Q tile size of 256
    and KV tile size of 128. Query padding is not avoidable in this case, but KV padding is avoided,
    which allows the mask to become fully block-sparse when using stride (next up).

    ```shell title="Neighborhood attention (Blackwell FNA)"
    python -m natten.profiler \
        --backend blackwell-fna \
        --q-tile 4 8 8 \
        --kv-tile 2 8 8 \
        -n 24 \
        -d 128 \
        -i 30 48 80 \
        -w 18 24 24
    ```

    <div class="result" markdown>

    ??? blackwell-example "Sample output from running on B200"
        ```
        --8<-- "docs/sample-outputs/profiler-3d-hunyuan-na-blackwell-fna-b200.txt"
        ```

        *FLOP-wise speedup (upper bound) with ~91% sparsity: 11.1X*

        Speedup w/o token permute time: 135.265 ms / 42.243 ms = **3.2X**

        Speedup w/ token permute time: 135.265 ms / 47.353 ms = **2.9X**

    </div>




Finally let's try strided neighborhood attention, with stride `(16, 8, 8)` (the same configuration
as in the [GNA paper](https://arxiv.org/abs/2504.16922)).
Given the use case, and Q and KV tile shapes, this stride results in a fully block-sparse mask.



=== "Hopper"

    ```shell title="Neighborhood attention (Hopper FNA) w/ stride"
    python -m natten.profiler \
        --backend hopper-fna \
        --q-tile 2 8 8 \
        --kv-tile 2 8 8 \
        -n 24 \
        -d 128 \
        -i 30 48 80 \
        -w 18 24 24 \
        -s 16 8 8
    ```

    <div class="result" markdown>

    ??? hopper-example "Sample output from running on H100"
        ```
        --8<-- "docs/sample-outputs/profiler-3d-hunyuan-gna-hopper-fna-h100.txt"

        ```

        *FLOP-wise speedup (upper bound) with ~91% sparsity: 11.1X*

        Speedup w/o token permute time: 283.327 ms / 23.359 ms = **12.1X**

        Speedup w/ token permute time: 283.327 ms / 29.109 ms = **9.7X**

        !!! note
            The baseline is cuDNN Attention, which in this particlar case is slightly slower than
            CUTLASS FMHA, which is why the speedup without token permute exceeds the 11.1X FLOP-wise
            limit. Additionally, runtime variance and other factors may affect observable speedups.

    </div>

=== "Blackwell"

    ```shell title="Neighborhood attention (Blackwell FNA) w/ stride"
    python -m natten.profiler \
        --backend blackwell-fna \
        --q-tile 4 8 8 \
        --kv-tile 2 8 8 \
        -n 24 \
        -d 128 \
        -i 30 48 80 \
        -w 18 24 24 \
        -s 16 8 8
    ```

    <div class="result" markdown>

    ??? blackwell-example "Sample output from running on B200"
        ```
        --8<-- "docs/sample-outputs/profiler-3d-hunyuan-gna-blackwell-fna-b200.txt"

        ```

        *FLOP-wise speedup (upper bound) with ~91% sparsity: 11.1X*

        Speedup w/o token permute time: 135.265 ms / 13.010 ms = **10.4X**

        Speedup w/ token permute time: 135.265 ms / 18.538 ms = **7.3X**


    </div>




## Limitations

We will be expanding the current profiling toolkit to support third-party backends (i.e. Flash
Attention 3), more complete argument bindings, automatic comparison to reference with speedup
measurements, and more.
