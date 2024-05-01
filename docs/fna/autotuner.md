## NATTEN Auto-tuner

In many cases, there usually exist more than just one compatible kernel configuration given a specific problem and device.
This is true with Fused Neighborhood Attention, specifically in 2-D and 3-D, where tiling shape, and other such settings
can affect the overall latency while not affecting the outcome except for minor differences due to numerical precision
and lack of associativity in floating point accumulation.

Because of this, instead of always picking the "default" configuration for each use case, we're trying
to offer solutions that pick more suitable configurations.
Our new Auto-tuner is a naive approach of doing just that.

When enabled, auto-tuner benchmarks every new problem (identified by problem size, data type, and other parameters) 
once and caches its optimal kernel configuration, which is reused throughout the lifetime of your program.
This means that when auto-tuner is activated, your first forward pass is expected to take longer than typical, 
but iterations that follow will likely be more performant than without auto-tuning.

This feature is still in early stages and relatively untested, and is not part of libnatten, and is therefore experimental and 
subject to change. Bug reports related to it and FNA in general are strongly appreciated.

### When should I NOT use auto-tuner?

1. If you don't have the patience to sit and wait for auto-tuner every time.
  * Auto-tuner is at a very early stage. It is not fast, it is not optimal, and it might not
    improve performance as much as you'd expect.
    
2. If you're using graph mode / torch.compile
  * NATTEN does not yet support torch.compile, but will in the future, and when that happens,
    auto-tuner will likely not be supported immediately.

3. If you expect varying input shapes
  * Auto-tuner works best only when your input shapes rarely change, because it will try to
    benchmark and tune every single new input shape. This means every time FNA ops are called
    with a new input shape, that forward pass will be relatively slower, and follow up calls are
    the only thing that is improved.
  * Frequently changing input shapes might hurt your performance more than it could help.

4. If you use [PyTorch with deterministic algorithms](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
  * Auto-tuner is not deterministic. It is not guaranteed to pick the same configuration every time, and
    different kernel configurations will have near but non zero differences. This holds true for any generic
    kernel with different tiling configurations.
  * Using auto-tuner with PyTorch's deterministic flag on will trigger failure from NATTEN.

#### If you're doing distributed training, please be extra cautious
Auto-tuner context is limited to one process only, and there is no infrastructure for communication.
This means different devices (GPUs, nodes) will run auto-tuner on their own, and may pick different
configurations. There's plenty that could go wrong here, because the right approach would be to distribute
tuning configurations across devices and then broadcast the results back, and possibly cache to disk.
If you have any ideas on handling this, please open an issue.

### When should I use auto-tuner?
We mostly recommend using auto-tuner for running inference, unless you're very patient and would
like to experiment with this feature, in which case, we would appreciate any feedback regarding your experience.

If you find that the auto-tuner's stuck, please refer to [log levels](#log-levels) and set the logger level
to debug, and NATTEN will start logging auto-tuner reports.

### How do I use auto-tuner?

It's simple, just call `natten.use_autotuner`:

```python
import natten

natten.use_fused_na(True)
# FNA runs according to the default configuration for
# every problem size.

natten.use_autotuner(forward_pass=True, backward_pass=False)
# FNA optimizes the configuration according
# to problem size. Enable for forward pass,
# disable for backward pass.

natten.use_autotuner(forward_pass=True, backward_pass=True)
# FNA optimizes the configuration according
# to problem size. Enable for forward pass,
# and backward pass.

natten.disable_autotuner()
# Disable auto-tuner
# Or alternatively:
# natten.use_autotuner(False, False)
```

If you're training, we recommend also enabling KV paralllelism:

```python
import natten

natten.use_kv_parallelism_in_fused_na(True)
```

For more information, please refer to [KV parallelism](kv-parallelism.md).

### Auto-tuner settings

As mentioned above, auto-tuner can be toggled for forward and backward pass separately:

```python
natten.use_autotuner(forward_pass=True, backward_pass=False)
```

Backward pass generally takes more time and there are usually more configurations available.
Because of this, and since auto-tuner only performs a search with no heuristics over configs,
we define a "thorough" mode for auto-tuner in which all possible configurations are benchmarked.
This means considerably longer auto-tuning time, but at the same time higher chance of finding the "best" config.

```python
natten.use_autotuner(
  backward_pass=True,
  thorough_mode_backward=True,
)
```

In addition, the number of warmup steps and benchmark steps eventually determine how well auto-tuner performs in terms of
finding the fastest configuration.
You can change the number of warmup and benchmark steps if you wish, but we recommend setting them to a minimum of 5.

```python
natten.use_autotuner(
  forward_pass=True,
  backward_pass=True,
  warmup_steps_forward=10,
  warmup_steps_backward=20,
  steps_forward=5,
  steps_backward=10,
)
```

### Log levels

Auto-tuner logs details on each problem that it benchmarks if the log level is set to `DEBUG`.
NATTEN logger sets its log level based on the environment variable `NATTEN_LOG_LEVEL`, and if unset defaults to
log level `INFO`.

If your application is taking a long time running auto-tuner, or you'd just like to be informed,
you can run your program with log level `DEBUG`:

```bash
export NATTEN_LOG_LEVEL="debug"

python3 test.py

# Or alternatively, if you don't want to set the env variable
# in your current shell session

NATTEN_LOG_LEVEL="debug" python3 test.py

```

This is highly recommended if you use auto-tuner, particularly in thorough mode or when it's enabled for the backward pass.

