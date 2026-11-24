# RelayVSR SM75 NATTEN wheel

This branch is a deliberately narrow NATTEN 0.21.6 build for the validated RelayVSR T4 path.

## Fixed build profile

- Upstream source: `SHI-Labs/NATTEN` commit `4472b95f46f55a8ba2c5f9869eba252bbf5a22f5`
- Local package version: `0.21.6+relayvsr1.sm75.torch2110cu130`
- Python: CPython 3.13 (cp313)
- PyTorch: 2.11.0, CUDA runtime 13.0
- CUDA toolkit used by CI: 13.0.2
- Target architecture: NVIDIA SM75 (T4)
- Kernel scope: FNA, 2D, FP16, non-causal, forward/backward GEMM tile 64x64x64
- RelayVSR window semantics: query-centred clipped windows for stride 1

The generator is specialized to this exact workload to avoid generating unrelated architectures, dtypes, dimensions, causal variants, and tile shapes. Do not use this wheel as a general NATTEN distribution.

## Build

The GitHub Actions workflow `.github/workflows/build-relayvsr-sm75.yml` compiles the CUDA extension on a CPU-only hosted runner. A GPU is not needed to compile CUDA kernels; NVCC emits SM75 device code. The workflow checks the produced wheel metadata and imports its compiled `libnatten` extension.

Successful workflow runs publish a prerelease containing the wheel and a SHA-256 manifest. The release is a build artifact, not proof of runtime correctness on a physical T4.

## Required runtime validation

Before enabling this wheel in a production RelayVSR run, install it in the exact Kaggle environment and execute the existing RelayVSR NATTEN geometry/numeric tests on a T4. Confirm that only the three prequalified geometry signatures dispatch to NATTEN and all other cases continue to fall back to the original SDPA path. A successful GitHub build/import check does not replace that GPU runtime test.
