# SofL Sobol CUDA evaluation

This repository lets you verify and measure a closed SofL Sobol generator
against NVIDIA cuRANDDx Sobol32 on a **Tesla T4 (sm_75)**. The library emits
ordinary, unscrambled Joe–Kuo 6 Sobol points, including point zero. Outputs
are dimension-major. It provides raw 32-bit words, float32 uniforms in
`[0,1)`, and a fixed fused consumer that writes only one checksum per
256 values.

The public benchmark source calls the distributed
`lib/libsofl_sobol_sm75.so` and uses cuRANDDx 0.2.4 as its comparator.
Both fused routes perform the same normalization and consumer:

```cpp
float u = bit_cast<float>((raw >> 9) | 0x3f800000) - 1.0f;
float x = fma(0.3f, u, 0.7f);
checksum += bit_cast<uint32_t>(x * x);  // modular uint32 sum
```

The primary workload generates **65,536 points in each of 32 dimensions**.
A second workload generates **60 million points in D1**. Both routes write
only the chunk checksums during the fused timing. The benchmark measures
cuRANDDx with state constructed inside the timed kernel (fresh) and with
reusable state snapshots prepared before timing (prepared). SofL requires no
external state preparation. State preparation is also reported separately.

## Run

Requirements: NVIDIA Tesla T4, CUDA Toolkit **13.0**, Python 3, a C++20
capable NVCC and common command-line tools. A driver displaying “CUDA 13”
in `nvidia-smi` does not establish the NVCC version; check
`nvcc --version`.

```bash
git clone https://github.com/Pableric/sofl-sobol-cuda-evaluation.git
cd sofl-sobol-cuda-evaluation
./scripts/run_t4.sh
```

For a T4 Colab runtime, use the copy-and-paste cells in
[the Colab guide](docs/COLAB.md). No GitHub token is needed for this public
repository.

The runner downloads the pinned NVIDIA MathDx package after verifying its
SHA-256, compiles the public comparator, checks every raw and float output
word against a CPU recurrence, checks both fused routes against the frozen
checksum, then takes five sessions of 100 randomized paired CUDA-event
measurements with 20 warmups. The materialized outputs are used for
correctness only. Allocation, transfers, validation and state preparation
are excluded from fused timing. Each result goes to a new directory and a
compact hashed archive.

For a quick integration check, set `SOFL_SESSIONS=1
SOFL_REPETITIONS=10 SOFL_WARMUP=5` before the command. That smaller run
is not a performance result.

## Status and scope

The [binary audit](evidence/binary_audit.json) reports a 72,008-byte
stripped `sm_75` library with three public ABI symbols, nine SASS kernels,
no embedded PTX, no debug sections and no spills. The
[validation log](evidence/validation.log) records exact output hashes,
guard checks and deterministic reruns on the Tesla T4. The binary SHA-256
is `fc01e128ee013875e11bc5d52c5d9b163ff51b2e73075045106a7c0b485d8f60`.

The earlier cuRANDDx timings came from a private benchmark executable.
This repository's runner will produce the public comparison for the
distributed `.so`; until that run is completed, no speedup is claimed
for this binary.

Supported library point counts are 8,192, 65,536 and 60,000,000, with a
one-based contiguous dimension range within D1–D64. The public benchmark
covers the two workloads above. The `sm_75` binary is specific to T4; a
different GPU architecture requires a separately built library.

The binary is offered under [evaluation terms](EVALUATION_TERMS.md).
The cuRANDDx archive is not distributed by this repository.
