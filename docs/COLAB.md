# Tesla T4 Colab run

Select **Runtime → Change runtime type → T4 GPU**. This release is an
`sm_75` binary; other Colab GPUs are not supported by this file.

Run these two cells in order. The first cell prints each stage, installs
CUDA Toolkit 13.0 only when an NVCC 13 compiler is unavailable, checks out
the public repository, and runs full correctness and paired benchmarks.
The NVIDIA driver version shown by `nvidia-smi` is **not** the NVCC version.
No GitHub token, Docker, or Podman is needed.

## Cell 1 — setup and run

```bash
%%bash
set -euo pipefail

echo '=== GPU ==='
gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
gpu_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1p')
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')
printf 'Model: %s\nCompute capability: %s\nDriver: %s\n' "$gpu_model" "$gpu_cap" "$driver"
[[ $gpu_model == 'Tesla T4' && $gpu_cap == '7.5' ]] || {
    echo 'This release requires a Tesla T4 (sm_75).' >&2
    exit 1
}

echo '=== CUDA compiler ==='
nvcc13=''
if command -v nvcc >/dev/null && nvcc --version | grep -q 'release 13\.'; then
    nvcc13=$(command -v nvcc)
elif [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
    nvcc13=/usr/local/cuda-13.0/bin/nvcc
fi

if [[ -z $nvcc13 ]]; then
    echo 'NVCC 13 not found; installing CUDA Toolkit 13.0 in this temporary runtime.'
    [[ $(id -u) -eq 0 ]] || { echo 'Colab root access is required.' >&2; exit 1; }
    . /etc/os-release
    [[ $ID == ubuntu && $VERSION_ID == 24.04 ]] || {
        echo 'This installation step expects Ubuntu 24.04.' >&2
        exit 1
    }

    signed=/etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
    unsigned=/etc/apt/sources.list.d/cuda.list
    expected_signed='deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /'
    expected_unsigned='deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /'
    if [[ -f $unsigned ]]; then
        [[ -f $signed && $(<"$signed") == "$expected_signed" && $(<"$unsigned") == "$expected_unsigned" ]] || {
            echo 'Unexpected CUDA APT sources; inspect them before changing anything.' >&2
            exit 1
        }
        mv -- "$unsigned" "$unsigned.disabled"
    fi
    if [[ ! -f $signed ]]; then
        curl -fL \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
            -o /tmp/sofl-cuda-keyring.deb
        dpkg -i /tmp/sofl-cuda-keyring.deb
    fi
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-13-0
    nvcc13=/usr/local/cuda-13.0/bin/nvcc
fi
[[ -x $nvcc13 ]] && "$nvcc13" --version | grep -q 'release 13\.' || {
    echo 'NVCC 13 verification failed.' >&2
    exit 1
}
export PATH="$(dirname "$nvcc13"):$PATH"
nvcc --version

echo '=== Public repository ==='
repo=/content/sofl-sobol-cuda-evaluation
url=https://github.com/Pableric/sofl-sobol-cuda-evaluation.git
if [[ -e $repo ]]; then
    [[ -d $repo/.git && $(git -C "$repo" remote get-url origin) == "$url" ]] || {
        echo 'Existing repository path is not the expected public checkout.' >&2
        exit 1
    }
    [[ -z $(git -C "$repo" status --porcelain) ]] || {
        echo 'Existing checkout has local changes; refusing to overwrite them.' >&2
        exit 1
    }
    git -C "$repo" fetch origin main
    git -C "$repo" checkout main
    git -C "$repo" merge --ff-only origin/main
else
    git clone "$url" "$repo"
fi
printf 'Public commit: %s\n' "$(git -C "$repo" rev-parse HEAD)"

echo '=== Correctness and paired benchmark ==='
cd "$repo"
./scripts/run_t4.sh
```

## Cell 2 — inspect and download

Run this after Cell 1 succeeds. A browser download can be retried without
rerunning the benchmark.

```python
import json
import pathlib
from google.colab import files

repo = pathlib.Path("/content/sofl-sobol-cuda-evaluation")
results = sorted(
    (path for path in repo.glob("results/t4_*")
     if (path / "status.txt").exists() and (path / "summary.json").exists()),
    key=lambda path: path.stat().st_mtime,
)
if not results:
    raise RuntimeError("No completed T4 result directory exists")
result = results[-1]
print("RESULT:", result)
print((result / "status.txt").read_text())
print(json.dumps(json.loads((result / "summary.json").read_text()), indent=2))

archive = pathlib.Path("/content") / (result.name + ".tar.gz")
checksum = pathlib.Path(str(archive) + ".sha256")
if not archive.exists() or not checksum.exists():
    raise RuntimeError("Result archive or its checksum is missing")
print(checksum.read_text())
files.download(str(archive))
files.download(str(checksum))
```

These are hardware measurements, not a claim that the closed binary is
portable to other GPUs. The repository does not contain SofL CUDA source.
