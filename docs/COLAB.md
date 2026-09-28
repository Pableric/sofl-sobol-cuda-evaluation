# Tesla T4 Colab run

Select **Runtime → Change runtime type → T4 GPU**. This release is an
`sm_75` binary; other Colab GPUs are not supported by this file.

Cell 1 checks the actual compiler and GPU. The driver version shown by
`nvidia-smi` is not the NVCC version.

```python
import subprocess
subprocess.run(["nvidia-smi"], check=True)
subprocess.run(["nvcc", "--version"], check=True)
```

cuRANDDx 0.2.4 requires NVCC 13.0 or newer. If Cell 1 shows CUDA 12.x,
run Cell 2. It installs only the CUDA 13.0 toolkit in the temporary Colab
runtime. It also handles the exact duplicate unsigned NVIDIA repository
entry that some Colab images contain; it refuses other unexpected APT
source configurations instead of silently changing them.

```python
import pathlib
import subprocess

os_release = pathlib.Path("/etc/os-release").read_text()
if 'ID=ubuntu' not in os_release or 'VERSION_ID="24.04"' not in os_release:
    raise RuntimeError("This installation cell expects Ubuntu 24.04")

signed = pathlib.Path("/etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list")
unsigned = pathlib.Path("/etc/apt/sources.list.d/cuda.list")
expected_signed = (
    "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] "
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
)
expected_unsigned = (
    "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /"
)
if unsigned.exists():
    if not signed.exists() or signed.read_text().strip() != expected_signed:
        raise RuntimeError("Unexpected CUDA APT sources; inspect them manually")
    if unsigned.read_text().strip() != expected_unsigned:
        raise RuntimeError("Unexpected cuda.list; inspect it manually")
    unsigned.rename(unsigned.with_suffix(".list.disabled"))

if not signed.exists():
    keyring = pathlib.Path("/tmp/sofl-cuda-keyring.deb")
    subprocess.run([
        "curl", "-fL",
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb",
        "-o", str(keyring),
    ], check=True)
    subprocess.run(["dpkg", "-i", str(keyring)], check=True)

subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "cuda-toolkit-13-0"], check=True)
subprocess.run(["/usr/local/cuda-13.0/bin/nvcc", "--version"], check=True)
```

Cell 3 clones this public repository and runs the full correctness and
paired benchmark. It needs no secret or token. The PATH selects CUDA 13
when Colab has multiple toolkit versions installed.

```python
import os
import pathlib
import subprocess

repo = pathlib.Path("/content/sofl-sobol-cuda-evaluation")
if repo.exists():
    raise RuntimeError("Repository path already exists; use a fresh runtime")
subprocess.run([
    "git", "clone", "https://github.com/Pableric/sofl-sobol-cuda-evaluation.git",
    str(repo),
], check=True)
print("PUBLIC COMMIT:", subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
environment = os.environ.copy()
environment["PATH"] = "/usr/local/cuda-13.0/bin:" + environment["PATH"]
subprocess.run(["./scripts/run_t4.sh"], cwd=repo, env=environment, check=True)
```

Cell 4 shows the summary and downloads the compact hashed archive.

```python
import json
import pathlib
from google.colab import files

repo = pathlib.Path("/content/sofl-sobol-cuda-evaluation")
results = sorted(repo.glob("results/t4_*"), key=lambda p: p.stat().st_mtime)
if not results:
    raise RuntimeError("No completed T4 result directory exists")
result = results[-1]
print((result / "status.txt").read_text())
print(json.dumps(json.loads((result / "summary.json").read_text()), indent=2))
archive = pathlib.Path("/content") / (result.name + ".tar.gz")
if not archive.exists():
    raise RuntimeError("Result archive is missing")
print((pathlib.Path(str(archive) + ".sha256")).read_text())
files.download(str(archive))
files.download(str(archive) + ".sha256")
```

These are hardware measurements, not a claim that the closed binary is
portable to other GPUs. The repository does not contain SofL CUDA source.
