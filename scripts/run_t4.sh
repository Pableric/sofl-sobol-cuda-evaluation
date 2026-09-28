#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$root"
for command in curl git nvidia-smi nvcc python3 sha256sum tar; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 1
    }
done
[[ $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader |
    sed -n '1p') == 7.5 ]] || {
    echo "this release binary requires sm_75 (Tesla T4)" >&2
    exit 1
}
nvcc --version | grep -q 'release 13\.' || {
    echo "cuRANDDx 0.2.4 requires a CUDA 13 compiler" >&2
    exit 1
}

sha256sum -c SHA256SUMS
python3 scripts/verify_public_package.py

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
result="$root/results/t4_$timestamp"
[[ ! -e "$result" ]] || { echo "result directory already exists" >&2; exit 1; }
mkdir -p "$result"/{machine,logs,timing}
work=$(mktemp -d "${TMPDIR:-/tmp}/sofl-public-t4.XXXXXX")
cleanup() { rm -rf -- "$work"; }
trap cleanup EXIT

echo "[1/5] Checking NVIDIA dependency"
archive="$work/nvidia-mathdx-26.06.1-cuda13.tar.gz"
if [[ -n ${MATHDX_ARCHIVE:-} ]]; then
    cp "$MATHDX_ARCHIVE" "$archive"
else
    curl -fL \
        https://developer.nvidia.com/downloads/compute/cuRANDDx/redist/cuRANDDx/cuda13/nvidia-mathdx-26.06.1-cuda13.tar.gz \
        -o "$archive"
fi
printf '%s  %s\n' \
    59a9233db34b75568acbcc5284e6cefe6fad5577ee644f85044971d62eeea353 \
    "$archive" | sha256sum -c -
tar -xzf "$archive" -C "$work"
include="$work/nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06/include"
[[ -f "$include/curanddx.hpp" ]] || exit 1

echo "[2/5] Building the public comparator"
nvcc -std=c++20 -O3 -arch=sm_75 -Xptxas=-v \
    -Iinclude -I"$include" bench/compare.cu \
    -Llib -l:libsofl_sobol_sm75.so -lcurand \
    -o "$work/compare" 2> "$result/logs/compiler.log"
nvidia-smi -q > "$result/machine/nvidia_smi_q.txt"
nvcc --version > "$result/machine/nvcc_version.txt"
cp evidence/binary_audit.json "$result/binary_audit.json"
cp evidence/BUILD_INFO.json "$result/binary_build_info.json"
cp vendor/mathdx.json "$result/mathdx.json"
sha256sum lib/libsofl_sobol_sm75.so > "$result/library_sha256.txt"
git rev-parse HEAD > "$result/public_commit.txt"

export LD_LIBRARY_PATH="$root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "[3/5] Verifying all raw, float and fused output"
"$work/compare" check > "$result/logs/correctness.log"

sessions=${SOFL_SESSIONS:-5}
repetitions=${SOFL_REPETITIONS:-100}
warmup=${SOFL_WARMUP:-20}
[[ $sessions =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $repetitions =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $warmup =~ ^[0-9]+$ ]] || exit 2

echo "[4/5] Measuring paired fresh and prepared contracts"
for workload in "65536 32" "60000000 1"; do
    read -r points dimensions <<< "$workload"
    for ((session = 0; session < sessions; ++session)); do
        "$work/compare" bench "$points" "$dimensions" "$session" \
            "$warmup" "$repetitions" \
            > "$result/timing/n${points}_d${dimensions}.session${session}.jsonl"
    done
done
python3 scripts/analyze.py "$result"/timing/*.jsonl \
    --output "$result/summary.json" > "$result/logs/summary.txt"

echo "[5/5] Freezing results"
printf '%s\n' \
    "PUBLIC_T4_BINARY_CORRECTNESS=PASS" \
    "CURANDDX_FRESH=MEASURED" \
    "CURANDDX_PREPARED=MEASURED" \
    "CUDA_ARCHITECTURE=sm_75" \
    > "$result/status.txt"
python3 scripts/write_result_hashes.py "$result"
archive_root=${SOFL_ARCHIVE_DIR:-/content}
[[ -d "$archive_root" ]] || archive_root=/tmp
output="$archive_root/$(basename "$result").tar.gz"
[[ ! -e "$output" && ! -e "$output.sha256" ]] || exit 1
tar -czf "$output" -C "$(dirname "$result")" "$(basename "$result")"
sha256sum "$output" > "$output.sha256"
cat "$result/logs/summary.txt"
echo "RESULT_ARCHIVE=$output"
cat "$output.sha256"
