// Public cuRANDDx comparator for the closed SofL sm_75 library.
#include "sofl_sobol.h"

#include <curand.h>
#include <curanddx.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr unsigned kChunk = 256;
constexpr unsigned kThreads = 64;
constexpr unsigned kGuardWords = 64;
constexpr std::uint32_t kGuard = 0xa5a5a5a5U;
using Rng = decltype(curanddx::Generator<curanddx::sobol32>() +
                     curanddx::SM<750>() + curanddx::Thread());
using Direction = typename Rng::direction_vector_type;

void ck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

void ck_curand(curandStatus_t status, const char* operation) {
    if (status != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cuRAND status %d\n", operation,
                     static_cast<int>(status));
        std::exit(1);
    }
}

__device__ __forceinline__ std::uint32_t consume(std::uint32_t raw) {
    const float u = __fsub_rn(
        __uint_as_float((raw >> 9U) | 0x3f800000U), 1.0f);
    const float x = __fmaf_rn(0.3f, u, 0.7f);
    return __float_as_uint(__fmul_rn(x, x));
}

__device__ __forceinline__ std::uint32_t consume_chunk(Rng& rng) {
    std::uint32_t a = 0, b = 0, c = 0, d = 0;
#pragma unroll 1
    for (unsigned i = 0; i < kChunk / 4U; ++i) {
        a += consume(rng.generate());
        b += consume(rng.generate());
        c += consume(rng.generate());
        d += consume(rng.generate());
    }
    return (a + b) + (c + d);
}

template <unsigned N, unsigned D>
__global__ __launch_bounds__(kThreads) void fresh(
    Direction* directions, std::uint32_t* output) {
    CURANDDX_SKIP_IF_NOT_APPLICABLE_SM(Rng);
    constexpr unsigned chunks = N / kChunk;
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= D * chunks) return;
    const unsigned dimension = index / chunks;
    const unsigned chunk = index - dimension * chunks;
    Rng rng(dimension, directions, chunk * kChunk);
    output[index] = consume_chunk(rng);
}

template <unsigned N, unsigned D>
__global__ __launch_bounds__(kThreads) void prepare(
    Direction* directions, Rng* states) {
    CURANDDX_SKIP_IF_NOT_APPLICABLE_SM(Rng);
    constexpr unsigned chunks = N / kChunk;
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= D * chunks) return;
    const unsigned dimension = index / chunks;
    const unsigned chunk = index - dimension * chunks;
    states[index] = Rng(dimension, directions, chunk * kChunk);
}

template <unsigned N, unsigned D>
__global__ __launch_bounds__(kThreads) void prepared(
    const Rng* states, std::uint32_t* output) {
    CURANDDX_SKIP_IF_NOT_APPLICABLE_SM(Rng);
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= D * (N / kChunk)) return;
    Rng rng = states[index];
    output[index] = consume_chunk(rng);
}

template <unsigned N, unsigned D, bool FloatOutput>
__global__ void materialized(Direction* directions, void* output) {
    CURANDDX_SKIP_IF_NOT_APPLICABLE_SM(Rng);
    constexpr std::size_t words = static_cast<std::size_t>(N) * D;
    const std::size_t stride = blockDim.x * static_cast<std::size_t>(gridDim.x);
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < words; index += stride) {
        const unsigned dimension = static_cast<unsigned>(index / N);
        const unsigned point = static_cast<unsigned>(index % N);
        Rng rng(dimension, directions, point);
        const std::uint32_t raw = rng.generate();
        if constexpr (FloatOutput) {
            const float bits = __uint_as_float((raw >> 9U) | 0x3f800000U);
            static_cast<float*>(output)[index] = __fsub_rn(bits, 1.0f);
        } else {
            static_cast<std::uint32_t*>(output)[index] = raw;
        }
    }
}

template <unsigned N, unsigned D>
constexpr unsigned blocks() {
    return (D * (N / kChunk) + kThreads - 1U) / kThreads;
}

enum class Route { Sofl, Fresh, Prepared };
const char* route_name(Route route) {
    switch (route) {
        case Route::Sofl: return "sofl";
        case Route::Fresh: return "curanddx_fresh";
        case Route::Prepared: return "curanddx_prepared";
    }
    std::abort();
}

template <unsigned N, unsigned D>
void launch(Route route, Direction* directions, const Rng* states,
            std::uint32_t* output, cudaStream_t stream) {
    if (route == Route::Sofl) {
        ck(sofl_sobol_fused_checksum_u32(output, N, 1U, D, stream),
           "launch SofL fused");
    } else if (route == Route::Fresh) {
        fresh<N, D><<<blocks<N, D>(), kThreads, 0, stream>>>(
            directions, output);
        ck(cudaPeekAtLastError(), "launch cuRANDDx fresh");
    } else {
        prepared<N, D><<<blocks<N, D>(), kThreads, 0, stream>>>(
            states, output);
        ck(cudaPeekAtLastError(), "launch cuRANDDx prepared");
    }
}

template <unsigned N, unsigned D>
void launch_prepare(Direction* directions, Rng* states, cudaStream_t stream) {
    prepare<N, D><<<blocks<N, D>(), kThreads, 0, stream>>>(directions, states);
    ck(cudaPeekAtLastError(), "launch state preparation");
}

std::uint64_t fnv(const std::vector<std::uint32_t>& words) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (std::uint32_t word : words) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            hash ^= (word >> shift) & 0xffU;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

constexpr std::uint64_t golden(unsigned N, bool floating, bool fused) {
    if (N == 65536U) {
        if (fused) return UINT64_C(0x29b2ef3ce0404b09);
        return floating ? UINT64_C(0x5c703635fe11d60d)
                        : UINT64_C(0x9591b0bd0a2c1b25);
    }
    if (fused) return UINT64_C(0xcad6868f7de32f78);
    return floating ? UINT64_C(0xab7b621983f9eafa)
                    : UINT64_C(0x63c0cf2b5565f3a5);
}

std::uint32_t host_f32(std::uint32_t raw) {
    volatile float one = 1.0f;
    return std::bit_cast<std::uint32_t>(
        std::bit_cast<float>((raw >> 9U) | 0x3f800000U) - one);
}

template <class Launch>
std::vector<std::uint32_t> capture(std::size_t words, Launch launch) {
    std::uint32_t* allocation = nullptr;
    ck(cudaMalloc(&allocation, (words + 2 * kGuardWords) * 4U),
       "allocate guarded output");
    std::vector<std::uint32_t> first;
    for (unsigned repetition = 0; repetition < 2; ++repetition) {
        ck(cudaMemset(allocation, 0xa5, (words + 2 * kGuardWords) * 4U),
           "initialize guards");
        launch(allocation + kGuardWords);
        ck(cudaDeviceSynchronize(), "synchronize correctness");
        std::vector<std::uint32_t> guarded(words + 2 * kGuardWords);
        ck(cudaMemcpy(guarded.data(), allocation, guarded.size() * 4U,
                      cudaMemcpyDeviceToHost), "copy correctness output");
        for (unsigned i = 0; i < kGuardWords; ++i) {
            if (guarded[i] != kGuard ||
                guarded[words + kGuardWords + i] != kGuard) {
                std::fprintf(stderr, "guard changed\n");
                std::exit(1);
            }
        }
        std::vector<std::uint32_t> current(
            guarded.begin() + kGuardWords,
            guarded.begin() + kGuardWords + words);
        if (repetition && current != first) {
            std::fprintf(stderr, "nondeterministic output\n");
            std::exit(1);
        }
        if (!repetition) first = std::move(current);
    }
    cudaFree(allocation);
    return first;
}

template <unsigned N, unsigned D, bool Floating>
void check_materialized(Direction* directions,
                        const curandDirectionVectors32_t* host_directions) {
    constexpr std::size_t words = static_cast<std::size_t>(N) * D;
    const auto sofl = capture(words, [&](std::uint32_t* output) {
        if constexpr (Floating) {
            ck(sofl_sobol_generate_f32(reinterpret_cast<float*>(output),
                                        N, 1U, D, nullptr), "launch SofL f32");
        } else {
            ck(sofl_sobol_generate_u32(output, N, 1U, D, nullptr),
               "launch SofL u32");
        }
    });
    const auto dx = capture(words, [&](std::uint32_t* output) {
        constexpr unsigned threads = 256;
        constexpr unsigned grid = 4096;
        materialized<N, D, Floating><<<grid, threads>>>(
            directions, output);
        ck(cudaPeekAtLastError(), "launch cuRANDDx materialized");
    });
    if (sofl != dx || fnv(sofl) != golden(N, Floating, false)) {
        std::fprintf(stderr, "materialized route mismatch N=%u D=%u f32=%d\n",
                     N, D, Floating);
        std::exit(1);
    }
    for (unsigned dimension = 0; dimension < D; ++dimension) {
        std::uint32_t state = 0;
        for (unsigned point = 0; point < N; ++point) {
            const std::size_t index =
                static_cast<std::size_t>(dimension) * N + point;
            const std::uint32_t expected = Floating ? host_f32(state) : state;
            if (sofl[index] != expected) {
                std::fprintf(stderr, "CPU oracle mismatch D=%u P=%u\n",
                             dimension + 1U, point);
                std::exit(1);
            }
            if (point + 1U < N) {
                state ^= host_directions[dimension][__builtin_ctz(point + 1U)];
            }
        }
    }
    std::printf("CORRECT_N%u_D%u_%s=PASS fnv1a64=%016llx\n", N, D,
                Floating ? "f32" : "u32",
                static_cast<unsigned long long>(fnv(sofl)));
}

template <unsigned N, unsigned D>
void check_fused(Direction* directions) {
    constexpr std::size_t words = D * (N / kChunk);
    Rng* states = nullptr;
    ck(cudaMalloc(&states, words * sizeof(Rng)), "allocate state snapshots");
    launch_prepare<N, D>(directions, states, nullptr);
    ck(cudaDeviceSynchronize(), "prepare state snapshots");
    std::vector<std::uint32_t> reference;
    for (Route route : {Route::Sofl, Route::Fresh, Route::Prepared}) {
        auto output = capture(words, [&](std::uint32_t* destination) {
            launch<N, D>(route, directions, states, destination, nullptr);
        });
        if (reference.empty()) reference = output;
        if (output != reference || fnv(output) != golden(N, false, true)) {
            std::fprintf(stderr, "fused route mismatch: %s\n",
                         route_name(route));
            std::exit(1);
        }
    }
    cudaFree(states);
    std::printf("CORRECT_N%u_D%u_FUSED=PASS fnv1a64=%016llx\n",
                N, D, static_cast<unsigned long long>(fnv(reference)));
}

template <unsigned N, unsigned D>
void correctness(Direction* directions,
                 const curandDirectionVectors32_t* host_directions) {
    check_materialized<N, D, false>(directions, host_directions);
    check_materialized<N, D, true>(directions, host_directions);
    check_fused<N, D>(directions);
}

template <class Launch>
unsigned calibrate(Launch launch, cudaEvent_t start, cudaEvent_t stop,
                   cudaStream_t stream) {
    unsigned batch = 1;
    for (;;) {
        ck(cudaEventRecord(start, stream), "calibration start");
        for (unsigned i = 0; i < batch; ++i) launch();
        ck(cudaEventRecord(stop, stream), "calibration stop");
        ck(cudaEventSynchronize(stop), "calibration synchronize");
        float milliseconds = 0;
        ck(cudaEventElapsedTime(&milliseconds, start, stop),
           "calibration elapsed");
        if (milliseconds >= 2.0f || batch >= 1U << 18U) return batch;
        batch *= 2U;
    }
}

template <unsigned N, unsigned D>
void benchmark(Direction* directions, unsigned session,
               unsigned warmup, unsigned repetitions) {
    constexpr unsigned work = D * (N / kChunk);
    std::uint32_t* output = nullptr;
    Rng* states = nullptr;
    ck(cudaMalloc(&output, work * 4U), "allocate benchmark output");
    ck(cudaMalloc(&states, work * sizeof(Rng)), "allocate benchmark states");
    ck(cudaMemset(output, 0, work * 4U), "touch benchmark output");
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
       "create stream");
    ck(cudaEventCreate(&start), "create start event");
    ck(cudaEventCreate(&stop), "create stop event");
    launch_prepare<N, D>(directions, states, stream);
    ck(cudaStreamSynchronize(stream), "prepare reusable state");
    auto call = [&](Route route) {
        launch<N, D>(route, directions, states, output, stream);
    };
    const unsigned batch = calibrate(
        [&] { call(Route::Sofl); }, start, stop, stream);
    std::vector<Route> routes = {Route::Sofl, Route::Fresh, Route::Prepared};
    std::mt19937_64 random(UINT64_C(0x435552414e444458) + session + N);
    for (unsigned i = 0; i < warmup; ++i) {
        std::shuffle(routes.begin(), routes.end(), random);
        for (Route route : routes) call(route);
    }
    ck(cudaStreamSynchronize(stream), "warmup synchronize");
    for (unsigned repetition = 0; repetition < repetitions; ++repetition) {
        std::shuffle(routes.begin(), routes.end(), random);
        for (Route route : routes) {
            ck(cudaStreamSynchronize(stream), "pre-sample synchronize");
            ck(cudaEventRecord(start, stream), "sample start");
            for (unsigned i = 0; i < batch; ++i) call(route);
            ck(cudaEventRecord(stop, stream), "sample stop");
            ck(cudaEventSynchronize(stop), "sample synchronize");
            float milliseconds = 0;
            ck(cudaEventElapsedTime(&milliseconds, start, stop),
               "sample elapsed");
            std::printf(
                "{\"type\":\"sample\",\"route\":\"%s\",\"points\":%u,"
                "\"dimensions\":%u,\"session\":%u,\"repetition\":%u,"
                "\"event_ns\":%.9f,\"batch\":%u,"
                "\"logical_values\":%zu,\"final_output_bytes\":%u}\n",
                route_name(route), N, D, session, repetition,
                static_cast<double>(milliseconds) * 1.0e6 / batch, batch,
                static_cast<std::size_t>(N) * D, work * 4U);
        }
        std::fflush(stdout);
    }
    for (unsigned i = 0; i < warmup; ++i) {
        launch_prepare<N, D>(directions, states, stream);
    }
    ck(cudaStreamSynchronize(stream), "setup warmup synchronize");
    for (unsigned repetition = 0; repetition < repetitions; ++repetition) {
        ck(cudaEventRecord(start, stream), "setup start");
        launch_prepare<N, D>(directions, states, stream);
        ck(cudaEventRecord(stop, stream), "setup stop");
        ck(cudaEventSynchronize(stop), "setup synchronize");
        float milliseconds = 0;
        ck(cudaEventElapsedTime(&milliseconds, start, stop), "setup elapsed");
        std::printf(
            "{\"type\":\"sample\",\"route\":\"curanddx_setup\","
            "\"points\":%u,\"dimensions\":%u,\"session\":%u,"
            "\"repetition\":%u,\"event_ns\":%.9f,\"batch\":1,"
            "\"logical_values\":%zu,\"state_bytes\":%zu}\n",
            N, D, session, repetition,
            static_cast<double>(milliseconds) * 1.0e6,
            static_cast<std::size_t>(N) * D, work * sizeof(Rng));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(states);
    cudaFree(output);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: compare check | bench N D session warmup repetitions\n");
        return 2;
    }
    int device = 0;
    cudaDeviceProp properties{};
    ck(cudaGetDevice(&device), "get device");
    ck(cudaGetDeviceProperties(&properties, device), "get device properties");
    if (properties.major != 7 || properties.minor != 5) {
        std::fprintf(stderr, "this library is built for Tesla T4 / sm_75\n");
        return 2;
    }
    curandDirectionVectors32_t* host = nullptr;
    ck_curand(curandGetDirectionVectors32(
        &host, CURAND_DIRECTION_VECTORS_32_JOEKUO6),
        "get Joe-Kuo 6 vectors");
    Direction* directions = nullptr;
    ck(cudaMalloc(&directions, 32 * sizeof(Direction)),
       "allocate direction vectors");
    ck(cudaMemcpy(directions, host, 32 * sizeof(Direction),
                  cudaMemcpyHostToDevice), "copy direction vectors");
    std::printf("{\"type\":\"machine\",\"gpu\":\"%s\","
                "\"compute_capability\":\"%d.%d\",\"sm_count\":%d,"
                "\"curanddx_version\":%d,\"cuda_runtime\":%d}\n",
                properties.name, properties.major, properties.minor,
                properties.multiProcessorCount, CURANDDX_VERSION,
                CUDART_VERSION);
    const std::string mode = argv[1];
    if (mode == "check" && argc == 2) {
        correctness<65536U, 32U>(directions, host);
        correctness<60000000U, 1U>(directions, host);
        std::puts("PUBLIC_BINARY_CORRECTNESS=PASS");
    } else if (mode == "bench" && argc == 7) {
        const unsigned points = std::stoul(argv[2]);
        const unsigned dimensions = std::stoul(argv[3]);
        const unsigned session = std::stoul(argv[4]);
        const unsigned warmup = std::stoul(argv[5]);
        const unsigned repetitions = std::stoul(argv[6]);
        if (!repetitions || repetitions > 1000U || warmup > 1000U) return 2;
        if (points == 65536U && dimensions == 32U)
            benchmark<65536U, 32U>(directions, session, warmup, repetitions);
        else if (points == 60000000U && dimensions == 1U)
            benchmark<60000000U, 1U>(
                directions, session, warmup, repetitions);
        else return 2;
    } else {
        return 2;
    }
    cudaFree(directions);
    return 0;
}
