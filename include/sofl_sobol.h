#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#if defined(_WIN32)
#define SOFL_SOBOL_API __declspec(dllimport)
#else
#define SOFL_SOBOL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Dimension-major output: output[dimension_slot * points + point].
// first_dimension is one-based. Supported point counts are exactly 8192,
// 65536 and 60000000; the available dimensions are D1 through D64.
SOFL_SOBOL_API cudaError_t sofl_sobol_generate_u32(
    uint32_t* output, uint32_t points, uint32_t first_dimension,
    uint32_t dimension_count, cudaStream_t stream);

// Exact mapping: ((raw >> 9) | 0x3f800000) interpreted as float, minus 1.0f.
SOFL_SOBOL_API cudaError_t sofl_sobol_generate_f32(
    float* output, uint32_t points, uint32_t first_dimension,
    uint32_t dimension_count, cudaStream_t stream);

// Transparent benchmark consumer. One uint32 checksum is written per
// 256-value chunk after u=normalize(raw), x=fma(0.3,u,0.7), and modular
// accumulation of bit_cast<uint32_t>(x*x).
SOFL_SOBOL_API cudaError_t sofl_sobol_fused_checksum_u32(
    uint32_t* output, uint32_t points, uint32_t first_dimension,
    uint32_t dimension_count, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#undef SOFL_SOBOL_API
