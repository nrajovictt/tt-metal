// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct EluBf16Config {
    static constexpr uint32_t kSegments = 2u;
    static constexpr uint32_t kCoefficientOffset = kSegments + 1;
    static constexpr uint32_t kCoefficientsPerSegment = 16u;
    static constexpr uint32_t kDegree[] = {14, 1};
    static constexpr uint32_t kLutBits[] = {
        0xc1200000u, 0x00000000u, 0x41200000u, 0x00000000u, 0x3f800000u, 0x3effffeau, 0x3e2aa982u,
        0x3d2a9f4au, 0x3c084e54u, 0x3ab4a13au, 0x394a3902u, 0x37bf4b2eu, 0x36161ca6u, 0x343c9f4au,
        0x32342bb8u, 0x2ff2d2c2u, 0x2d4bc41cu, 0x2a1f125cu, 0x00000000u, 0x00000000u, 0x3f800000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};
    static constexpr int kParkRows[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static constexpr uint32_t kRowBase = 32u;
    static constexpr uint32_t kRowLimit = 128u;
    static constexpr uint32_t kParkedCount = 0u;
    static constexpr uint32_t kLowerBits = 0xc1200000u;
    static constexpr uint32_t kTerminalBits = 0xbf800000u;
    static constexpr bool kOrdered = false;
    static constexpr bool kRawIngress = true;
    static constexpr bool kRawTerminal = false;
    static constexpr uint32_t degree(uint32_t segment) { return kDegree[segment]; }
    static constexpr int park_row(uint32_t index) { return kParkRows[index]; }
    static constexpr float lut(uint32_t index) { return __builtin_bit_cast(float, kLutBits[index]); }
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_cascade_dense.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_elu_bf16() {
    ckernel::sfpu::bf16::calculate_dense_polynomial<ckernel::sfpu::EluBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
