// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SeluBf16Config {
    static constexpr uint32_t kSegments = 2u;
    static constexpr uint32_t kCoefficientOffset = kSegments + 1;
    static constexpr uint32_t kCoefficientsPerSegment = 12u;
    static constexpr uint32_t kDegree[] = {11, 1};
    static constexpr uint32_t kLutBits[] = {
        0xc1200000u, 0x00000000u, 0x41200000u, 0x00000000u, 0x3fe10916u, 0x3f60fd44u, 0x3e95b200u,
        0x3d94228au, 0x3c645a36u, 0x3b09de58u, 0x397ea37cu, 0x37abb264u, 0x359d79eau, 0x332e0546u,
        0x302dbd62u, 0x00000000u, 0x3f867d5fu, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};
    static constexpr int kParkRows[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static constexpr uint32_t kRowBase = 64u;
    static constexpr uint32_t kRowLimit = 108u;
    static constexpr uint32_t kParkedCount = 0u;
    static constexpr uint32_t kLowerBits = 0xc1200000u;
    static constexpr uint32_t kTerminalBits = 0xbfe10000u;
    static constexpr bool kOrdered = true;
    static constexpr bool kRawIngress = false;
    static constexpr bool kRawTerminal = false;
    static constexpr uint32_t degree(uint32_t segment) { return kDegree[segment]; }
    static constexpr int park_row(uint32_t index) { return kParkRows[index]; }
    static constexpr float lut(uint32_t index) { return __builtin_bit_cast(float, kLutBits[index]); }
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_cascade_dense.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_selu_bf16() {
    ckernel::sfpu::bf16::calculate_dense_polynomial<ckernel::sfpu::SeluBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
