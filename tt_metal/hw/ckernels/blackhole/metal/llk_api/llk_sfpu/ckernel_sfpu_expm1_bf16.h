// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct Expm1Bf16Config {
    static constexpr unsigned kLowerBf16 = 0x0000c0c8u;
    static constexpr unsigned kUpperBf16 = 0x000042b2u;
    static constexpr unsigned kRoundingBiasBits = 0x4b400000u;
    static constexpr unsigned kRawEqualMask = 0x000080ffu;
    static constexpr unsigned kRawEqualValue = 0x000080ffu;
    static constexpr unsigned kRawNonzeroMask = 0x00007f00u;
    static constexpr unsigned kRawCount = 0x0000007fu;
    static constexpr unsigned kInfinityExponent = 0x000000ffu;
    static constexpr unsigned kInverseScaleBits = 0x3fb8aa3bu;
    static constexpr unsigned kResidualScaleBits = 0xbf317218u;
    static constexpr unsigned kHalfBits = 0x3f000000u;
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x3f000000u),
        __builtin_bit_cast(float, 0x3e2aaa56u),
        __builtin_bit_cast(float, 0x3d2b440cu),
        __builtin_bit_cast(float, 0x3c0914c2u)};
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_factored_cw_expm1.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_expm1_bf16() {
    ckernel::sfpu::bf16::calculate_factored_cw_expm1<ckernel::sfpu::Expm1Bf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
