// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <array>
namespace ckernel::sfpu {
struct MishBf16Config {
    static constexpr float kMultiplier = __builtin_bit_cast(float, 0x3fb8aa3bu);
    static constexpr float kBias = __builtin_bit_cast(float, 0x43060000u);
    static constexpr float kScale = __builtin_bit_cast(float, 0x43000000u);
    static constexpr float kNegativeSlope = __builtin_bit_cast(float, 0x3f804189u);
    static constexpr float kNegativeBias = __builtin_bit_cast(float, 0x43800000u);
    static constexpr float kPositiveScale = __builtin_bit_cast(float, 0x46ffbe77u);
    static constexpr float kLowerBound = __builtin_bit_cast(float, 0x00000000u);
    static constexpr float kUpperBound = __builtin_bit_cast(float, 0x00000000u);
    static constexpr uint32_t kRawNegativeNanWord = 32640u;
    static constexpr bool kSymmetric = false;
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0x3f803884u),
        __builtin_bit_cast(float, 0x3f285adeu),
        __builtin_bit_cast(float, 0x3eaca410u)};
    static constexpr uint32_t kBiasBf16 = 0x4306u;
    static constexpr uint32_t kCoordinateUpperBf16 = 17172u;
    static constexpr uint32_t kOriginEncoded = 0u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_exponent_alu_product.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_mish_bf16() {
    ckernel::sfpu::bf16::calculate_exponent_alu_product<ckernel::sfpu::MishBf16Config, ITERATIONS>();
}
inline void init_mish_bf16() { ckernel::sfpu::bf16::init_exponent_alu_product<ckernel::sfpu::MishBf16Config>(); }

}  // namespace ckernel::sfpu
