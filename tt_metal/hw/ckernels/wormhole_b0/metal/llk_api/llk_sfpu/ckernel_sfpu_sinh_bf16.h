// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SinhBf16Config {
    static constexpr uint32_t kDegree = 4u;
    static constexpr bool kOdd = true;
    static constexpr bool kBare = false;
    static constexpr uint32_t kScaledCoefficientBits[] = {
        0x3f7ffebfu, 0x33b16ce3u, 0x277740bdu, 0x1ad505f7u, 0x0e5c710fu};
    static constexpr uint32_t kMultiplierBits = 0x3fb8aa3bu;
    static constexpr uint32_t kComposeScaleBits = 0x3e8002f9u;
    static constexpr uint32_t kOriginCubicBits = 0x3e2a0000u;
    static constexpr uint32_t kTerminalBoundBits = 0x42b30000u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_hyperbolic_exp.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_sinh_bf16() {
    ckernel::sfpu::bf16::calculate_hyperbolic_exp<ckernel::sfpu::SinhBf16Config, ITERATIONS>();
}
inline void init_sinh_bf16() { ckernel::sfpu::bf16::init_hyperbolic_exp<ckernel::sfpu::SinhBf16Config>(); }

}  // namespace ckernel::sfpu
