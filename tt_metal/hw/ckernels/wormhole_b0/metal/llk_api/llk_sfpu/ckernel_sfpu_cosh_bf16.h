// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct CoshBf16Config {
    static constexpr uint32_t kDegree = 4u;
    static constexpr bool kOdd = false;
    static constexpr bool kBare = true;
    static constexpr uint32_t kScaledCoefficientBits[] = {
        0x3f800072u, 0x33b165a1u, 0x27774d30u, 0x1ad524a6u, 0x0e5dd2acu};
    static constexpr uint32_t kMultiplierBits = 0x3fb8aa3bu;
    static constexpr uint32_t kComposeScaleBits = 0x3e8001d5u;
    static constexpr uint32_t kOriginCubicBits = 0x00000000u;
    static constexpr uint32_t kTerminalBoundBits = 0x42b30000u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_hyperbolic_exp.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_cosh_bf16() {
    ckernel::sfpu::bf16::calculate_hyperbolic_exp<ckernel::sfpu::CoshBf16Config, ITERATIONS>();
}
inline void init_cosh_bf16() { ckernel::sfpu::bf16::init_hyperbolic_exp<ckernel::sfpu::CoshBf16Config>(); }

}  // namespace ckernel::sfpu
