// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_min_max.h"
}
#include "ckernel_sfpu_bf16_hyperbolic_exp_core.h"
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_hyperbolic_exp() {
    sfpu_reciprocal_init<false>();
}
template <typename Config, int Iterations = 32>
inline void calculate_hyperbolic_exp() {
    static_assert(Iterations == 32 && Config::kDegree == 4);
    static_assert(Config::kOdd ? Config::kOriginCubicBits == 0x3e2a0000u : Config::kBare);
    sfpi::l_reg[sfpi::LRegs::LReg7] = sfpi::vFloat(__builtin_bit_cast(float, Config::kMultiplierBits));
    sfpi::vFloat multiplier = sfpi::l_reg[sfpi::LRegs::LReg7];
    sfpi::l_reg[sfpi::LRegs::LReg6] =
        sfpi::vFloat(__builtin_bit_cast(float, Config::kComposeScaleBits ^ (Config::kOdd ? 0x80000000u : 0u)));
    sfpi::vFloat scale = sfpi::l_reg[sfpi::LRegs::LReg6];
#pragma GCC unroll 32
    for (int row = 0; row < Iterations; ++row) {
        sfpi::vFloat x = sfpi::dst_reg[row];
        if constexpr (!Config::kOdd) {
            x = sfpi::setsgn(x, 0);
        }
        sfpi::l_reg[sfpi::LRegs::LReg5] = x;
        sfpi::vFloat saved = sfpi::l_reg[sfpi::LRegs::LReg5];
        sfpi::vFloat y = sfpi::hyperbolic_exp_wh<Config>(saved, multiplier, scale, saved);
        y = sfpi::hyperbolic_wh_terminal<Config>(saved, y);
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[row] = y;
    }
}
}  // namespace ckernel::sfpu::bf16
