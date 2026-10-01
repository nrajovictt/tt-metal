// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_bf16_newton_root_core.h"
// Sibling stock wrappers may include another generated config while this
// shared runtime is being parsed. Expose its adapter interface first.
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_newton_root();
template <typename Config, int Iterations = 8>
inline void calculate_newton_root();
}  // namespace ckernel::sfpu::bf16
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_newton_root() {
    if constexpr (Config::kKind == 2) {
        sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kC0Bits);
    } else {
        sfpi::vConstIntPrgm0 = Config::kMagic;
    }
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kC1Bits);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kC2Bits);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}

template <typename Config, int Iterations>
inline void calculate_newton_root() {
    static_assert(Config::kKind <= 2);
    static_assert(Iterations == 8 || Iterations == 32);
    static_assert(Config::kKind != 0 || Config::kNegativeExponentZeroOutput == 0x7f80u);
    static_assert(Config::kBodySlots == (Config::kKind == 0 ? 32u : Config::kKind == 1 ? 31u : 23u));
    if constexpr (Config::kKind == 2) {
        constexpr float negative_inv_n_scaled = (-1.0f / 3.0f) / 256.0f;
        constexpr float magic = ((float)Config::kMagic) / 256.0f + 8388608.0f;
        constexpr uint32_t scale_bits = __builtin_bit_cast(uint32_t, negative_inv_n_scaled);
        constexpr uint32_t magic_bits = __builtin_bit_cast(uint32_t, magic);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, scale_bits >> 16);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, scale_bits & 0xffffu);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, magic_bits >> 16);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, magic_bits & 0xffffu);
    }
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    if constexpr (Config::kKind == 0) {
        static_assert(Config::kNegativeExponentZeroOutput == 0x7f80u);
        sfpi::newton_sqrt_body<true, true, false, ADDR_MOD_7, ADDR_MOD_6>();
    } else if constexpr (Config::kKind == 1) {
        sfpi::newton_rsqrt_body<ADDR_MOD_7, ADDR_MOD_6>();
    } else {
        sfpi::newton_cbrt_body<ADDR_MOD_7, ADDR_MOD_6>();
    }
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
}  // namespace ckernel::sfpu::bf16
