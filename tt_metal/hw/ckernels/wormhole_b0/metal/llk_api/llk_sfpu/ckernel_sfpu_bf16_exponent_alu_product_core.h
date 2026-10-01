// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Selected product bodies. Caller owns initialization and SFPU start/done.
namespace sfpi {
template <class Config, uint32_t Hold>
inline void product_symmetric_wh() {
    static_assert(Config::kDomainActionCount == 3);
    static_assert(sizeof(Config::kCoefficients) / sizeof(float) == 3);
    constexpr uint32_t origin_bf16 = ((Config::kOriginEncoded & 0xffu) << 7) | ((Config::kOriginEncoded >> 8) & 0x7fu);
    static_assert(origin_bf16 > 0 && origin_bf16 < 2048);
    constexpr uint32_t middle = __builtin_bit_cast(uint32_t, Config::kCoefficients[1] * 0x1p-23f);
    constexpr uint32_t top = __builtin_bit_cast(uint32_t, Config::kCoefficients[2] * 0x1p-46f);
    constexpr uint32_t bottom = __builtin_bit_cast(uint32_t, Config::kCoefficients[0]);
    constexpr uint32_t multiplier = __builtin_bit_cast(uint32_t, (float)Config::kMultiplier);
    TTI_SFPLOADI(6, SFPLOADI_MOD0_LOWER, multiplier & 65535);
    TTI_SFPLOADI(6, SFPLOADI_MOD0_UPPER, multiplier >> 16);
    // Upstream LLK blocks several tiles in one DEST half, so the three coefficients
    // are immediates rather than DEST rows. Their loads do not fit the 32-slot replay
    // body, so it is recorded in two parts around one MAD and the last coefficient load.
#pragma GCC unroll 32
    for (int row = 0; row < 32; ++row) {
        TTI_SFPLOAD(2, 0, Hold, row * 2);
        if (row == 0) {
            TTI_REPLAY(0, 13, 1, 1);
            TTI_SFPSETSGN(0, 2, 0, 1);
            TTI_SFPLOADI(1, SFPLOADI_MOD0_FLOATB, Config::kBiasBf16);
            TTI_SFPMAD(0, 6, 1, 0, 0);
            TTI_SFPLOADI(4, SFPLOADI_MOD0_LOWER, middle & 65535);
            TTI_SFPLOADI(4, SFPLOADI_MOD0_UPPER, middle >> 16);
            TTI_SFPEXEXP(0, 0, 3, 0);
            TTI_SFPEXMAN(0, 0, 1, 0);
            TTI_SFPSHFT(0, 3, 1, 0);
            TTI_SFPEXEXP(0, 1, 0, 1);
            TTI_SFPEXMAN(0, 1, 1, 1);
            TTI_SFPCAST(1, 1, 0);
            TTI_SFPLOADI(5, SFPLOADI_MOD0_LOWER, top & 65535);
            TTI_SFPLOADI(5, SFPLOADI_MOD0_UPPER, top >> 16);
        } else {
            TTI_REPLAY(0, 13, 0, 0);
        }
        TTI_SFPMAD(5, 1, 4, 3, 0);
        TTI_SFPLOADI(7, SFPLOADI_MOD0_LOWER, bottom & 65535);
        TTI_SFPLOADI(7, SFPLOADI_MOD0_UPPER, bottom >> 16);
        if (row == 0) {
            TTI_REPLAY(13, 19, 1, 1);
            TTI_SFPMAD(3, 1, 7, 1, 0);
            TTI_SFPLOADI(4, SFPLOADI_MOD0_FLOATB, __builtin_bit_cast(uint32_t, Config::kDomainActions[2].bound) >> 16);
            TTI_SFPSETEXP(0, 1, 0, 0);
            TTI_SFPLOADI(1, SFPLOADI_MOD0_FLOATB, __builtin_bit_cast(uint32_t, -Config::kScale) >> 16);
            TTI_SFPMAD(0, 1, 11, 1, 0);
            TTI_SFPMAD(11, 2, 4, 4, 0);
            TTI_SFPMAD(12, 1, 13, 3, 0);
            TTI_SFPLOADI(5, SFPLOADI_MOD0_FLOATB, __builtin_bit_cast(uint32_t, Config::kDomainActions[1].bound) >> 16);
            TTI_SFPMAD(3, 1, 14, 3, 0);
            TTI_SFPMAD(11, 2, 5, 5, 0);
            TTI_SFPMAD(1, 3, 10, 1, 0);
            TTI_SFPSETSGN(0, 2, 7, 1);
            TTI_SFPMAD(3, 1, 3, 1, 0);
            TTI_SFPMUL(7, 0, 9, 0, 0);
            TTI_SFPMULI(__builtin_bit_cast(uint32_t, -2.0f * Config::kScale) >> 16, 1, 0);
            TTI_SFPADD(10, 7, 2, 3, 0);
            TTI_SFPSHFT((-16 & 4095), 7, 7, 1);
            TTI_SFPMAD(0, 1, 3, 0, 0);
            TTI_SFPIADD((-origin_bf16 & 4095), 7, 7, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);
        } else {
            TTI_REPLAY(13, 19, 0, 0);
        }
        TTI_SFPSWAP(0, 0, 2, 1);
        TTI_SFPMULI(0x3f00, 0, 0);
        TTI_SFPLOAD(1, 0, Hold, row * 2);
        TTI_SFPSETCC(0, 7, 0, SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETMAN(0, 1, 0, 1);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSETCC(0, 4, 0, SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPMOV(0, 1, 0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSETCC(0, 5, 0, SFPSETCC_MOD1_LREG_GTE0);
        TTI_SFPMUL(1, 9, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, 0, 0, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(0, 0, Hold, row * 2);
    }
}
template <class Config, class Pin>
__attribute__((always_inline)) inline vFloat product_bounded_wh_core(
    vFloat x, vFloat multiplier, vFloat c0, vFloat top, Pin pin_square) {
    vFloat x_original = x;
    vFloat bias_register = Config::kBias;
    vFloat xlog2 = __builtin_rvtt_sfpmad(x.get(), multiplier.get(), bias_register.get(), SFPMAD_MOD1_OFFSET_NONE);
    vInt e = exexp(xlog2);
    vInt m = exman(xlog2, MantissaMode::ImplicitOne);
    m = shft(m, e, ShiftMode::Logical);
    vFloat z = as<vFloat>(m);
    vFloat promoted_middle = Config::kCoefficients[1] * 0x1p-23f;
    vInt ep = exexp(z, ExponentMode::Biased);
    vMag fm = exman(z);
    vFloat f = convert<vFloat>(fm, RoundMode::Nearest);
    vFloat p = top * f + promoted_middle;
    p = p * f + c0;
    vFloat y = setexp(p, ep);
    vFloat shifted = y + Config::kScale;
    vFloat square_constant = pin_square(Config::kScale * Config::kScale);
    vFloat denominator = shifted * shifted + square_constant;
    vFloat negative_x = copyman(-1.0f, denominator);
    vFloat inv = vConstFloatPrgm1 + vConstFloatPrgm0 * negative_x;
    vUInt scale_bits = ~as<vUInt>(denominator);
    inv = vConstFloatPrgm2 + inv * negative_x;
    vFloat scale = setman(as<vFloat>(scale_bits), 0);
    vFloat t = 1.0f + negative_x * inv;
    scale *= 0.5f;
    inv = inv + inv * t;
    inv = inv * scale;
    vFloat negative_slope = -Config::kNegativeSlope;
    vFloat bias = -Config::kNegativeBias;
    vFloat negative_numerator =
        __builtin_rvtt_sfpmad(negative_slope.get(), y.get(), bias.get(), SFPMAD_MOD1_OFFSET_NONE);
    vFloat positive_scale = Config::kPositiveScale;
    vFloat negative_scale = positive_scale;
    vFloat one = 1.0f;
    vFloat factor = __builtin_rvtt_sfpmad(negative_scale.get(), inv.get(), one.get(), SFPMAD_MOD1_OFFSET_NONE);
    vFloat coordinate = x_original;
    v_if(x_original < 0.0f) {
        factor = negative_numerator * inv;
        coordinate = coordinate * y;
    }
    v_endif;
    y = coordinate * factor;
    return y;
}

template <class Config>
inline vFloat product_bounded_wh_prepare(vFloat raw) {
    vFloat prepared = raw;
    v_if(prepared < Config::kLowerBound) { prepared = Config::kLowerBound; }
    v_elseif(prepared > Config::kUpperBound) { prepared = Config::kUpperBound; }
    v_endif;
    return prepared;
}

template <class Config>
inline void product_bounded_wh_terminal(vFloat raw, vFloat& result) {
    constexpr uint32_t origin_bf16 = (Config::kOriginEncoded & 0x8000u) | ((Config::kOriginEncoded & 0xffu) << 7) |
                                     ((Config::kOriginEncoded >> 8) & 0x7fu);
    v_if(as<vUInt>(raw) == vUInt(origin_bf16 << 16)) { result = -std::numeric_limits<float>::min(); }
    v_endif;
    vFloat minus_one = -1.0f;
    vFloat upper = Config::kDomainActions[2].bound;
    vFloat upper_distance = __builtin_rvtt_sfpmad(minus_one.get(), raw.get(), upper.get(), SFPMAD_MOD1_OFFSET_NONE);
    vFloat lower = Config::kDomainActions[1].bound;
    vFloat lower_distance = __builtin_rvtt_sfpmad(minus_one.get(), raw.get(), lower.get(), SFPMAD_MOD1_OFFSET_NONE);
    v_if(upper_distance < 0.0f) { result = raw; }
    v_endif;
    v_if(lower_distance >= 0.0f) { result = raw * 0.0f; }
    v_endif;
}

template <LRegs Register>
sfpi_inline vFloat product_pin(vFloat value) {
    l_reg[Register] = value;
    return l_reg[Register];
}
template <class Config>
inline void product_bounded_wh_tile() {
    vFloat top = product_pin<LRegs::LReg5>(Config::kCoefficients[2] * 0x1p-46f);
    vFloat c0 = product_pin<LRegs::LReg7>(Config::kCoefficients[0]);
    vFloat multiplier = product_pin<LRegs::LReg6>(Config::kMultiplier);
#pragma GCC unroll 32
    for (int row = 0; row < 32; ++row) {
        vFloat x = product_bounded_wh_prepare<Config>(dst_reg[row]);
        vFloat result = product_bounded_wh_core<Config>(
            x, multiplier, c0, top, [](vFloat value) { return product_pin<LRegs::LReg4>(value); });
        vFloat raw = dst_reg[row];
        product_bounded_wh_terminal<Config>(raw, result);
        result = convert<vFloat16b>(result, RoundMode::Nearest);
        dst_reg[row] = result;
    }
}

}  // namespace sfpi
