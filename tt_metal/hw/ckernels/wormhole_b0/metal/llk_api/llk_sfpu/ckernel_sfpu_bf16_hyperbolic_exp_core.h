// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Selected P4 exp/reciprocal bodies. Callers own init, traversal and stores.
namespace sfpi {

template <typename Config>
__attribute__((always_inline)) inline vFloat hyperbolic_exp_wh(
    vFloat x, vFloat multiplier, vFloat compose_scale, vFloat original) {
    if constexpr (Config::kOdd) {
        x = setsgn(x, 0);
    }
    l_reg[LRegs::LReg4] = vFloat(126.0f);
    vFloat bias_register = l_reg[LRegs::LReg4];
    vFloat xlog2 = x * multiplier + bias_register;
    vInt e = exexp(xlog2);
    vInt m = exman(xlog2, MantissaMode::ImplicitOne);
    m = shft(m, e, ShiftMode::Logical);
    vFloat z = as<vFloat>(m);
    vInt ep = exexp(z, ExponentMode::Biased);
    vMag fm = exman(z);
    vFloat f = convert<vFloat>(fm, RoundMode::Nearest);
    vFloat c4 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[4]);
    vFloat c3 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[3]);
    vFloat c2 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[2]);
    vFloat p = c4 * f + c3;
    vFloat c1 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[1]);
    p = p * f + c2;
    vFloat c0 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[0]);
    p = p * f + c1;
    p = p * f + c0;
    vFloat y;
    if constexpr (Config::kBare) {
        y = setexp(p, ep);
    } else {
        y = p * as<vFloat>(ep << 23);
    }
    vFloat odd_origin_ax;
    if constexpr (Config::kOdd) {
        odd_origin_ax = setsgn(original, 0);
    }
    vFloat negative_mantissa = -p;
    vFloat reciprocal = vConstFloatPrgm1 + vConstFloatPrgm0 * negative_mantissa;
    vInt inverse_exponent = ~ep;
    reciprocal = vConstFloatPrgm2 + reciprocal * negative_mantissa;
    inverse_exponent += 255;
    vFloat error = 1.0f + negative_mantissa * reciprocal;
    vFloat power = setexp(vFloat(1.0f), inverse_exponent);
    reciprocal = reciprocal + reciprocal * error;
    vFloat odd_origin_term;
    if constexpr (Config::kOdd) {
        odd_origin_term = __builtin_bit_cast(float, Config::kOriginCubicBits) * odd_origin_ax;
    }
    vFloat hyperbolic_inv = reciprocal * power;
    vFloat odd_origin_factor;
    if constexpr (Config::kOdd) {
        odd_origin_factor = 1.0f + odd_origin_term * odd_origin_ax;
    }
    if constexpr (Config::kOdd) {
        vFloat magnitude = compose_scale * hyperbolic_inv + y;
        vFloat lower = odd_origin_ax * odd_origin_factor;
        ordered_min_max(lower, magnitude);
        y = magnitude;
    } else {
        y = compose_scale * hyperbolic_inv + y;
    }
    return y;
}

template <typename Config>
inline vFloat hyperbolic_wh_terminal(vFloat x, vFloat y) {
    vFloat magnitude = x;
    if constexpr (Config::kOdd) {
        magnitude = setsgn(x, 0);
    }
    v_if(magnitude >= __builtin_bit_cast(float, Config::kTerminalBoundBits)) {
        y = std::numeric_limits<float>::infinity();
    }
    v_endif;
    if constexpr (Config::kOdd) {
        y = copysgn(y, x);
    }
    return y;
}
}  // namespace sfpi
