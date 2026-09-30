// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Replay-buffer slots for the per-iteration register-only compute bodies of
// the quant / requant / dequant kernels. The body is recorded once by each
// op's _init_{quant,requant,dequant}_int32_ and then replayed by every
// invocation of the matching _{quant,requant,dequant}_int32_ kernel.
// Unlike the Blackhole port, the recorded body does not depend on
// SIGN_MAGNITUDE_FORMAT: WH performs the int32 sign-magnitude <-> 2's
// complement conversion in the SFPLOAD/SFPSTORE instr_mod0 field
// (INT32 = 4 vs INT32_2S_COMP = 12), and those load/store instructions are
// outside the replay window. So one slot per kernel suffices.
//
// Distinct slots between kernels are required so a single compute kernel
// can mix all three ops without each init clobbering the others' recordings.
// Int8 output records a longer body (the offset-128 pack) into the same family
// slot; slot spacing uses each family's max body length so the three ops can
// still coexist.
//
// Body content (see the inits for the exact emission order). SFPMAD / SFPADD have a 2-cycle
// write latency on WH, so each is followed by one SFPNOP before its result is read.
// <RNE MAD> and <RNE requant MAD> round to nearest even (see RNE_MAGIC_FP32); <excess-128 pack> is
// SFPSETCC, SFPMOV, SFPENCC, STOCH_RND followed by SFPIADD (int32 output) or SFPXOR (int8 output).
//   QUANT   (9)            : <RNE MAD: SFPMAD, SFPNOP, SFPADD, SFPNOP>, <excess-128 pack>   (int32 output)
//                            <RNE MAD>, STOCH_RND, 4x SFPNOP                                (uint8 output)
//   QUANT   (int8-out, 9)  : <RNE MAD>, <excess-128 pack>
//   REQUANT (12)           : SFPCAST(int->fp32), <RNE requant MAD: SFPMAD, SFPNOP, SFPADD, SFPNOP,
//                            SFPADD, SFPNOP>, <excess-128 pack>
//                            (uint8 output: STOCH_RND, 4x SFPNOP in place of the pack)
//   REQUANT (int8-out, 12) : SFPCAST(int->fp32), <RNE requant MAD>, <excess-128 pack>
//   DEQUANT (5)            : SFPCAST(int->fp32), SFPADD, SFPNOP, SFPMUL, SFPNOP
//
// calculate_quant_int32 and calculate_requant_int32 replay one length for both Int32 and UInt8
// output, so the UInt8 body is padded with SFPNOPs to the Int32 body's length.
//
// The int32 and int8 outputs both go through the excess-128 pack (see INT8_OFFSET_128_IMM16). Quant folds
// the +128 offset and -RNE_MAGIC into the fp32 zero-point once at init, so the per-iteration MAD + ADD already
// yields RNE(v) + 128 with no per-element SFPADDI. Requant keeps the host zero-point in LREG2 and carries
// the +128 in the constant that removes RNE_MAGIC.
constexpr std::uint32_t QUANT_REPLAY_SLOT = 0;
constexpr std::uint32_t QUANT_REPLAY_LEN = 9;
constexpr std::uint32_t QUANT_REPLAY_LEN_INT8_OUT = 9;
constexpr std::uint32_t QUANT_REPLAY_LEN_MAX = QUANT_REPLAY_LEN_INT8_OUT;
static_assert(QUANT_REPLAY_LEN <= QUANT_REPLAY_LEN_MAX, "QUANT_REPLAY_LEN_MAX must cover every quant body");

constexpr std::uint32_t REQUANT_REPLAY_SLOT = QUANT_REPLAY_SLOT + QUANT_REPLAY_LEN_MAX;
constexpr std::uint32_t REQUANT_REPLAY_LEN = 12;
constexpr std::uint32_t REQUANT_REPLAY_LEN_INT8_OUT = 12;
constexpr std::uint32_t REQUANT_REPLAY_LEN_MAX = REQUANT_REPLAY_LEN_INT8_OUT;
static_assert(REQUANT_REPLAY_LEN <= REQUANT_REPLAY_LEN_MAX, "REQUANT_REPLAY_LEN_MAX must cover every requant body");

constexpr std::uint32_t DEQUANT_REPLAY_SLOT = REQUANT_REPLAY_SLOT + REQUANT_REPLAY_LEN_MAX;
constexpr std::uint32_t DEQUANT_REPLAY_LEN = 5;

// Int8 L1 pack path:
// FP32_TO_INT8 rounding cannot be used here since it produces a sign-magnitude result
// whose magnitude is only 7 bits, so it saturates at +/-127 and cannot represent -128
// (both -127 and -128 collapse to 127). It also leaves negatives in sign-magnitude form.
//
// Int8 output is instead packed through the UInt8 packer, which passes the low byte of
// the dest word through unsigned (saturating to [0, 255]). The SFPU keeps the 2's complement
// byte in the low 8 bits using an offset-128 that spans the full [-128, 127] range and saturates:
//   u = FP32_TO_UINT8(v + 128) (u in [0, 255], saturates both ends)
//   b = u ^ 0x80  (excess-128 -> 2's complement byte)
// The byte is stored via INT32_2S_COMP and the UInt8 packer then emits raw b.
//
// The Int32 output (int8-range values in an int32 container) takes the same route for the same reason,
// then subtracts 128 as an integer: s = u - 128 is a 2's-complement int32 in [-128, 127], which the
// quant/requant kernels store raw (INT32) instead of through the INT32_2S_COMP conversion.
constexpr std::uint32_t INT8_OFFSET_128_IMM16 = 0x4300u;
constexpr std::uint32_t INT8_SIGN_MASK = 0x00000080u;
constexpr std::uint32_t INT8_MINUS_128_IMM12 = -128 & 0xfff;  // -128 as SFPIADD's 12-bit signed immediate

// Round to nearest even:
// STOCH_RND's non-stochastic mode rounds half-way cases away from zero on WH/BH (sfpi's
// SFPSTOCHRND_RND_EVEN is only a deprecated alias of SFPSTOCHRND_RND_NEAREST), while torch and ONNX
// quantization round them to even. The SFPU adder does round to nearest even, so adding RNE_MAGIC
// (1.5 * 2^23) to any |v| < 2^22 lands in [2^23, 2^24), where the fp32 spacing is 1, rounding v to an
// integer with ties to even. Removing RNE_MAGIC again is exact because both terms are integers below
// 2^24. Larger |v| saturate anyway. STOCH_RND then only converts an integer-valued float, so its own
// tie rule no longer matters.
constexpr std::uint32_t RNE_MAGIC_FP32 = 0x4b400000u;            // 12582912.0f = 1.5 * 2^23
constexpr std::uint32_t RNE_NEG_MAGIC_FP32 = 0xcb400000u;        // -12582912.0f
constexpr std::uint32_t RNE_128_MINUS_MAGIC_FP32 = 0xcb3fff80u;  // 128.0f - 12582912.0f (exact)

// Quant init: LREG5 = +RNE_MAGIC and LREG2 = zero-point (+128) - RNE_MAGIC. The quant zero-point is an
// integer, so the fold is exact. LREG6 is scratch.
inline void _quant_rne_init_() {
    _sfpu_load_imm32_(p_sfpu::LREG6, RNE_NEG_MAGIC_FP32);
    TTI_SFPNOP;  // a preceding _int8_bias_zero_point_ SFPADDI may still be writing LREG2
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG2, 0 /*mod1*/);
    TTI_SFPNOP;
    _sfpu_load_imm32_(p_sfpu::LREG5, RNE_MAGIC_FP32);
}

// v = RNE(A * B) + zero-point (+128). The MAD rounds A * B + RNE_MAGIC to an integer (ties to even) in
// its single rounding step; the ADD removes RNE_MAGIC and applies the zero-point exactly.
inline void _quant_rne_mad_() {
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG0, 0 /*mod1*/);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);
    TTI_SFPNOP;
}

// Requant init: LREG5 = +RNE_MAGIC, LREG6 = (128 for the excess-128 outputs) - RNE_MAGIC. The requant
// zero-point in LREG2 is the host-folded z_out - z_in * (s_in / s_out), fractional in general, so neither
// constant can be folded into it without rounding it.
template <DataFormat OUTPUT_FORMAT>
inline void _requant_rne_init_() {
    _sfpu_load_imm32_(p_sfpu::LREG5, RNE_MAGIC_FP32);
    _sfpu_load_imm32_(
        p_sfpu::LREG6, OUTPUT_FORMAT == DataFormat::UInt8 ? RNE_NEG_MAGIC_FP32 : RNE_128_MINUS_MAGIC_FP32);
}

// v = RNE(A * B + zero-point) (+128): the documented requant expression rounded to nearest even.
inline void _requant_rne_mad_() {
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);  // v = A * B + zp
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG0, 0 /*mod1*/);  // + MAGIC: RNE(v)
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG0, 0 /*mod1*/);  // - MAGIC (+128)
    TTI_SFPNOP;
}

// Int8 L1 input path:
// Int8 input is read through the UInt8 unpacker, which zero-extends the raw byte
// into DST. The native Int8 unpacker is avoided because it widens negatives into a form that
// can be mishandled by subsequent SFPU INT32->FP32 cast.
//
// XOR-ing that byte with 0x80 recovers the excess-128 value e = s + 128, where s
// is the true signed value.
// For u < 128, u ^ 0x80 = u + 128 = s + 128
// For u >= 128 (s = u - 256), u ^ 0x80 = u - 128 = s + 128
// SFPU then casts e to fp32 and runs the normal formula. The constant 128 term is cancelled on the
// host by folding -128 into the zero point (dequant) or by an extra -128 * scale
// term in the zero-point constant (requant).
inline void _int8_input_unbias_() { TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG0, 0); }

// Configure the SFPU "dest += 2" addr_mod slot. With math::set_addr_mod_base()
// active, the SFPU addr_mod field "2" indexes real config slot ADDR_MOD_6,
// so writing this slot is what the per-iteration SFPSTOREs below pick up
// via ADDR_MOD_2. The addr-mod base is flipped on by every binary-SFPU
// dispatch: _llk_math_eltwise_binary_sfpu_params_ (in
// llk_math_eltwise_binary_sfpu_params.h) calls _llk_math_eltwise_sfpu_start_
// (in llk_math_eltwise_sfpu_common.h), which in turn calls
// math::set_addr_mod_base() before invoking the kernel body.
//
// The matching "no increment" slot used for the SFPLOADs (real ADDR_MOD_7,
// addressed as ADDR_MOD_3 from the SFPU instructions) is already programmed
// to {0,0,0} by eltwise_binary_sfpu_configure_addrmod() in the LLK init, so
// it doesn't need to be set here.
//
// quant_int32 isn't in the LLK init's "configure ADDR_MOD_6 with dest+=2"
// allow-list (that list covers mul_int32 / max / min / cmp ops), so we have
// to program it ourselves. The dest auto-increment of one SFPU dst row
// (sfpi::SFP_DESTREG_STRIDE == 2 dst-address units) is what walks dst_reg
// through the face's 4-row x 8-col blocks. Called once by each
// _init_{quant,requant,dequant}_int32_ since the addrmod state is per-tensix
// and only needs to be set up once. Replaces sfpi::dst_reg++ in the kernel
// bodies and lets each loop be purely TTI-issued.
inline void _quant_kernels_configure_dest_incr_addrmod_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = sfpi::SFP_DESTREG_STRIDE},
    }
        .set(ADDR_MOD_6);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_quant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A is input (fp32).
    // Operand B is scaling factor (fp32).
    // LREG2 holds the zero-point constant (fp32) loaded by _init_quant_int32_.
    // Output is int32 scaled to int8 range (sign-magnitude or 2's-complement
    // depending on SIGN_MAGNITUDE_FORMAT - the conversion is done by SFPSTORE).
    //
    // Tile layout in Dest: each tile occupies 64 dest-address units. Each
    // SFPLOAD/SFPSTORE moves 4 dest rows x 8 SFPU lanes, so advancing dst_reg
    // by +2 between iterations walks one face's eight 4-row x 8-col blocks
    // (= one full call site, ITERATIONS == 8).
    //
    // The replay-buffer body at QUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_quant_int32_, which must run before the
    // first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    // The recorded body leaves a 2's-complement int32 (or a non-negative uint8) in LREG0, so the
    // default 2's-complement output is stored raw. The sign-magnitude variant runs it through the
    // store's sign-magnitude <-> 2's-complement swap, which is its own inverse.
    constexpr InstrModLoadStore out_mode =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32_2S_COMP : InstrModLoadStore::INT32;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: inline TT_SFPLOADs (variable addresses can't live inside
    // the replay buffer because TT_* macros write to instrn_buffer[0]), replay
    // the recorded compute, then SFPSTORE under ADDR_MOD_2 (real slot 6) which
    // also auto-advances dst_reg by 2 for the next iteration's loads.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_3, in0_off);  // operand A (fp32)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);  // operand B (fp32 scaler)
        lltt::replay(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN);                        // RNE MAD + pack
        TT_SFPSTORE(p_sfpu::LREG0, out_mode, ADDR_MOD_2, out_off);                // store + dst_reg += 2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false, bool INT8_INPUT = false>
inline void calculate_requant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A is input to requant (int32, sign-magnitude or 2's complement bits or UInt8-unpacked int8 byte).
    // Operand B is scaling factor (fp32).
    // LREG2 holds the zero-point constant (fp32) loaded by _init_requant_int32_.
    // Output is int32 scaled to int8 range.
    //
    // The int32 in/out format conversion is done by the SFPLOAD/SFPSTORE
    // instr_mod0 field (INT32 vs INT32_2S_COMP); the recorded compute body
    // is identical for both SIGN_MAGNITUDE_FORMAT variants.
    //
    // The replay-buffer body at REQUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_requant_int32_, which must run before the
    // first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr InstrModLoadStore in_mode =
        (SIGN_MAGNITUDE_FORMAT && !INT8_INPUT) ? InstrModLoadStore::INT32 : InstrModLoadStore::INT32_2S_COMP;
    // The recorded body leaves a 2's-complement int32 (or a non-negative uint8) in LREG0; see
    // calculate_quant_int32 for the output store mode.
    constexpr InstrModLoadStore out_mode =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32_2S_COMP : InstrModLoadStore::INT32;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: hoist both TT_SFPLOADs ahead of the recorded compute
    // (the int->fp cast doesn't touch LREG1 so reordering is safe), replay
    // the recorded body, then SFPSTORE under ADDR_MOD_2 which auto-advances
    // dst_reg by 2 for the next iteration.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, in_mode, ADDR_MOD_3, in0_off);  // operand A (int32 -> sign-magn LREG0)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);  // operand B (fp32 scaler)
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80 (excess-128)
        }
        lltt::replay(REQUANT_REPLAY_SLOT, REQUANT_REPLAY_LEN);      // CAST + RNE MAD + pack
        TT_SFPSTORE(p_sfpu::LREG0, out_mode, ADDR_MOD_2, out_off);  // store + dst_reg += 2
    }
}

// Fold +128.0 into the fp32 zero-point in LREG2 so the per-iteration MAD yields v + 128 directly
inline void _int8_bias_zero_point_() { TTI_SFPADDI(INT8_OFFSET_128_IMM16, p_sfpu::LREG2, 0); }

// Clamp / convert the integer-valued excess-128 MAD result v + 128 to u in [0, 255].
inline void _excess_128_clamp_convert_() {
    // Values below -128 (i.e. v + 128 < 0) must saturate to -128. FP32_TO_UINT8 returns the
    // magnitude of a negative input instead of 0, so clamp these lanes to 0.0 first:
    // FP32_TO_UINT8(0.0) = 0, which is the excess-128 encoding of -128 (the minimum int8 value).
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_NEAREST,
        0 /*imm8*/,
        p_sfpu::LCONST_0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8);  // u = v + 128 in [0, 255], exact for integer-valued v
}

// Int8 output: low 8 bits of LREG0 hold the 2's complement int8 byte.
inline void _int8_pack_fixup_() {
    _excess_128_clamp_convert_();
    TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);  // b = u ^ 0x80
}

// Int32 output: LREG0 holds the 2's-complement int32 in [-128, 127], stored raw by the kernel.
inline void _int32_pack_fixup_() {
    _excess_128_clamp_convert_();
    TTI_SFPIADD(
        INT8_MINUS_128_IMM12, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
}

// UInt8 output: FP32_TO_UINT8 of the integer-valued v, padded with SFPNOPs to the int32 body's
// length because calculate_{quant,requant}_int32 replay one length for both output formats.
inline void _uint8_pack_padded_() {
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_NEAREST,
        0 /*imm8*/,
        p_sfpu::LCONST_0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_quant_int32_int8_pack(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Int8 output:
    // The RNE MAD + offset-128 pack body is recorded once into QUANT_REPLAY_SLOT and replayed,
    constexpr std::uint32_t dst_tile_size = 64;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_3, in0_off);  // operand A (fp32)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);  // operand B (fp32 scaler)
        lltt::replay(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN_INT8_OUT);               // RNE MAD + offset-128 pack
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_2, out_off);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool INT8_INPUT = false>
inline void calculate_requant_int32_int8_pack(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Int8 output:
    // The CAST + RNE MAD + offset-128 pack body is recorded once into REQUANT_REPLAY_SLOT and replayed.
    // The int8-input unbias (byte ^ 0x80) stays inline before the replay.
    constexpr std::uint32_t dst_tile_size = 64;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, in0_off);  // operand A (int32/byte)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);  // operand B (fp32 scaler)
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80
        }
        lltt::replay(REQUANT_REPLAY_SLOT, REQUANT_REPLAY_LEN_INT8_OUT);  // CAST + RNE MAD + offset-128 pack
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_2, out_off);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false, bool INT8_INPUT = false>
inline void calculate_dequant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A[LREG0] is input to dequant (int32, sign-magnitude or 2's complement bits or UInt8-unpacked int8 byte).
    // Operand B[LREG1] is scaling factor (fp32).
    // LREG2 holds the (negated) zero-point constant loaded by _init_dequant_int32_;
    // i.e. the formula computed is (A + LREG2) * B, which is (A - zero_point) * B
    // when the caller passes -zero_point through the init.
    //
    // SFPLOAD's instr_mod0 normalizes both int32 representations to LREG0
    // sign-magnitude before the recorded compute runs, so the body is the
    // same for both SIGN_MAGNITUDE_FORMAT variants.
    //
    // The replay-buffer body at DEQUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_dequant_int32_, which must run before the
    // first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr InstrModLoadStore in_mode =
        (SIGN_MAGNITUDE_FORMAT && !INT8_INPUT) ? InstrModLoadStore::INT32 : InstrModLoadStore::INT32_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: hoist both TT_SFPLOADs ahead of the recorded compute,
    // replay the body, then SFPSTORE under ADDR_MOD_2 which auto-advances
    // dst_reg by 2 for the next iteration.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, in_mode, ADDR_MOD_3, in0_off);  // operand A (int32 -> sign-magn LREG0)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);   // operand B (fp32 scaler)
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80
        }
        lltt::replay(DEQUANT_REPLAY_SLOT, DEQUANT_REPLAY_LEN);                     // CAST + ADD + SFPNOP + MUL + SFPNOP
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_2, out_off);  // store fp32 + dst_reg += 2
    }
}

// OUTPUT_FORMAT selects the quantized output tensor dtype: Int32 (the default), UInt8, or Int8. All
// three round to nearest even (see RNE_MAGIC_FP32). Int32 - int8-range values [-128, 127] held in an
// int32 container - and Int8 go through the excess-128 pack above so -128 is reachable; UInt8
// converts via FP32_TO_UINT8 into [0, 255].
template <
    bool APPROXIMATION_MODE /*unused*/,
    bool SIGN_MAGNITUDE_FORMAT /*unused*/ = false,
    DataFormat OUTPUT_FORMAT = DataFormat::Int32>
void quant_init(const uint zero_point) {
    static_assert(
        OUTPUT_FORMAT == DataFormat::Int32 || OUTPUT_FORMAT == DataFormat::UInt8 || OUTPUT_FORMAT == DataFormat::Int8,
        "quant_init OUTPUT_FORMAT must be Int32, UInt8 or Int8");
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
        _sfpu_load_imm32_(p_sfpu::LREG4, INT8_SIGN_MASK);
    }
    if constexpr (OUTPUT_FORMAT != DataFormat::UInt8) {
        _int8_bias_zero_point_();  // fold +128 into the fp32 zero-point in LREG2 (excess-128 pack)
    }
    _quant_rne_init_();  // LREG5 = MAGIC, LREG2 = zp (+128) - MAGIC
    _quant_kernels_configure_dest_incr_addrmod_();

    // The replay buffer feeds the SFPU pipe directly, so the recorded bodies use TTI_SFPNOP (SFPU NOP)
    // rather than the generic Tensix TTI_NOP for their pipeline bubbles.
    constexpr std::uint32_t REPLAY_LEN =
        OUTPUT_FORMAT == DataFormat::Int8 ? QUANT_REPLAY_LEN_INT8_OUT : QUANT_REPLAY_LEN;
    lltt::record<lltt::NoExec>(QUANT_REPLAY_SLOT, REPLAY_LEN);
    {
        _quant_rne_mad_();  // v = RNE(A * B) + zp (+128)
        if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
            _int8_pack_fixup_();
        } else if constexpr (OUTPUT_FORMAT == DataFormat::Int32) {
            _int32_pack_fixup_();
        } else {
            _uint8_pack_padded_();
        }
    }
}

template <
    bool APPROXIMATION_MODE /*unused*/,
    bool SIGN_MAGNITUDE_FORMAT /*unused*/ = false,
    DataFormat OUTPUT_FORMAT = DataFormat::Int32,
    bool INT8_INPUT = false>
void requant_init(const uint zero_point) {
    static_assert(
        OUTPUT_FORMAT == DataFormat::Int32 || OUTPUT_FORMAT == DataFormat::UInt8 || OUTPUT_FORMAT == DataFormat::Int8,
        "requant_init OUTPUT_FORMAT must be Int32, UInt8 or Int8");
    // The body rounds the whole expression q * (s_in / s_out) + zp to nearest even, zp being the
    // host-folded z_out - z_in * (s_in / s_out); this matches the op's golden
    // round((q - z_in) * (s_in / s_out) + z_out).
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    if constexpr (INT8_INPUT || OUTPUT_FORMAT == DataFormat::Int8) {
        _sfpu_load_imm32_(p_sfpu::LREG4, INT8_SIGN_MASK);
    }
    _requant_rne_init_<OUTPUT_FORMAT>();  // LREG5 = MAGIC, LREG6 = (128) - MAGIC
    _quant_kernels_configure_dest_incr_addrmod_();

    constexpr std::uint32_t REPLAY_LEN =
        OUTPUT_FORMAT == DataFormat::Int8 ? REQUANT_REPLAY_LEN_INT8_OUT : REQUANT_REPLAY_LEN;
    lltt::record<lltt::NoExec>(REQUANT_REPLAY_SLOT, REPLAY_LEN);
    {
        // int32 sign-magnitude (loaded that way regardless of input bit
        // representation, via SFPLOAD instr_mod0) -> fp32.
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
        _requant_rne_mad_();  // v = RNE(A * B + zp) (+128)
        if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
            _int8_pack_fixup_();
        } else if constexpr (OUTPUT_FORMAT == DataFormat::Int32) {
            _int32_pack_fixup_();
        } else {
            _uint8_pack_padded_();
        }
    }
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT /*unused*/ = false, bool INT8_INPUT = false>
void dequant_init(const uint zero_point) {
    // One-time setup for calculate_dequant; see quant_init for the
    // record/replay rationale. The caller passes -zero_point (so the
    // recorded body computes (A + LREG2) * B = (A - zero_point) * B).
    //
    // The recorded body is identical for both SIGN_MAGNITUDE_FORMAT variants;
    // the int32 input conversion happens in SFPLOAD's instr_mod0.
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    if constexpr (INT8_INPUT) {
        _sfpu_load_imm32_(p_sfpu::LREG4, INT8_SIGN_MASK);
    }
    _quant_kernels_configure_dest_incr_addrmod_();

    lltt::record<lltt::NoExec>(DEQUANT_REPLAY_SLOT, DEQUANT_REPLAY_LEN);
    {
        // int32 sign-magnitude -> fp32.
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
        // SFPADD = VA*VB + VC ; with LCONST_1 (LREG10) = 1.0 this collapses
        // to A + LREG2 (= A + zero_point as loaded by the caller).
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);
        // SFPADD has a 2-cycle write latency on LREG0; SFPMUL reads it next.
        TTI_SFPNOP;
        // SFPMUL with LCONST_0 (LREG9 = 0.0) ignored as +C :
        // LREG0 = (A + LREG2) * LREG1.
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /*mod1*/);
        // SFPMUL has a 2-cycle write latency on LREG0; the SFPSTORE that
        // follows the replay reads it. Keep this NOP inside the recorded
        // body rather than relying on the implicit gap between the replay
        // completing and TT_SFPSTORE issuing.
        TTI_SFPNOP;
    }
}

}  // namespace ckernel::sfpu
