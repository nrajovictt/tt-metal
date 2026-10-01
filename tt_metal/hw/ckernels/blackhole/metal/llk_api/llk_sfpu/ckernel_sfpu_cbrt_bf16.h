// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct CbrtBf16Config {
    static constexpr uint32_t kKind = 2u;
    static constexpr uint32_t kMagic = 0x548c2b4bu;
    static constexpr uint32_t kC0Bits = 0x3fe04c03u;
    static constexpr uint32_t kC1Bits = 0xbfa01f36u;
    static constexpr uint32_t kC2Bits = 0x3f0266d9u;
    static constexpr uint32_t kBodySlots = 23u;
    static constexpr uint32_t kNegativeExponentZeroOutput = 0x7f80u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_newton_root.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_cbrt_bf16() {
    ckernel::sfpu::bf16::calculate_newton_root<ckernel::sfpu::CbrtBf16Config, ITERATIONS>();
}
inline void init_cbrt_bf16() { ckernel::sfpu::bf16::init_newton_root<ckernel::sfpu::CbrtBf16Config>(); }

}  // namespace ckernel::sfpu
