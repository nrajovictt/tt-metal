// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"
#include "internal/risc_attribs.h"

using ttnn::operations::matmul::ActivationThread;
using ttnn::operations::matmul::KernelActivation;

namespace compute_kernel_lib {

// These activations currently have math-only APIs. Other activations support pack.
template <KernelActivation Act>
inline constexpr ActivationThread activation_thread =
    Act == KernelActivation::MISH || Act == KernelActivation::SQRT || Act == KernelActivation::LEAKY_RELU ||
            Act == KernelActivation::ELU || Act == KernelActivation::EXP || Act == KernelActivation::RECIP
        ? ActivationThread::Math
        : ActivationThread::Pack;

// Initialize once at kernel startup on the selected thread.
template <
    KernelActivation Act,
    uint32_t Param0 = 0,
    uint32_t Param1 = 0,
    ActivationThread Thread = activation_thread<Act>>
struct ActivationInitHelper {
    FORCE_INLINE static void init();
};

// Apply activation to DST on the packer thread, replacing tile_regs_wait().
template <KernelActivation Act, uint32_t Param0 = 0, uint32_t Param1 = 0, uint32_t Param2 = 0>
FORCE_INLINE void apply_activation_from_pack(uint32_t out_subblock_num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_activation_helpers.inl"
