// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

namespace compute_kernel_lib {

struct ActivationKernelConfig {
    bool pack_relu = false;
    ttnn::operations::matmul::KernelActivation activation = ttnn::operations::matmul::KernelActivation::NONE;
    ttnn::operations::matmul::ActivationThread thread = ttnn::operations::matmul::ActivationThread::Pack;
    uint32_t param0 = 0;
    std::map<std::string, std::string> defines;
};

// Only parameter-free operations with matching packer implementations run on
// pack. All other operations keep their math-thread unary defines.
inline ActivationKernelConfig get_activation_kernel_config(
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation) {
    using ttnn::operations::matmul::KernelActivation;
    using ttnn::operations::unary::UnaryOpType;

    if (!activation.has_value()) {
        return {};
    }
    if (activation->op_type == UnaryOpType::RELU) {
        return {.pack_relu = true};
    }

    std::optional<KernelActivation> packer_activation;
    if (activation->params.empty()) {
        switch (activation->op_type) {
            case UnaryOpType::SILU: packer_activation = KernelActivation::SILU; break;
            case UnaryOpType::HARDSIGMOID: packer_activation = KernelActivation::HARDSIGMOID; break;
            case UnaryOpType::GELU_TANH: packer_activation = KernelActivation::GELU_TANH; break;
            default: break;
        }
    }

    if (packer_activation.has_value()) {
        return {.activation = *packer_activation};
    }
    if (activation->op_type == UnaryOpType::GELU && activation->params.size() <= 1) {
        // The unary helper's parameterless GELU defaults to its fast variant.
        const float fast = activation->params.empty() ? 1.0f : activation->params[0];
        if (fast == 0.0f || fast == 1.0f) {
            return {
                .activation = KernelActivation::GELU,
                .thread = ttnn::operations::matmul::ActivationThread::Math,
                .param0 = static_cast<uint32_t>(fast)};
        }
    }
    return {
        .thread = ttnn::operations::matmul::ActivationThread::Math,
        .defines =
            ttnn::operations::unary::utils::get_defines(activation->op_type, activation->params, "ACTIVATION", "i")};
}

}  // namespace compute_kernel_lib
