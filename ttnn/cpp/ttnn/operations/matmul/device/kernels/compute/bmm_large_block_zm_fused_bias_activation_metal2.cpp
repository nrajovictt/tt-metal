// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blocked matmul with optional transpose, bias, activation, and untilize.

#include <cstdint>
#include "experimental/kernel_args.h"

#include "ttnn/cpp/ttnn/kernel_lib/matmul/matmul.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul/reblock_untilize_helpers.hpp"

#include "api/compute/pack.h"
#include "api/compute/transpose.h"
#include "api/compute/compute_kernel_hw_startup.h"

namespace {

// Transpose input tiles in DST-sized chunks, including a final partial chunk.
template <uint32_t in_block_num_tiles, uint32_t block_size = 4>
FORCE_INLINE void transpose_tile_block(DataflowBuffer& in_transpose_buf, DataflowBuffer& in_buf) {
    constexpr uint32_t num_blocks = in_block_num_tiles / block_size;
    constexpr uint32_t last_block_size = in_block_num_tiles % block_size;

    const uint32_t in_transpose_cb_id = in_transpose_buf.get_id();
    const uint32_t in_cb_id = in_buf.get_id();

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        in_transpose_buf.wait_front(block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            transpose_tile(in_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in_transpose_buf.pop_front(block_size);

        in_buf.reserve_back(block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            pack_tile(tile_idx, in_cb_id);
        }
        tile_regs_release();
        in_buf.push_back(block_size);
    }

    if constexpr (last_block_size > 0) {
        in_transpose_buf.wait_front(last_block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            transpose_tile(in_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in_transpose_buf.pop_front(last_block_size);

        in_buf.reserve_back(last_block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            pack_tile(tile_idx, in_cb_id);
        }
        tile_regs_release();
        in_buf.push_back(last_block_size);
    }
}

}  // namespace

// Please update
// tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp
// when making any changes to this file.
// Have to keep a copy because cannot import ttnn into tests/tt_metal.
// With FUSE_BIAS: row_broadcast_bias (row-broadcast vs elementwise add_tiles) is compile-time arg 18 here;
// the perf copy uses index 14 (different compile-time arg layout).

void kernel_main() {
    using namespace compute_kernel_lib;

#ifdef MATMUL_DRAM_SHARDED
    if (get_arg(args::is_worker_core) != 1) {
        return;
    }
#endif

    // Compile-time arguments
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t in0_num_subblocks = get_arg(args::in0_num_subblocks);
    constexpr uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr uint32_t in1_num_subblocks = get_arg(args::in1_num_subblocks);
    constexpr uint32_t in1_block_w = get_arg(args::in1_block_w);
    constexpr uint32_t num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr uint32_t num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr uint32_t num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    constexpr uint32_t out_subblock_h = get_arg(args::out_subblock_h);
    constexpr uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    constexpr uint32_t batch = get_arg(args::batch);
    constexpr bool untilize_out = get_arg(args::untilize_out);
    constexpr bool get_batch_from_reader = (bool)get_arg(args::get_batch_from_reader);
#ifdef IN0_TRANSPOSE_TILE
    constexpr bool in0_transpose_tile = true;
#else
    constexpr bool in0_transpose_tile = false;
#endif

    // Circular buffer IDs
#ifdef IN0_TRANSPOSE_TILE
    constexpr uint32_t in0_cb_id = dfb::in0_transposed;
#else
    constexpr uint32_t in0_cb_id = dfb::in0;
#endif
    constexpr uint32_t in1_cb_id = dfb::in1;
    constexpr uint32_t out_cb_id = dfb::out;
    constexpr uint32_t mm_partials_cb_id = dfb::intermed0;
    constexpr uint32_t in0_transpose_cb_id = dfb::in0;

#ifdef FUSE_BIAS
    constexpr bool with_bias = true;
    constexpr uint32_t bias_cb_id = dfb::bias;
    constexpr uint32_t bias_ntiles = get_arg(args::bias_ntiles);
    // true: row-0 broadcast ([N] / [...,1,N]); false: elementwise add_tiles (bias has multiple M rows).
    constexpr bool row_broadcast_bias = (bool)get_arg(args::row_broadcast_bias);
    constexpr MatmulBias bias_config{bias_cb_id, bias_ntiles};
#ifdef BIAS_FULL_BLOCK
    static_assert(!row_broadcast_bias, "BIAS_FULL_BLOCK requires elementwise bias");
    constexpr MatmulBiasMode bias_mode = MatmulBiasMode::FullBlockElementwise;
#else
    constexpr MatmulBiasMode bias_mode =
        row_broadcast_bias ? MatmulBiasMode::RowBroadcast : MatmulBiasMode::ColumnIndexed;
#endif
#else
    constexpr bool with_bias = false;
    constexpr MatmulBias bias_config{};
    constexpr MatmulBiasMode bias_mode = MatmulBiasMode::RowBroadcast;
#endif

    // Default activation parameters keep helper instantiations valid when activation is disabled.
#ifdef SFPU_ACTIVATION
    constexpr KernelActivation activation_type = static_cast<KernelActivation>(get_arg(args::activation_type));
    constexpr uint32_t activation_param0 = get_arg(args::activation_param0);
    constexpr uint32_t activation_param1 = get_arg(args::activation_param1);
    constexpr uint32_t activation_param2 = get_arg(args::activation_param2);
#else
    constexpr KernelActivation activation_type = KernelActivation::NONE;
    constexpr uint32_t activation_param0 = 0;
    constexpr uint32_t activation_param1 = 0;
    constexpr uint32_t activation_param2 = 0;
#endif

#ifdef MM_PARTIALS_RELOAD_ALIAS
    constexpr uint32_t partials_reload_cb_id = dfb::intermed0_reload_alias;
#else
    constexpr uint32_t partials_reload_cb_id = UINT32_MAX;
#endif
#ifdef MATMUL_DRAM_SHARDED
    constexpr uint32_t last_subblock_w_valid = get_arg(args::last_subblock_w_valid);
#else
    constexpr uint32_t last_subblock_w_valid = 0;
#endif

    // Feature flags
#ifdef IN1_TRANSPOSE_TILE
    constexpr bool in1_transpose_tile = true;
#else
    constexpr bool in1_transpose_tile = false;
#endif

    constexpr bool l1_acc =
#ifdef PACKER_L1_ACC
        true;
#else
        false;
#endif

#ifdef PACK_RELU
    constexpr bool matmul_pack_relu = true;
#else
    constexpr bool matmul_pack_relu = false;
#endif

    // Activate the completed tiled result before untilize, which only reblocks and copies tiles.
    using BmmActivation =
        MatmulActivation<activation_type, activation_param0, activation_param1, activation_param2, matmul_pack_relu>;

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;
    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;
    constexpr bool multiple_output_blocks = batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1;
    using Shape = StaticMatmulShape<
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_h,
        out_subblock_w,
        in0_block_w,
        num_blocks_inner_dim,
        1,
        last_subblock_w_valid,
        in1_block_w>;
    constexpr Shape shape{
        /*partials_reload_cb_id=*/partials_reload_cb_id,
        /*partials_alias_output_cb_id=*/out_cb_id};

    DataflowBuffer in0_buf(in0_cb_id), in0_transpose_buf(in0_transpose_cb_id);
#ifdef FUSE_BIAS
    // Construction can synchronize on Quasar; do not construct an unused CB.
    DataflowBuffer bias_buf(bias_cb_id);
#endif
    auto prepare_k_block = [&](uint32_t, uint32_t, bool) {
        if constexpr (in0_transpose_tile) {
            reconfig_data_format_srca(in1_cb_id, in0_transpose_cb_id);
            transpose_init(in0_transpose_cb_id);
            PACK((pack_reconfig_data_format(in0_cb_id)));
#ifdef ARCH_QUASAR
            // Quasar output-buffer switches require updating the pack descriptor.
            pack_init(in0_cb_id);
#endif
            if constexpr (l1_acc) {
                PACK((llk_pack_reconfig_l1_acc(0)));
            }
            transpose_tile_block<in0_block_num_tiles>(in0_transpose_buf, in0_buf);
            reconfig_data_format_srca(in0_transpose_cb_id, in1_cb_id);
            matmul_block_init(in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
            PACK((pack_reconfig_data_format(mm_partials_cb_id)));
#ifdef ARCH_QUASAR
            pack_init(mm_partials_cb_id);
#endif
        }
    };

    // Retain matmul state across calls for heterogeneous-tile DRAM-sharded inputs.
    // Initialize SFPU activation once at startup.
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, mm_partials_cb_id);
    matmul_block_init(in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    BmmActivation::init();

    // Main loop: batch × output blocks
    for (uint32_t b = 0; b < batch; b++) {
        if constexpr (get_batch_from_reader) {
#ifdef ARCH_QUASAR
            static_assert(
                !get_batch_from_reader,
                "get_batch_from_reader (batch sparsity) is unsupported on Quasar until the DM->TRISC "
                "is_batch_valid mailbox handoff is implemented; do not enable it on Quasar.");
            const bool is_batch_valid = true;
#else
            bool is_batch_valid = false;
            UNPACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            MATH(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            PACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
#endif
            if (!is_batch_valid) {
                continue;
            }
        }

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                // Reset packer state for this output block
                if constexpr (multiple_output_blocks) {
                    PACK((pack_reconfig_data_format(mm_partials_cb_id)));
#ifdef ARCH_QUASAR
                    // Restore the partials descriptor after the previous output block.
                    pack_init(mm_partials_cb_id);
#endif
                }

                // K-loop matmul, bias, and activation share one library call.
                const auto result = compute_kernel_lib::matmul<
                    in1_transpose_tile,
                    l1_acc,
                    matmul_config::InitMode::AssumeInitialized,
                    matmul_config::InputPolicy::WaitAndPopPerKBlock,
                    matmul_config::DataFormatReconfig::None,
                    BmmActivation,
                    with_bias,
                    bias_mode>(
                    in0_cb_id,
                    in1_cb_id,
                    untilize_mode_out_cb_id,
                    mm_partials_cb_id,
                    shape,
                    prepare_k_block,
                    bias_config);

#ifdef FUSE_BIAS
                if constexpr (num_blocks_w_dim > 1) {
                    bias_buf.pop_front(bias_ntiles);
                }
#endif

                // Untilize the completed output when requested.
                if constexpr (untilize_out) {
                    result.prepare_untilize_input();
#if defined ARCH_QUASAR && (defined FP32_DEST_ACC_EN || defined PACKER_L1_ACC)
                    // WH/BH pack_untilize_dest_init configures this itself. Quasar
                    // programs the output descriptor but still needs the gasket format.
                    if constexpr (!with_bias) {
                        PACK((pack_reconfig_data_format(out_cb_id)));
                    }
#endif

                    reblock_and_untilize<
                        out_subblock_w,
                        out_block_w,
                        /*reconfigure=*/false>(in0_num_subblocks, out_subblock_h, mm_partials_cb_id, out_cb_id);
                }

                // Reconfigure for next output block
                if constexpr (multiple_output_blocks) {
                    result.restore_input_formats();
                    matmul_block_init(
                        in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        }
    }
#ifdef FUSE_BIAS
    // Width-block bias is streamed; a single width block keeps it resident.
    if constexpr (num_blocks_w_dim == 1) {
        bias_buf.pop_front(bias_ntiles);
    }
#endif
}
