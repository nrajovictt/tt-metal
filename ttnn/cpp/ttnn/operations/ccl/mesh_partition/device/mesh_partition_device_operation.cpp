// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "mesh_partition_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::ccl {

namespace detail {
uint32_t get_cluster_axis_size(const ttnn::Tensor& input_tensor, const std::optional<uint32_t>& cluster_axis) {
    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    return cluster_axis.has_value() ? ((cluster_axis.value() == 0) ? mesh_view.num_rows() : mesh_view.num_cols())
                                    : mesh_view.num_devices();
}
}  // namespace detail

void MeshPartitionDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    uint32_t rank = input_tensor.logical_shape().rank();
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    if (tensor_args.optional_output_tensor.has_value()) {
        TT_FATAL(
            tensor_args.optional_output_tensor.value().tensor_spec() == output_spec,
            "Output tensor spec must match computed output spec");
    }
    const auto& input_shape = input_tensor.logical_shape();
    const auto& output_shape = output_spec.padded_shape();
    const auto& output_padded_shape = output_spec.padded_shape();

    TT_FATAL(
        !(operation_attributes.cluster_axis.has_value() && operation_attributes.cluster_axis.value() > 1),
        "Only support cluster axis of None, 0 or 1");

    TT_FATAL(operation_attributes.dim < rank, "dim must be less than the rank of the input tensor");

    const uint32_t cluster_axis_size = detail::get_cluster_axis_size(input_tensor, operation_attributes.cluster_axis);

    TT_FATAL(
        cluster_axis_size > 1,
        "Partition has only been tested with mesh axis size > 1, but has {} devices",
        cluster_axis_size);
    TT_FATAL(
        input_shape[operation_attributes.dim] % cluster_axis_size == 0,
        "input shape {} must be divisible by cluster axis size {}",
        input_tensor.logical_shape(),
        cluster_axis_size);

    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        TT_FATAL(
            output_shape[operation_attributes.dim] == output_padded_shape[operation_attributes.dim],
            "for tiled inputs, partitioning along dim {} should not create padding in output shape {}",
            operation_attributes.dim,
            output_shape);
    }
}

void MeshPartitionDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

MeshPartitionDeviceOperation::spec_return_value_t MeshPartitionDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto output_shape = input_tensor.logical_shape();

    const uint32_t cluster_axis_size = detail::get_cluster_axis_size(input_tensor, operation_attributes.cluster_axis);

    output_shape[operation_attributes.dim] = output_shape[operation_attributes.dim] / cluster_axis_size;
    return {tt::tt_metal::TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(),
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config))};
}

MeshPartitionDeviceOperation::tensor_return_value_t MeshPartitionDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }

    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
    return tensor;
}

std::vector<tt::tt_metal::TensorTopology> MeshPartitionDeviceOperation::compute_output_topologies(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;

    // Every device on the partitioned axis ends up with a distinct slice of `dim`, so the output is Shard{dim} on
    // that axis and unchanged elsewhere (the same contract reduce_scatter_minimal_async labels its output with).
    // `dim` is already normalised to [0, rank) by ttnn::prim::mesh_partition. This runs before validation, so it
    // must not dereference the optional cluster_axis unchecked.
    const auto& input_topology = tensor_args.input_tensor.tensor_topology();
    const auto& input_placements = input_topology.placements();
    const Shard shard_placement{static_cast<int>(operation_attributes.dim)};

    if (!operation_attributes.cluster_axis.has_value()) {
        // Whole-mesh partition linearises the devices row-major and hands each a distinct slice of `dim`. An N-D
        // label would need Shard{dim} on every axis, which the composer rejects (dims must be unique), so the honest
        // label is the collapsed 1-D one over the same row-major coordinates, matching ShardTensorToMesh.
        const auto num_shards = static_cast<uint32_t>(input_topology.distribution_shape().mesh_size());
        return {tt::tt_metal::TensorTopology(
            tt::tt_metal::distributed::MeshShape(num_shards), {shard_placement}, input_topology.mesh_coords())};
    }

    const size_t cluster_axis = operation_attributes.cluster_axis.value();
    if (input_placements.size() > 1) {
        if (cluster_axis >= input_placements.size()) {
            return {};
        }
        auto output_placements = input_placements;
        output_placements[cluster_axis] = shard_placement;
        return {tt::tt_metal::TensorTopology(
            input_topology.distribution_shape(), std::move(output_placements), input_topology.mesh_coords())};
    }

    // Collapsed 1-D label ({N},[placement]) from the default mappers. When the collapsed axis is the partitioned
    // axis (e.g. cluster_axis=1 on a 1xN mesh) every device in the label holds a distinct slice, so [Shard{dim}]
    // over the same coordinates is the honest label rather than leaving the input's Replicate in place.
    const auto num_label_devices = input_topology.distribution_shape().mesh_size();
    const uint32_t cluster_axis_size =
        detail::get_cluster_axis_size(tensor_args.input_tensor, operation_attributes.cluster_axis);
    if (num_label_devices == cluster_axis_size) {
        return {tt::tt_metal::TensorTopology(
            input_topology.distribution_shape(), {shard_placement}, input_topology.mesh_coords())};
    }

    // Replicated over the whole mesh but partitioned along one of several axes: only an N-D label over the device
    // mesh can say "Shard{dim} here, Replicate there". The collapsed coordinates already enumerate that mesh
    // row-major, so they carry over unchanged.
    const auto& mesh_shape = tensor_args.input_tensor.device()->get_view().shape();
    if (num_label_devices == mesh_shape.mesh_size() && cluster_axis < mesh_shape.dims() &&
        std::holds_alternative<Replicate>(input_placements[0])) {
        ttsl::SmallVector<Placement> output_placements(mesh_shape.dims(), Replicate{});
        output_placements[cluster_axis] = shard_placement;
        return {tt::tt_metal::TensorTopology(mesh_shape, std::move(output_placements), input_topology.mesh_coords())};
    }

    // A collapsed Shard label partitioned along one axis of a multi-axis mesh has no exact expression in either
    // form; fall back to the union default (input label) until the shared ccl topology helper lands.
    return {};
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor mesh_partition(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    using OperationType = ttnn::operations::ccl::MeshPartitionDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dim = (dim < 0 ? uint32_t(input_tensor.logical_shape().rank() + dim) : (uint32_t)dim),
            .cluster_axis = cluster_axis,
            .output_mem_config = memory_config,
        },
        OperationType::tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = optional_output_tensor});
}
}  // namespace ttnn::prim
