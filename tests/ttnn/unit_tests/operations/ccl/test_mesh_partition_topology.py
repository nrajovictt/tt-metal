# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TensorTopology labels produced by ttnn.mesh_partition.

Contract: after a partition along ``dim`` on ``cluster_axis`` the output is ``Shard(dim)`` on that mesh axis and
keeps the input's placement on every other axis. A whole-mesh partition (``cluster_axis=None``) linearises the
devices row-major, which only the collapsed 1-D label ``{N}, [Shard(dim)]`` can express. A collapsed replicated
input (``ReplicateTensorToMesh`` on a multi-axis mesh) partitioned along one axis is uncollapsed to the device mesh
shape so the label can say ``Shard(dim)`` on that axis and ``Replicate`` on the others.

Negative control: before ``MeshPartitionDeviceOperation::compute_output_topologies`` existed the framework's union
default copied the input label onto the output, so a replicated input stayed ``[Replicate]`` (the serialiser then
dedups shards on that axis and drops data) and a ``ShardTensor2dMesh`` input kept its pre-partition placements.
Every test here asserts the post-partition label, so each fails on the old code; ``_assert_label_changed`` makes the
"output label is not just the input label" check explicit.
"""

import pytest
import torch

import ttnn


def _placements_equal(actual, expected):
    # PlacementShard has no __eq__ binding, so compare structurally.
    if isinstance(expected, ttnn.PlacementShard):
        return isinstance(actual, ttnn.PlacementShard) and actual.dim == expected.dim
    return isinstance(expected, ttnn.PlacementReplicate) and isinstance(actual, ttnn.PlacementReplicate)


def _assert_topology(output, expected_shape, expected_placements, expected_coords):
    topology = output.tensor_topology()
    actual_shape = list(topology.distribution_shape())
    assert actual_shape == list(expected_shape), f"distribution shape {actual_shape} != {list(expected_shape)}"
    placements = list(topology.placements())
    assert len(placements) == len(expected_placements), f"{placements} vs {expected_placements}"
    for actual, expected in zip(placements, expected_placements):
        assert _placements_equal(actual, expected), f"{placements} != {expected_placements}"
    assert list(topology.mesh_coords()) == list(expected_coords), "mesh_coords must carry over from the input"


def _assert_label_changed(output, input_tensor):
    # Negative control: the union default (pre-change behaviour) would have produced exactly the input's label.
    assert output.tensor_topology() != input_tensor.tensor_topology()


def _assert_device_slices(output, input_tensor, dim, cluster_axis, mesh_shape):
    """Device (r, c) must hold chunk r (cluster_axis=0), c (cluster_axis=1) or r*cols+c (whole mesh) of its own
    input shard along ``dim``. Shards of both tensors are enumerated in the same row-major device order."""
    rows, cols = mesh_shape
    num_chunks = rows * cols if cluster_axis is None else mesh_shape[cluster_axis]
    output_shards = ttnn.get_device_tensors(output)
    input_shards = ttnn.get_device_tensors(input_tensor)
    coords = list(output.tensor_topology().mesh_coords())
    assert len(output_shards) == len(input_shards) == len(coords) == rows * cols
    for coord, out_shard, in_shard in zip(coords, output_shards, input_shards):
        r, c = coord[0], coord[1]
        index = r * cols + c if cluster_axis is None else (r if cluster_axis == 0 else c)
        expected = torch.chunk(ttnn.to_torch(in_shard), num_chunks, dim=dim)[index]
        assert torch.equal(ttnn.to_torch(out_shard), expected), f"data mismatch on device {coord}"


def _from_torch(torch_tensor, mesh_device, mesh_mapper):
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
@pytest.mark.parametrize(
    "input_dims, dim, cluster_axis, expected_placements",
    [
        # Rows shard dim 2, columns replicate; partition dim 3 across columns.
        ((2, None), 3, 1, [ttnn.PlacementShard(2), ttnn.PlacementShard(3)]),
        # Same, with a negative dim: the label must carry the normalised dim, not -1.
        ((2, None), -1, 1, [ttnn.PlacementShard(2), ttnn.PlacementShard(3)]),
        # Columns shard dim 3, rows replicate; partition dim 2 across rows.
        ((None, 3), 2, 0, [ttnn.PlacementShard(2), ttnn.PlacementShard(3)]),
        # Fully replicated N-D label; partition dim 1 across one axis, the other axis stays Replicate.
        ((None, None), 1, 1, [ttnn.PlacementReplicate(), ttnn.PlacementShard(1)]),
        ((None, None), 1, 0, [ttnn.PlacementShard(1), ttnn.PlacementReplicate()]),
    ],
)
def test_mesh_partition_2d_label(mesh_device, input_dims, dim, cluster_axis, expected_placements):
    mesh_shape = tuple(mesh_device.shape)
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _from_torch(
        torch_input, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=input_dims)
    )
    input_coords = list(tt_input.tensor_topology().mesh_coords())

    mesh_device.enable_program_cache()
    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)
    cache_entries = mesh_device.num_program_cache_entries()

    _assert_topology(tt_output, mesh_shape, expected_placements, input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, dim % torch_input.ndim, cluster_axis, mesh_shape)

    # The label is applied on the host launch path, so a program-cache hit must produce the same label.
    tt_output_cached = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)
    assert mesh_device.num_program_cache_entries() == cache_entries
    assert tt_output_cached.tensor_topology() == tt_output.tensor_topology()


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
@pytest.mark.parametrize("dim", [1, 3])
def test_mesh_partition_2d_input_whole_mesh_collapses(mesh_device, dim):
    """cluster_axis=None hands device r*cols+c chunk r*cols+c: an N-D label would need Shard(dim) on both axes,
    which the composer rejects, so the output carries the collapsed {N}, [Shard(dim)] label over the same
    row-major coordinates (identical to what ShardTensorToMesh(dim) would have produced)."""
    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _from_torch(
        torch_input, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(None, None))
    )
    input_coords = list(tt_input.tensor_topology().mesh_coords())

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=None)

    _assert_topology(tt_output, (num_devices,), [ttnn.PlacementShard(dim)], input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, dim, None, mesh_shape)

    # The label must round-trip through the label-driven composer as a plain concat along dim.
    composed = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
    assert torch.equal(composed, torch_input)


@pytest.mark.parametrize("mesh_device", [pytest.param(2, id="1x2")], indirect=True)
@pytest.mark.parametrize("dim", [1, 3, -1])
@pytest.mark.parametrize("cluster_axis", [1, None])
def test_mesh_partition_collapsed_replicate_on_ring(mesh_device, dim, cluster_axis):
    """ReplicateTensorToMesh gives the collapsed {2}, [Replicate] label. With cluster_axis=1 on a 1x2 mesh the
    collapsed axis is the partitioned axis, so the honest label is {2}, [Shard(dim)] over the same coordinates.
    Before the change the output kept [Replicate], which the serialiser dedups (dropping one device's slice)."""
    mesh_shape = tuple(mesh_device.shape)
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _from_torch(torch_input, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    input_coords = list(tt_input.tensor_topology().mesh_coords())
    assert list(tt_input.tensor_topology().distribution_shape()) == [2]
    assert _placements_equal(list(tt_input.tensor_topology().placements())[0], ttnn.PlacementReplicate())

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)

    normalised_dim = dim % torch_input.ndim
    _assert_topology(tt_output, (2,), [ttnn.PlacementShard(normalised_dim)], input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, normalised_dim, cluster_axis, mesh_shape)

    composed = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=normalised_dim))
    assert torch.equal(composed, torch_input)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_mesh_partition_collapsed_replicate_uncollapses_on_2d_mesh(mesh_device, dim, cluster_axis):
    """ReplicateTensorToMesh on a 2x4 mesh gives {8}, [Replicate]. Partitioning along one axis leaves the tensor
    replicated along the other, which only an N-D label over the device mesh can state; the collapsed row-major
    coordinates already enumerate that mesh, so they carry over. This is the deepseek_v3 mesh_partition path."""
    mesh_shape = tuple(mesh_device.shape)
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _from_torch(torch_input, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    input_coords = list(tt_input.tensor_topology().mesh_coords())
    assert list(tt_input.tensor_topology().distribution_shape()) == [mesh_shape[0] * mesh_shape[1]]

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)

    expected_placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
    expected_placements[cluster_axis] = ttnn.PlacementShard(dim)
    _assert_topology(tt_output, mesh_shape, expected_placements, input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, dim, cluster_axis, mesh_shape)

    # Concatenating along the partitioned axis reassembles the input on every line of the replicated axis.
    shards = {
        (coord[0], coord[1]): ttnn.to_torch(shard)
        for coord, shard in zip(tt_output.tensor_topology().mesh_coords(), ttnn.get_device_tensors(tt_output))
    }
    rows, cols = mesh_shape
    for other in range(cols if cluster_axis == 0 else rows):
        if cluster_axis == 0:
            line = [shards[(i, other)] for i in range(rows)]
        else:
            line = [shards[(other, j)] for j in range(cols)]
        assert torch.equal(torch.cat(line, dim=dim), torch_input)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
def test_mesh_partition_collapsed_replicate_whole_mesh(mesh_device):
    """Whole-mesh partition of a collapsed replicated input: {8}, [Replicate] -> {8}, [Shard(dim)]."""
    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    dim = 3
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _from_torch(torch_input, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    input_coords = list(tt_input.tensor_topology().mesh_coords())

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=None)

    _assert_topology(tt_output, (num_devices,), [ttnn.PlacementShard(dim)], input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, dim, None, mesh_shape)
    composed = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
    assert torch.equal(composed, torch_input)
