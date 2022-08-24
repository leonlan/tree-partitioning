from __future__ import annotations

from typing import Callable

from tree_partitioning.classes import Partition, Solution


def two_stage(
    case,
    n_clusters: int,
    partitioning_alg: Callable[..., Partition],
    line_switching_alg: Callable[..., Solution],
):
    """
    Solve the tree partitioning problem with n_clusters and minimize objective.
    """
    partition = partitioning_alg(n_clusters)
    assert partition.is_connected_clusters(case.G)

    solution = line_switching_alg(partition, objective="congestion")
    return solution
