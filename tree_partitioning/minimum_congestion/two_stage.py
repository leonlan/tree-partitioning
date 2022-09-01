from __future__ import annotations

from typing import Callable

from tree_partitioning.classes import Partition, Solution
from tree_partitioning.dcpf import dcpf
from tree_partitioning.utils import maximum_congestion


def two_stage(
    case,
    n_clusters: int,
    partitioning_alg: Callable[..., Partition],
    line_switching_alg: Callable[..., Solution],
):
    """
    Solve the tree partitioning problem with n_clusters and minimize objective.
    """
    start = perf_counter()
    partition = partitioning_alg(n_clusters)
    assert partition.is_connected_clusters(case.G)
    end = perf_counter() - start

    solution = line_switching_alg(partition, objective="congestion")

    return Result(
        case=case.name,
        n_clusters=len(generators),
        generator_sizes=[len(v) for v in generators.values()],
        power_flow_disruption=cost,
        runtime=end,
        n_switched_lines=len(rg.edges()) - (len(generators) - 1),
        cluster_sizes=[len(v) for v in partition.clusters.values()],
        pre_max_congestion=maximum_congestion(G_pre),
        post_max_congestion=maximum_congestion(G_post),
        algorithm=f"2-stage-{tpi_objective}",
    )
