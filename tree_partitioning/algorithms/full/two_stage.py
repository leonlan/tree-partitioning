from __future__ import annotations
from typing import Callable

from time import perf_counter
from tree_partitioning.classes import Case
from tree_partitioning.algorithms.partitioning import (
    spectral_clustering,
    fastgreedy,
    obi_main,
)
from tree_partitioning.algorithms.line_switching import milp_line_switching, brute_force


def two_stage(
    n_clusters: int,
    objective: str,
    partitioning_alg: Callable[..., Partition],
    line_switching_alg: Callable[..., Solution],
    results=False,
):
    """
    Solve the tree partitioning problem with n_clusters and minimize objective.
    """
    # Initialization
    case = Case()
    net, netdict, G, igg = case.all_objects

    runtime_partition = -perf_counter()
    partition = partitioning_alg(n_clusters)
    runtime_partition += perf_counter()

    assert partition.is_connected_clusters(G)

    runtime_line_switching = -perf_counter()
    solution = line_switching_alg(partition, objective)
    runtime_line_switching += perf_counter()

    print(runtime_partition, runtime_line_switching)

    if results:
        # Compute partitioning results
        method_partition = partitioning_alg.__name__
        runtime_partition = runtime_partition
        n_clusters = n_clusters
        bridge_block_sizes = ...
        n_cross_edges = ...
        fraction_cross_edges = ...
        n_lines_to_be_switched_off = ...

        # Compute line switching results
        method_line_switchiing = line_switching_alg.__name__
        runtime_line_switching = runtime_line_switching
        objective = solution.objective
        n_congested_lines = ...
        n_lines_to_be_switched_off = n_lines_to_be_switched_off
        fraction_lines_switched_off = ...
        power_flow_disruption = ...

        runtime_line_switching
    return solution
