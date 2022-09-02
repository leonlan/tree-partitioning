from collections import defaultdict
from time import perf_counter

import networkx as nx
import pyomo.environ as pyo

from tree_partitioning.classes import Case, Partition, ReducedGraph
from tree_partitioning.dcpf import dcpf
from tree_partitioning.line_switching import maximum_spanning_tree
from tree_partitioning.partitioning import milp_cluster, model2partition
from tree_partitioning.utils import maximum_congestion, remove_lines

from .Result import Result


def two_stage(case, generators, tpi_objective="power_flow_disruption", time_limit=300):
    """
    Solve the tree partitioning problem minimizing transient problem using the
    two-stage MILP+MST approach.
    """
    start_partitioning = perf_counter()
    model, result = milp_cluster(case, generators, tpi_objective, time_limit)
    partition = model2partition(model)
    time_partitioning = perf_counter() - start_partitioning
    rg = ReducedGraph(case.G, partition).RG.to_undirected()

    start_line_switching = perf_counter()
    cost, lines = maximum_spanning_tree(case.G, partition)
    time_line_switching = perf_counter() - start_line_switching

    G_pre = case.G
    G_post = dcpf(remove_lines(G_pre, lines)[0])[0]

    # Post-switching graph assertions
    assert len(G_pre.edges) == len(G_post.edges) + len(lines)
    assert nx.algorithms.components.is_weakly_connected(G_post)

    return Result(
        case=case.name,
        n_clusters=len(generators),
        generator_sizes=[len(v) for v in generators.values()],
        power_flow_disruption=cost,
        runtime_total=time_partitioning + time_line_switching,
        runtime_partitioning=time_partitioning,
        runtime_line_switching=time_line_switching,
        mip_gap_single_stage=None,
        mip_gap_partitioning_stage=(
            result.problem.upper_bound - result.problem.lower_bound
        )
        / result.problem.upper_bound,
        mip_gap_line_switching_stage=None,
        n_cross_edges=len(rg.edges()),
        n_switched_lines=len(rg.edges()) - (len(generators) - 1),
        cluster_sizes=[len(v) for v in partition.clusters.values()],
        pre_max_congestion=maximum_congestion(G_pre),
        post_max_congestion=maximum_congestion(G_post),
        algorithm=f"2-stage-{tpi_objective}",
    )
