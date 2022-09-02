from collections import defaultdict
from time import perf_counter

import networkx as nx

from tree_partitioning.dcpf import dcpf
from tree_partitioning.single_stage import single_stage_transient_stability
from tree_partitioning.utils import maximum_congestion, remove_lines

from .Result import Result


def single_stage(case, generators, time_limit):

    start = perf_counter()
    model, result = single_stage_transient_stability(case, generators, time_limit)
    end = perf_counter() - start
    G_pre = case.G
    lines = get_switched_lines(model)
    G_post = dcpf(remove_lines(G_pre, lines)[0])[0]

    # Post-switching graph assertions
    assert len(G_pre.edges) == len(G_post.edges) + len(lines)
    assert nx.algorithms.components.is_weakly_connected(G_post)

    return Result(
        case=case.name,
        n_clusters=len(generators),
        generator_sizes=[len(v) for v in generators.values()],
        power_flow_disruption=model.objective(),
        runtime_total=end,
        runtime_line_switching=None,
        runtime_partitioning=None,
        mip_gap_single_stage=(result.problem.upper_bound - result.problem.lower_bound)
        / result.problem.upper_bound,
        mip_gap_partitioning_stage=None,
        mip_gap_line_switching_stage=None,
        n_cross_edges=len(get_cross_edges(model)),
        n_switched_lines=len(get_switched_lines(model)),
        cluster_sizes=get_cluster_sizes(model),
        pre_max_congestion=maximum_congestion(G_pre),
        post_max_congestion=maximum_congestion(G_post),
        algorithm="single stage",
    )


def get_cluster_sizes(model):
    cluster_sizes = defaultdict(int)

    for (_, cluster), val in model.assign_bus.items():
        if round(val()) == 1:
            cluster_sizes[cluster] += 1

    return list(cluster_sizes.values())


def get_switched_lines(model):
    lines = []

    for line, val in model.active_line.items():
        if round(val()) == 0:
            lines.append(line)

    return lines


def get_cross_edges(model):
    lines = defaultdict(int)

    for (*line, cluster), val in model.assign_line.items():
        lines[tuple(line)] += val()

    return [line for line, val in lines.items() if round(val) == 0]
