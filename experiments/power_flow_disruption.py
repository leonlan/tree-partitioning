from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from glob import glob
from time import perf_counter
from typing import List

import networkx as nx
import pyomo.environ as pyo

import tree_partitioning.milp.partitioning as partitioning
import tree_partitioning.milp.utils as model_utils
from tree_partitioning.classes import Case, Partition, ReducedGraph
from tree_partitioning.gci import mst_gci
from tree_partitioning.milp.line_switching import maximum_spanning_tree
from tree_partitioning.milp.tree_partitioning import power_flow_disruption
from tree_partitioning.utils import maximum_congestion, remove_lines


@dataclass
class Result:
    case: str
    n_clusters: int
    generator_sizes: List[int]
    power_flow_disruption: float
    runtime_total: float
    runtime_partitioning: float
    runtime_line_switching: float
    mip_gap_single_stage: float
    mip_gap_partitioning_stage: float
    mip_gap_line_switching_stage: float
    n_cross_edges: int
    n_switched_lines: int
    cluster_sizes: List[int]
    pre_max_congestion: float
    post_max_congestion: float
    algorithm: str

    def to_csv(self):
        data = [
            self.case,
            self.n_clusters,
            self.generator_sizes,
            self.power_flow_disruption,
            self.runtime_total,
            self.runtime_partitioning,
            self.runtime_line_switching,
            self.mip_gap_single_stage,
            self.mip_gap_partitioning_stage,
            self.mip_gap_line_switching_stage,
            self.n_cross_edges,
            self.n_switched_lines,
            self.cluster_sizes,
            self.pre_max_congestion,
            self.post_max_congestion,
            self.algorithm,
        ]

        return ";".join(str(x) for x in data) + "\n"


def single_stage(case, generators, **kwargs):
    """
    Solve TP-PFD considering transient stability.
    """
    model = power_flow_disruption(case.G, generators, **kwargs)

    solver, options = kwargs["solver"], kwargs["options"]
    solver.solve(model, tee=False, options=options)

    partition = model_utils.get_partition(model)
    inactive_lines = model_utils.get_inactive_lines(model)

    sanity_check(case.G, generators, partition, inactive_lines)
    print("single-stage", len(generators), model.objective())
    return 0, 0


def multi_stage(case, generators, **kwargs):
    """
    Solve the tree partitioning problem considering transient stability using
    the two-stage MILP+MST approach.
    """
    G = case.G
    groups = generators

    solver, options = kwargs["solver"], kwargs["options"]

    final_partition = [list(G.nodes)]
    final_lines = []
    cluster = final_partition[0]

    total = 0

    for _ in range(len(generators) - 1):
        model = partitioning.power_flow_disruption(
            G.subgraph(cluster), groups, recursive=True, **kwargs
        )
        solver.solve(model, tee=True, options=options)

        new_partition = model_utils.get_partition(model)
        assert new_partition.is_connected_clusters(G.subgraph(cluster))

        cost, lines = maximum_spanning_tree(G.subgraph(cluster), new_partition)
        final_lines += lines
        total += cost

        # Update the final partition
        final_partition = update(final_partition, new_partition)

        # Find the cluster with the most generator groups
        num_groups = lambda cl: sum(gens[0] in cl for idx, gens in generators.items())
        cluster = max([cl for cl in final_partition], key=num_groups)

        groups = {
            idx: gens
            for idx, gens in generators.items()
            if all(g in cluster for g in gens)
        }

    print("multi_stage", len(generators), total)
    sanity_check(
        G, generators, Partition(dict(enumerate(final_partition))), final_lines
    )

    return Result, 0


def update(partition: list, subpartition: Partition):
    # Update the final partition with the newly found partition
    nodes = [v for cluster in subpartition.clusters.values() for v in cluster]

    # Remove all clusters that have nodes in the subpartition
    partition = [cluster for cluster in partition if cluster[0] not in nodes]

    # Add the new subpartition clusters
    for cluster in subpartition.clusters.values():
        partition.append(cluster)

    return partition


def sanity_check(G, generators, partition, lines):
    """
    Check if all the following conditions are satisfied:
    - The partition has connected clusters;
    - Each generator group belongs to one cluster;
    - The post switching graph has correct number of edges/nodes;
    - The post switching reduced graph is a tree.
    """
    assert partition.is_connected_clusters(G)

    assert all(
        any(
            len(set(gens).intersection(set(cluster)))
            for cluster in partition.clusters.values()
        )
        for gens in generators.values()
    )

    G_post = remove_lines(G, lines)[0]  # First connected component
    assert len(G.edges) == len(G_post.edges) + len(lines)
    assert len(G.nodes) == len(G_post.nodes)

    assert ReducedGraph(G_post, partition).is_tree()


def two_stage(case, generators, **kwargs):
    """
    Solve the tree partitioning problem minimizing transient problem using the
    two-stage MILP+MST approach.
    """
    solver, options = kwargs["solver"], kwargs["options"]

    G = case.G

    # Partitioning stage
    start_partitioning = perf_counter()
    model1 = partitioning.power_flow_disruption(G, generators, **kwargs)
    solver.solve(model1, tee=False, options=options)

    time_partitioning = perf_counter() - start_partitioning

    partition = model_utils.get_partition(model1)

    # Line switching stage
    start_line_switching = perf_counter()

    rg = ReducedGraph(G, partition).RG.to_undirected()
    cost, lines = maximum_spanning_tree(G, partition)

    time_line_switching = perf_counter() - start_line_switching

    sanity_check(G, generators, partition, lines)
    print("two_stage", len(generators), cost)

    return (0, 0)
    # Result(
    #     case=case.name,
    #     n_clusters=len(generators),
    #     generator_sizes=[len(v) for v in generators.values()],
    #     power_flow_disruption=cost,
    #     runtime_total=time_partitioning + time_line_switching,
    #     runtime_partitioning=time_partitioning,
    #     runtime_line_switching=time_line_switching,
    #     mip_gap_single_stage=None,
    #     mip_gap_partitioning_stage=(
    #         res1.problem.upper_bound - res1.problem.lower_bound
    #     )
    #     / res1.problem.upper_bound,
    #     mip_gap_line_switching_stage=None,
    #     n_cross_edges=len(rg.edges()),
    #     n_switched_lines=len(rg.edges()) - (len(generators) - 1),
    #     cluster_sizes=[len(v) for v in partition.clusters.values()],
    #     algorithm=f"2-stage-pfd",
    # ),
    # G_post,


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_pattern", default="instances/pglib_opf_*.mat")
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--max_size", type=int, default=30)
    parser.add_argument("--min_size", type=int, default=30)
    parser.add_argument("--results_path", type=str, default="results.txt")

    return parser.parse_args()


def name2size(name: str) -> int:
    """
    Extracts the instance size (i.e., num clients) from the instance name.
    """
    return int(re.search(r"_case(\d+)", name).group(1))


def setup_config(args):
    solver = pyo.SolverFactory("gurobi", solver_io="python")
    options = {"TimeLimit": args.time_limit}

    config = {"solver": solver, "options": options}
    return config


def main():
    args = parse_args()
    instances = sorted(glob(args.instance_pattern), key=name2size)
    config = setup_config(args)

    with open(args.results_path, "w") as fi:
        for path in instances:

            if not (args.min_size <= name2size(path) <= args.max_size):
                continue

            case = Case.from_file(path, merge_lines=True)

            print(case.name)
            for k in range(2, args.n_clusters + 1):
                generator_groups = mst_gci(case, k)

                single_stage(case, generator_groups, **config)
                two_stage(case, generator_groups, **config)
                multi_stage(case, generator_groups, **config)

    # return Result(
    #     case=case.name,
    #     n_clusters=len(generators),
    #     generator_sizes=[len(v) for v in generators.values()],
    #     power_flow_disruption=model.objective(),
    #     runtime_total=end,
    #     runtime_line_switching=None,
    #     runtime_partitioning=None,
    #     mip_gap_single_stage=(result.problem.upper_bound - result.problem.lower_bound)
    #     / result.problem.upper_bound,
    #     mip_gap_partitioning_stage=None,
    #     mip_gap_line_switching_stage=None,
    #     n_cross_edges=len(model_utils.get_cross_edges(model)),
    #     n_switched_lines=len(model_utils.get_switched_lines(model)),
    #     cluster_sizes=model_utils.get_cluster_sizes(model),
    #     pre_max_congestion=maximum_congestion(G_pre),
    #     post_max_congestion=maximum_congestion(G_post),
    #     algorithm="single stage",
    # )


if __name__ == "__main__":
    main()
