from dataclasses import dataclass
from typing import List, Optional

import networkx as nx

import tree_partitioning.utils as utils
from tree_partitioning.classes import ReducedGraph
from tree_partitioning.dcpf import dcpf


# TODO make result for every algorithm
def make_result(case, generators, partition, lines, **kwargs):
    G = case.G
    weights = nx.get_edge_attributes(G, "weight")

    G_post = G.copy()
    G_post.remove_edges_from(lines)
    dcpf(G_post, in_place=True)

    rg = ReducedGraph(G, partition).RG.to_undirected()

    return Result(
        # Instance stats
        case=case.name,
        n_buses=len(G.nodes),
        n_lines=len(G.edges),
        n_clusters=len(generators),
        pre_congestion=utils.maximum_congestion(G),
        # Objective and runtimes
        power_flow_disruption=sum(weights[line] for line in lines),
        maximum_congestion=utils.maximum_congestion(G_post),
        # TP stats
        n_cross_edges=len(rg.edges()),
        n_switched_lines=len(rg.edges()) - (len(generators) - 1),
        cluster_sizes=[len(v) for v in partition.clusters.values()],
        generator_sizes=[len(v) for v in generators.values()],
        generators=dict(generators),
        partition=dict(partition.clusters),
        lines=lines,
        # # MIP gaps
        # mip_gap_single_stage=(res.problem.upper_bound - res.problem.lower_bound)
        # / res.problem.upper_bound,
        # mip_gap_partitioning=(res1.problem.upper_bound - res1.problem.lower_bound)
        # / res1.problem.upper_bound,
        # mip_gap_line_switching=(res2.problem.upper_bound - res2.problem.lower_bound)
        # / res2.problem.upper_bound,
        # runtime=..., # TODO must be specified by algorithm
        # algorithm=...,
        **kwargs
    )


@dataclass
class Result:
    case: str
    n_buses: int
    n_lines: int
    n_clusters: int
    pre_congestion: float
    algorithm: str
    power_flow_disruption: float
    maximum_congestion: float
    runtime: float
    n_cross_edges: int
    n_switched_lines: int
    cluster_sizes: List[int]
    generator_sizes: List[int]
    generators: dict
    partition: dict
    lines: List
    mip_gap_single_stage: Optional[float] = None
    mip_gap_partitioning_stage: Optional[float] = None
    mip_gap_line_switching_stage: Optional[float] = None

    def to_csv(self, path):
        data = [
            self.case,
            self.n_buses,
            self.n_lines,
            self.n_clusters,
            self.pre_congestion,
            self.algorithm,
            self.power_flow_disruption,
            self.maximum_congestion,
            self.runtime,
            self.n_cross_edges,
            self.n_switched_lines,
            self.cluster_sizes,
            self.generator_sizes,
            self.generators,
            self.partition,
            self.lines,
            self.mip_gap_single_stage,
            self.mip_gap_partitioning_stage,
            self.mip_gap_line_switching_stage,
        ]

        with open(path, "w") as fi:
            fi.write(";".join(str(x) for x in data))
