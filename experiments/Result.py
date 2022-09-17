# TODO make result for every algorithm
def make_result(case, generators, partition, lines, **kwargs):
    G = case.G

    Result(
        case=case.name,
        n_buses=len(G.nodes),
        n_lines=len(G.edges),
        n_clusters=len(G.generators),
        power_flow_disruption=...,
        runtime_total=...,
        runtime_partitioning=...,
        runtime_line_switching=...,
        mip_gap_single_stage=...,
        mip_gap_partitioning_stage=(res1.problem.upper_bound - res1.problem.lower_bound)
        / res1.problem.upper_bound,
        mip_gap_line_switching_stage=None,
        n_cross_edges=len(rg.edges()),
        n_switched_lines=len(rg.edges()) - (len(generators) - 1),
        cluster_sizes=[len(v) for v in partition.clusters.values()],
        algorithm=f"2-stage",
    )


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
