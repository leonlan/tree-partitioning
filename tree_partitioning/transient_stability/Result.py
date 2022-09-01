from dataclasses import dataclass
from typing import List


@dataclass
class Result:
    case: str
    n_clusters: int
    generator_sizes: List[int]
    power_flow_disruption: float
    runtime_total: float
    runtime_line_switching: float
    runtime_partitioning: float
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
            self.runtime_line_switching,
            self.runtime_partitioning,
            self.n_switched_lines,
            self.cluster_sizes,
            self.pre_max_congestion,
            self.post_max_congestion,
            self.algorithm,
        ]

        return ";".join(str(x) for x in data) + "\n"
