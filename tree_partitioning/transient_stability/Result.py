from dataclasses import dataclass
from typing import List


@dataclass
class Result:
    case: str
    n_clusters: int
    generator_sizes: List[int]
    power_flow_disruption: float
    runtime: float
    n_switched_lines: int
    cluster_sizes: List[int]
    algorithm: str

    def to_csv(self):
        data = [
            self.case,
            self.n_clusters,
            self.generator_sizes,
            self.power_flow_disruption,
            self.runtime,
            self.n_switched_lines,
            self.cluster_sizes,
            self.algorithm,
        ]

        return ";".join(str(x) for x in data) + "\n"
