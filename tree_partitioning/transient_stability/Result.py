from dataclasses import dataclass


@dataclass
class Result:
    power_flow_disruption: float
    runtime: float
    n_switched_lines: int
    partition: dict
