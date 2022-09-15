import pyomo.environ as pyo

from ._base_partitioning import _base_partitioning


def power_flow_disruption(G, generators, recursive=False, **kwargs):
    m = _base_partitioning(G, generators, recursive, **kwargs)

    @m.Objective(sense=pyo.minimize)
    def objective(m):
        return sum(
            data["f"] * (1 - m.active_line[line]) for line, data in m.line_data.items()
        )

    return m
