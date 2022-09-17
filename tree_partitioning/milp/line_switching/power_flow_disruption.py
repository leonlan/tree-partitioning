import pyomo.environ as pyo

from ._base_line_switching import _base_line_switching


def power_flow_disruption(G, partition, **kwargs):
    m = _base_line_switching(G, partition, **kwargs)

    @m.Objective(sense=pyo.minimize)
    def objective(m):
        return sum(m.line_data[tuple(line)]["weight"] for (u, v, *line) in m.cross)

    return m
