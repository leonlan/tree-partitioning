import pyomo.environ as pyo

from ._base_tree_partitioning import _base_tree_partitioning


def power_flow_disruption(G, generators, **kwargs):
    """
    Solve the tree partitioning problem for transient stability using the
    single-stage MILP approach.
    """
    m = _base_tree_partitioning(G, generators, **kwargs)

    @m.Objective(sense=pyo.minimize)
    def objective(m):
        return sum(m.line_data[e]["f"] * (1 - m.active_line[e]) for e in m.lines)

    return m
