import pyomo.environ as pyo

from .base_model import base_model


def transient_stability(case, generators, **kwargs):
    """
    Solve the tree partitioning problem for transient stability using the
    single-stage MILP approach.
    """
    m = base_model(case, generators)

    @m.Objective(sense=pyo.minimize)
    def objective(m):
        return sum(m.line_data[e]["f"] * (1 - m.active_line[e]) for e in m.lines)

    return m
