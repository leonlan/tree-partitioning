import pyomo.environ as pyo

from .._base import _base
from .._grouping_constraint import _grouping_constraint
from ._single_commodity_flow import _single_commodity_flow


def _base_partitioning(G, groups, recursive=False, **kwargs) -> pyo.ConcreteModel:
    """
    Base model for partitioning. The following conditions should hold:
    - Each generator group must belong to some cluster (grouping constraint)
    - Each cluster must be connected (single commodity flow)
    """
    n_clusters = len(groups) if not recursive else 2

    m = _base(G, n_clusters)
    _grouping_constraint(m, groups, recursive)
    _single_commodity_flow(m)

    return m
