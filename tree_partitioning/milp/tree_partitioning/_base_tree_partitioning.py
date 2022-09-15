from .._base import _base
from .._grouping_constraint import _grouping_constraint
from ._reduced_graph_tree import _reduced_graph_tree
from ._single_commodity_flow import _single_commodity_flow


def _base_tree_partitioning(G, generators, **kwargs):
    """
    Base model for single-stage tree partitioning. The following conditions should hold:
    - Each generator block must belong to some cluster (grouping constraint)
    - The post-switching network must be connected (single commodity flow)
    - The reduced graph must be a tree (exactly k cross edges)
    """
    model = _base(G, len(generators))
    _grouping_constraint(model, generators)
    _single_commodity_flow(model)
    _reduced_graph_tree(model)

    return model
