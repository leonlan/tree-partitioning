from __future__ import annotations

from typing import Callable

from tree_partitioning.classes import Partition, Solution

from .line_switching import brute_force, milp_line_switching


def recursive(
    case,
    n_clusters: int,
    partitioning_alg: Callable[..., Partition],
    line_switching_alg: Callable[..., Solution],
    results=True,
):
    """
    Solve the tree partitioning problem with n_clusters and minimize objective.
    """
    # TODO
    pass
