from tree_partitioning.classes import Case
from tree_partitioning.algorithms.partitioning import (
    spectral_clustering,
    fastgreedy,
    obi_main,
)
from tree_partitioning.algorithms.line_switching import milp_line_switching, brute_force

# from ._utils import _partitioning_alg_selection, _line_switching_alg_selection


def two_stage(
    n_clusters: int = 2,
    objective: str = "congestion",
    partitioning: str = "spectral_clustering",
    line_switching: str = "milp_line_switching",
):
    """
    Solve the tree partitioning problem with n_clusters and minimize objective.
    """
    # Initialization
    case = Case()
    net, netdict, G, igg = case.all_objects

    # Algorithm selection
    # partitioning_alg = _partitioning_alg_selection(partitioning)
    # line_switching_alg = _line_switching_alg_selection(line_switching)
    # partition = partitioning_alg(igg_subgraph, n_clusters=n_clusters).extend(G)

    partition = obi_main(case, n_clusters, method="FastGreedy")
    solution = milp_line_switching(partition, objective=objective)

    return solution
