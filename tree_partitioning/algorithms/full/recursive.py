from __future__ import annotations

from typing import Callable

from time import perf_counter
from tree_partitioning.classes import Case, ReducedGraph
from tree_partitioning.algorithms.line_switching import milp_line_switching, brute_force
from ._legacy import recursivealgorithm


def recursive(
    n_clusters: int,
    objective: str,
    partitioning_alg: Callable[..., Partition],
    line_switching_alg: Callable[..., Solution],
    results=True,
):
    """
    Solve the tree partitioning problem with n_clusters and minimize objective.
    """
    # Initialization
    case = Case()
    net, netdict, G, igg = case.all_objects

    method = {
        "normalized_laplacian": "LaplacianN",
        "fastgreedy": "FastGreedy",
        "normalized_modularity": "ModularityN",
    }[partitioning_alg.__name__]

    try:
        res = recursivealgorithm(
            net, igg, n_clusters, method=method, steps=list(range(n_clusters))
        )

        if results:

            # all_results = [
            #     case.name,
            #     partitioning_alg.__name__,
            #     res["time_partitioning"],
            #     res["n_clusters"],
            #     res["block_sizes"][0][:5],
            #     res["cross_edges"],
            #     res["cross_edges"] / len(G.edges),
            #     res["#removed_lines"],
            #     res["modularity"],
            #     "brute_force",
            #     res["time_line_switching"],
            #     res["spanning_trees"],
            #     res["max_cong"][0],
            #     res["max_cong"][1],
            #     res["block_sizes"][-1][:5],
            #     res["#congested_lines"],
            #     res["#removed_lines"],
            #     res["#removed_lines"] / len(G.edges),
            #     "recursive",
            # # ]

            res.to_csv(
                "results_recursive_iterative.csv",
                ";",
                mode="a",
                index=False,
                header=False,
            )

        solution = True
    except:
        solution = False
    return solution
