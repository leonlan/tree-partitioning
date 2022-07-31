from pathlib import Path

from tree_partitioning.classes import Case, ReducedGraph

from tree_partitioning.algorithms.line_switching import milp_line_switching, brute_force
from tree_partitioning.algorithms.full import two_stage, recursive
from tree_partitioning.algorithms.partitioning.legacy import obi_main
from tree_partitioning.algorithms.partitioning import (
    normalized_laplacian,
    normalized_modularity,
    fastgreedy,
    modularity_score,
)

import cProfile
import pstats


def main():
    # PARAMS
    k = 5
    objective = "congestion"

    # case = Case.from_file(Path("data/pglib_opf_case2737sop_k.mat"), merge_lines=False)

    # case = Case.from_file(Path("data/pglib_opf_case793_goc.mat"), merge_lines=False)
    case = Case.from_file(Path("data/pglib_opf_case118_ieee.mat"), merge_lines=False)
    # case = Case.from_file(Path("data/pglib_opf_case300_ieee.mat"), merge_lines=False)
    # case = Case.from_file(Path("data/pglib_opf_case2000_goc.mat"), merge_lines=False)
    # for f in [normalized_laplacian, normalized_modularity, fastgreedy]:
    #     print(f.__name__, f(k).is_connected_clusters(case.G))

    # solution = two_stage(
    #     n_clusters=k,
    #     objective=objective,
    #     # partitioning_alg=normalized_laplacian,
    #     # partitioning_alg=normalized_modularity,
    #     partitioning_alg=fastgreedy,
    #     line_switching_alg=brute_force,
    #     results=True,
    # )

    # solution.plot(
    #     f"img/{case.name}_{normalized_laplacian.__name__}_{milp_line_switching.__name__}_{k}_{objective}.jpg"
    # )

    # print(modularity_score(partition))
    # print(modularity_score(partition, normalized=True))
    # print(solution.objective)
    # assert solution.is_tree_partition()

    recursive(
        n_clusters=k,
        objective=objective,
        # partitioning_alg=normalized_laplacian,
        # partitioning_alg=normalized_modularity,
        partitioning_alg=fastgreedy,
        line_switching_alg=brute_force,
        results=True,
    )


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="needs_profiling.prof")
