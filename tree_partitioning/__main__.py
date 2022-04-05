from pathlib import Path

from tree_partitioning.classes import Case
from tree_partitioning.algorithms.full.two_stage import two_stage


def main():
    # PARAMS
    k = 3
    objective = "congestion"
    partitioning_method = "LaplacianN"

    case = Case.from_file(Path("data/pglib_opf_case118_ieee.mat"), merge_lines=True)
    solution = two_stage(n_clusters=k, objective=objective, partitioning="ModularityN")
    solution.plot(f"img/{case.name}_{partitioning_method}_{objective}.jpg")

    assert solution.is_tree_partition()


if __name__ == "__main__":
    main()
