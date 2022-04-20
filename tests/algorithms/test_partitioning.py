from pathlib import Path

from tree_partitioning.algorithms.partitioning import (
    fastgreedy,
    spectral_clustering,
    constrained_spectral_clustering,
    normalized_laplacian,
    normalized_modularity,
)
from tree_partitioning.classes import Case


# class TestPartitioning:
#     case = Case.from_file(Path("data/pglib_opf_case793_goc.mat"), merge_lines=True)
#     G = case.G
#     igg = case.igg
#     n_clusters = 5


def test_normalized_laplacian(small_cases):
    for _case in small_cases:
        case = _case()
        partition = normalized_laplacian(4)
        assert partition.is_connected_clusters(case.G)


def test_normalized_modularity(small_cases):
    for _case in small_cases:
        case = _case()
        partition = normalized_modularity(4)
        assert partition.is_connected_clusters(case.G)
