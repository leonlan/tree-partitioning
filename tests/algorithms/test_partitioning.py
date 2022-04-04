from pathlib import Path

from tree_partitioning.algorithms.partitioning import (
    fastgreedy,
    spectral_clustering,
    constrained_spectral_clustering,
    milp,
    obi_main,
)
from tree_partitioning.classes import Case


class TestPartitioning:
    case = Case.from_file(Path("data/pglib_opf_case793_goc.mat"), merge_lines=True)
    G = case.G
    igg = case.igg
    n_clusters = 5

    # def test_fastgreedy(self):
    #     partition = fastgreedy(self.igg, n_clusters=self.n_clusters)

    #     assert partition.is_partition(self.G)
    #     assert partition.is_connected_clusters(self.G)

    # def test_spectral_clustering(self):
    #     partition = spectral_clustering(self.igg, n_clusters=self.n_clusters)

    #     assert partition.is_partition(self.G)
    #     assert partition.is_connected_clusters(self.G)

    # def test_milp(self):
    #     partition = milp(self.igg)

    #     assert partition.is_partition(self.G)
    #     assert partition.is_connected_clusters(self.G)

    def test_legacy(self):
        partition = obi_main(self.case, self.n_clusters, method="LaplacianN")

        assert partition.is_partition(self.G)
        assert partition.is_connected_clusters(self.G)


class TestPartitioningConstrained:
    case = Case.from_file(Path("data/pglib_opf_case14_ieee.mat"), merge_lines=True)
    G = case.G
    igg = case.igg
    n_clusters = 3

    # def test_constrained_clustering(self):
    #     partition = constrained_spectral_clustering(self.igg, self.coherent_generators)

    #     assert partition.is_partition(self.G)
    #     assert partition.is_connected_clusters(self.G)

    # def test_milp_constrained(self):
    #     partition = milp(self.igg, self.coherent_generators)

    #     assert partition.is_partition(self.G)
    #     assert partition.is_connected_clusters(self.G)
