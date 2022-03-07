from pathlib import Path

from tree_partitioning.algorithms.partitioning import fastgreedy, spectral_clustering
from tree_partitioning.classes import Case


class TestPartitioning:
    case = Case.from_file(Path("data/pglib_opf_case179_goc.mat"), merge_lines=True)
    G = case.G
    igg = case.igg

    def test_fastgreedy(self):
        partition = fastgreedy(self.igg, n_clusters=4)

        assert partition.is_partition(self.G)
        assert partition.is_connected_clusters(self.G)

    def test_spectral_clustering(self):
        partition = fastgreedy(self.igg, n_clusters=5)

        assert partition.is_partition(self.G)
        assert partition.is_connected_clusters(self.G)
