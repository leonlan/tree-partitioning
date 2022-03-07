from pathlib import Path

import networkx as nx

from tree_partitioning.classes import Case, Partition


class TestPartition:
    partition = Partition({0: [0, 1, 2,], 1: [3, 4], 2: [5, 6, 7, 8, 9]})
    G = nx.complete_graph(10)

    def test_clusters_to_buses(self):
        membership = self.partition.membership
        assert all(
            bus in self.partition[cluster] for bus, cluster in membership.items()
        )

    def test_len(self):
        assert len(self.partition) == 3

    def test_is_partition(self):
        assert self.partition.is_partition(self.G)

    # def test_is_bbd(self):
    #     assert not self.partitiong.is_bbd(self.G)


class TestPartitionFromCase:
    case = Case.from_file(Path("data/pglib_opf_case118_ieee.mat"), merge_lines=True)
    net = case.net
    netdict = case.netdict
    G = case.G
    igg = case.igg

    def test_is_partition(self):
        """
        Checks if the made up partitions are indeed partitions.
        """
        partition = Partition({0: [bus for bus in self.netdict["buses"].keys()]})
        assert partition.is_partition(self.G)


class TestBBD:
    ...
