from pathlib import Path

import networkx as nx
import igraph as ig

from tree_partitioning.classes import Case, Partition
from tree_partitioning.algorithms.line_switching import milp_ls, brute_force
from tree_partitioning.algorithms.partitioning import spectral_clustering


class TestLineSwitching:
    case = Case.from_file(Path("data/pglib_opf_case118_ieee.mat"), merge_lines=True)
    net = case.net
    netdict = case.netdict
    G = case.G
    igg = case.igg
    partition = spectral_clustering(igg, n_clusters=3)

    def test_milp(self):
        tree_partition = milp_ls(self.partition)
        assert tree_partition.is_tree_partition(G)

    def test_brute_force(self):
        tree_partition = brute_force(self.partition)
        assert tree_partition.is_tree_partition(G)
