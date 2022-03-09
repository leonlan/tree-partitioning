from pathlib import Path

import networkx as nx
import igraph as ig

from tree_partitioning.classes import Case, Partition


class TestTreePartition:
    case = Case.from_file(Path("data/pglib_opf_case118_ieee.mat"), merge_lines=True)
    net = case.net
    netdict = case.netdict
    G = case.G
    igg = case.igg

    def test_is_tree_partition(self):
        pass
