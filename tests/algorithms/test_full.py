from pathlib import Path

import networkx as nx
import igraph as ig

from tree_partitioning.classes import Case, Partition
from tree_partitioning.algorithms.line_switching import milp_line_switching
from tree_partitioning.algorithms.partitioning import obi_main
from tree_partitioning.algorithms.full import two_stage


class TestFull:
    case = Case.from_file(Path("data/pglib_opf_case118_ieee.mat"), merge_lines=False)
    net, netdict, G, igg = case.all_objects

    partition = obi_main(case, k=4, method="LaplacianN")

    def test_two_stage(self):
        pass
