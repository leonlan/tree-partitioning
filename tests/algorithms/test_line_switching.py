from pathlib import Path

import networkx as nx
import igraph as ig
import pytest

from tree_partitioning.classes import Case, Partition
from tree_partitioning.algorithms.line_switching import milp_line_switching, brute_force
from tree_partitioning.algorithms.partitioning import obi_main


class TestLineSwitching:
    case = Case.from_file(Path("data/pglib_opf_case300_ieee.mat"), merge_lines=False)
    net = case.net
    netdict = case.netdict
    G = case.G
    igg = case.igg
    partition = obi_main(k=2, method="LaplacianN")

    def test_milp(self):
        solution = milp_line_switching(self.partition)
        assert solution.is_tree_partition()

    def test_brute_force(self):
        solution = brute_force(self.partition)
        assert solution.is_tree_partition()

    def test_objective_milp_eq_brute_force(self):
        bf = brute_force(self.partition)
        milp = milp_line_switching(self.partition)

        assert bf.objective == pytest.approx(milp.objective)
