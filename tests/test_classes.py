#!/usr/bin/env ipython
from tree_partitioning.classes import Case

from pathlib import Path


class TestCaseNoMerge:
    case = Case(Path("data/pglib_opf_case300_ieee.mat"), merge_lines=False)
    net = case.net
    netdict = case.netdict
    G = case.G
    igg = case.igg

    def test_net_and_netdict(self):
        """
        Verify identical characteristics of net and netdict.
        """
        net = self.net
        netdict = self.netdict

        assert len(net.line) + len(net.trafo) == len(netdict["lines"])
        assert len(net.bus) == len(netdict["buses"])
        assert set(net.res_bus.p_mw) == set(
            data["p_mw"] for i, data in netdict["buses"].items()
        )

    def test_netdict_and_G(self):
        """
        Verify identical characteristics of netdict and nx.G.
        """
        netdict = self.netdict
        G = self.G

        assert len(netdict["lines"]) == len(G.edges)
        assert len(netdict["buses"]) == len(G.nodes)
        assert all(
            data["p_mw"] == G.nodes[bus]["p_mw"]
            for bus, data in netdict["buses"].items()
        )

    def test_netdict_and_igg(self):
        """
        Verify identical characteristics of netdict and ig.igg.
        """
        netdict = self.netdict
        igg = self.igg

        assert len(netdict["lines"]) == len(igg.es)
        assert len(netdict["buses"]) == len(igg.vs)
        assert all(
            data["p_mw"] == igg.vs[bus]["p_mw"]
            for bus, data in netdict["buses"].items()
        )


class TestCaseMerge:
    case = Case(Path("data/pglib_opf_case5_pjm.mat"), merge_lines=True)
