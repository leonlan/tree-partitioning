#!/usr/bin/env ipython
from tree_partitioning.classes import Case


class TestCaseNoMerge:
    case = Case("../../data/pglib_opf_case5_pjm.mat/", merge_lines=False)

    def test_net_and_netdict(self):
        """
        Verify identical characteristics of net and netdict.
        """
        net = self.case.net
        netdict = self.case.netdict

        assert len(net.line) + len(net.trafo) == len(netdict["edges"])
