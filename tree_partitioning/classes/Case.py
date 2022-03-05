import itertools
import os
from pathlib import Path

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp

from .Singleton import Singleton
from ._pp_utils import _load_pp_case, _netdict_from_pp_net
from ._nx_utils import _G_from_netdict
from ._ig_utils import _igg_from_netdict


class Case(metaclass=Singleton):
    """
    Case instance of a single power network.

    This class contains multiple representation of the network, among others:
    - a pandapower power network (net)
    - a generic dictionary representation of the network buses and lines (netdict)
    - a networkx graph (G)
    - an igraph graph (igg)

    These multiple representations are helpful as some computations are implemented
    for e.g. pandapower only.

    ---
    Params
    """

    _name: str
    _net: ...
    _netdict: dict
    _G: nx.MultiDiGraph
    _igg: ig.Graph

    @property
    def name(self) -> str:
        return self._name

    @property
    def net(self):
        return self._net

    @property
    def netdict(self):
        return self._netdict

    @property
    def G(self):
        return self._G

    @property
    def igg(self):
        return self._igg

    @classmethod
    def from_file(cls, path, merge_lines=True, opf_init=True, ac=False):
        cls.clear()

        case = cls()
        case._name = str(path).split("pglib_opf_")[-1].split(".mat")[0]
        case._net = _load_pp_case(path, opf_init, ac)
        case._netdict = _netdict_from_pp_net(case._net, merge_lines)
        case._G = _G_from_netdict(case._netdict)
        case._igg = _igg_from_netdict(case._netdict)

        return case

    def __str__(self):
        return f"Case object for test case {self.name}."

    def __repr__(self):
        return self.__str__()


def create(file_name, name, pn=None):
    """Creates a Testcase namedtuple for the case.

    Calculates the full path and gives upper cased name.
    """
    path = DATA_DIR / Path(f"pglib_opf_case{file_name}.mat")
    return TEST_CASE(path, name, pn)
