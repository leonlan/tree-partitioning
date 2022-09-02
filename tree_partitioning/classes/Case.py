import functools
import itertools
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd

from ._nx_utils import _G_from_netdict
from ._pp_utils import _load_pp_case, _netdict_from_pp_net


class Case:
    """
    Case instance of a single power network.

    This class contains multiple representation of the network, among others:
    - a pandapower power network (net)
    - a generic dictionary representation of the network buses and lines (netdict)
    - a networkx graph (G)

    These multiple representations are helpful as some computations are implemented
    for e.g. pandapower only.

    """

    _name: str
    _net: ...
    _netdict: dict
    _G: nx.MultiDiGraph

    @property
    def name(self) -> str:
        return self._name

    @property
    def net(self):
        return self._net

    @property
    def G(self):
        return self._G

    @classmethod
    @functools.lru_cache(maxsize=None)
    def from_file(cls, path, merge_lines=False, opf_init=True, ac=False):
        case = cls()
        case._name = str(path).split("pglib_opf_")[-1].split(".mat")[0]
        case._net = _load_pp_case(path, opf_init, ac)
        case._netdict = _netdict_from_pp_net(case._net, merge_lines)
        case._G = _G_from_netdict(case._netdict)

        return case

    def __str__(self):
        return f"Case object for test case {self.name}."

    def __repr__(self):
        return self.__str__()
