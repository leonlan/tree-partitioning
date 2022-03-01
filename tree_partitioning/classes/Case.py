import itertools
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ._pp_utils import _load_pp_case, _netdict_from_pp_net
from ._nx_utils import _G_from_netdict
from ._ig_utils import _igg_from_netdict


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            up = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = up

        return cls._instances[cls]

    def clear(cls):
        try:
            del Singleton._instances[cls]
        except KeyError:
            pass


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

    def __init__(self, path, merge_lines=True, opf_init=True, ac=False):
        self.net = _load_pp_case(path, opf_init, ac)
        self.netdict = _netdict_from_pp_net(self.net, merge_lines)
        self.G = _G_from_netdict(self.netdict)
        self.igg = _igg_from_netdict(self.netdict)

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
