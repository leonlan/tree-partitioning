import functools
from pathlib import Path

from ._nx_utils import _G_from_netdict
from ._pp_utils import _load_pp_case, _netdict_from_pp_net


class Case:
    """
    Case instance of a single power network.

    This class contains multiple representation of the network, among others:
    - a pandapower power network (net)
    - a networkx graph (G)
    - a generic dictionary representation of the network (netdict)

    These multiple representations are helpful as some computations are
    implemented for e.g. pandapower only.

    """

    def __init__(self, name, net, netdict, G):
        self._name = name
        self._net = net
        self._netdict = netdict
        self._G = G

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
        path = Path(path)

        name = path.stem
        net = _load_pp_case(path, opf_init, ac)
        netdict = _netdict_from_pp_net(net, merge_lines)
        G = _G_from_netdict(netdict)

        return cls(name, net, netdict, G)

    def __str__(self):
        return f"Case object for test case {self.name}."

    def __repr__(self):
        return self.__str__()
