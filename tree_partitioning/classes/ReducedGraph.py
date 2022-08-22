from functools import lru_cache

import networkx as nx

from .Partition import Partition


class ReducedGraph:
    def __init__(self, G: nx.MultiDiGraph, partition: Partition):
        self.G = G
        self.partition = partition
        self.clusters = self.partition.clusters.keys()
        self.cross_edges = self._compute_cross_edges()
        self.RG = nx.MultiDiGraph()
        self.RG.add_edges_from(self.cross_edges)

    def is_tree(self):
        return (
            nx.is_weakly_connected(self.RG)
            and len(self.cross_edges) == len(self.clusters) - 1
        )

    def _compute_cross_edges(self):
        # For each line (i, j, k), create a line (u, v, line_name)
        # if u, v belong to opposite clusters
        cross_edges = []

        for i, j, k in self.G.edges:
            u = self.partition.membership[i]
            v = self.partition.membership[j]

            if u != v:
                cross_edges.append((u, v, (i, j, k)))

        return cross_edges

    @property
    def cross_edge_to_line(self):
        return self._cross_edge_to_line()

    @lru_cache(maxsize=1)
    def _cross_edge_to_line(self):
        mapper = dict()
        for u, v, (i, j, k) in self.cross_edges:
            mapper[(u, v, (i, j, k))] = (i, j, k)
        return mapper

    def incidence(self, u: int):
        """Returns the edges incident to cluster u."""
        return self._incidence()[u]

    @lru_cache(maxsize=1)
    def _incidence(self):
        L = {}
        for u in self.clusters:
            incidences = []
            for v, w, line in self.cross_edges:
                if u == v:
                    sign = -1  # Outgoing
                elif u == w:
                    sign = 1  # Incoming
                else:
                    continue

                incidences.append(((v, w, line), sign))

            L[u] = incidences

        return L
