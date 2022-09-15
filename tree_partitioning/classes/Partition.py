from functools import lru_cache

import networkx as nx


class Partition:
    """
    A Partition object to store graph clusters.
    """

    def __init__(self, clusters: dict):
        self.clusters = clusters

    def __repr__(self):
        return str(self.clusters)

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, key):
        return self.clusters[key]

    @property
    def membership(self):
        return self._get_membership()

    @property
    def clusters_size(self):
        return [len(v) for v in self.clusters.values()]

    @lru_cache(maxsize=1)
    def _get_membership(self):
        """
        Return a dictionary that maps bus indices to cluster numbers.
        """
        return {
            bus: cluster for cluster, buses in self.clusters.items() for bus in buses
        }

    def is_partition(self, G):
        """
        Checks if the partition is indeed a partition of G.
        """
        return all(bus in G.nodes for bus in self.membership)

    def is_connected_clusters(self, G):
        """
        Checks if the clusters form connected subgraphs in G.
        """
        is_connected = nx.is_weakly_connected if nx.is_directed(G) else nx.is_connected

        return all(is_connected(G.subgraph(nodes)) for nodes in self.clusters.values())


def reduced_graph(G, P):
    """Compute the reduced graph of G given partition P.

    Args:
        G: NetworkX graph
        P: Partition object

    Returns:
        rg: NetworkX reduced graph, where the vertices are labeled the
            cluster u and the edges (u, v, k) where k is the i-th multiedge.

    """
    rg = nx.MultiGraph()
    Q = P.vertex2cluster()

    # Get all edges and add them to the reduced graph based on vertex idx
    for e in G.edges:
        i, j = e[0], e[1]
        # Only consider edges if they are given in the partition
        # Partitions are sometimes not "maximal" since we can prune bridge-blocks
        if i in Q and j in Q:
            u, v = Q[i], Q[j]
            # Do not allow loops
            if u != v:
                name = G.get_edge_data(*e)["name"]
                weight = G.get_edge_data(*e)["weight"]
                rg.add_edge(u, v, name=name, weight=weight)

    return rg
