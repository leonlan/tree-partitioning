from collections import defaultdict
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

    def is_bbd(self, G):
        """
        Verifies if the graph is a BBD w.r.t. the partition.
        The definition is loose here: we only check if the reduced graph GP
        has a spanning tree.
        """
        H = reduced_graph(G, self.partition)
        return nx.is_connected(H) and len(H.edges) == len(P.keys()) - 1

    @staticmethod
    def extend(partition, G):
        """
        Extends a partition to include the entire graph.

        In some functions, the partitions are computed only on the largest non-trivial
        bridge-block. This leaves out the smaller bridge-blocks, but we need those
        for initialization in the MILP.

        The idea is to run a shortest path from each excluded node. The first node
        it reaches gives the corresponding cluster number it belongs to. Note that
        there exists a unique node with minimum length due to the existence of a
        bridge.
        """
        dist = dict(nx.all_pairs_shortest_path_length(G))

        Q = defaultdict(list)
        v2c = partition.membership
        V = v2c.keys()
        for u in G.nodes:
            if u not in V:
                min_dist = 10000
                min_node = None
                for v in V:
                    if dist[u][v] < min_dist:
                        min_dist = dist[u][v]
                        min_node = v

                # Get the cluster of the closest node
                r = v2c[min_node]
                Q[r].append(u)

        # Add the new nodes to the original partition
        new_partition = copy.copy(partition)
        for cluster, nodes in Q.items():
            new_partition[cluster].extend(nodes)

        assert new_partition.is_partition(G)

        return new_partition


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
