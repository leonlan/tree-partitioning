import networkx as nx


class Partition:
    """
    A Partition object to store graph clusters.
    """

    def __init__(self, clusters: dict):
        self.clusters = clusters
        self.membership = self._get_membership()

    def __repr__(self):
        return str(self.clusters)

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, key):
        return self.clusters[key]

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

    @classmethod
    def from_clustering(cls, clustering):
        """
        Load Partition from igraph.Clustering and/or igraph.VertexClustering.
        """
        return cls({cluster: buses for cluster, buses in enumerate(clustering)})

    def is_connected_clusters(self, G):
        """
        Checks if the clusters form connected subgraphs in G.
        """
        is_connected = nx.is_weakly_connected if nx.is_directed(G) else nx.is_connected

        return all(is_connected(G.subgraph(nodes)) for nodes in self.clusters.values())
