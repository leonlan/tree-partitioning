class Partition(dict):
    """
    A Partition object to store graph clusters.

    dict
    """

    def __init__(self, clusters: dict):
        self.clusters = clusters
        self.bus2cluster = self._bus_to_cluster()

    def __repr__(self):
        return str(self.clusters)

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, key):
        return self.clusters[key]

    def _bus_to_cluster(self):
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
        return all(bus in G.nodes for bus in self.bus2cluster)
