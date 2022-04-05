import networkx as nx

from tree_partitioning.classes import Case, Partition, ReducedGraph


class TestReducedGraphComplete:
    G = nx.complete_graph(10, create_using=nx.MultiDiGraph)
    partition = Partition({0: [0, 1, 2,], 1: [3, 4], 2: [5, 6, 7, 8, 9]})
    reduced_graph = ReducedGraph(G, partition)

    def test_is_tree(self):
        assert not self.reduced_graph.is_tree()


class TestReducedGraphTree:
    G = nx.random_tree(100, create_using=nx.MultiDiGraph)
    partition = Partition({v: [v] for v in G.nodes})
    reduced_graph = ReducedGraph(G, partition)

    def test_is_tree(self):
        assert self.reduced_graph.is_tree()

    def test_cross_edges(self):
        assert len(self.reduced_graph.cross_edges) == len(self.G.edges)
