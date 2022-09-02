import networkx as nx

from tree_partitioning.classes import ReducedGraph


def maximum_spanning_tree(G, partition, weight="weight"):
    """
    Find the maximum spanning tree using the passed-in edge weights.
    Return the total cost in terms of weight and the edges of the switched lines.
    """
    # MST only works on undirected graphs
    rg = ReducedGraph(G, partition).RG.to_undirected()
    neg = {(u, v, (e)): {"neg_weight": -G.edges[e][weight]} for (u, v, (e)) in rg.edges}
    nx.set_edge_attributes(rg, neg)

    T = nx.algorithms.minimum_spanning_tree(rg, weight="neg_weight")

    # Total power flow disruption is the sum of edges minus the weight of MST
    cost = abs(sum(nx.get_edge_attributes(rg, "neg_weight").values()))
    cost -= abs(sum(nx.get_edge_attributes(T, "neg_weight").values()))

    # Switched lines (not in T)
    lines = [e for (u, v, (e)) in rg.edges if (u, v, (e)) not in T.edges]

    return cost, lines
