import networkx as nx

from tree_partitioning.classes import ReducedGraph


def maximum_spanning_tree(G, partition, weight="weight", **kwargs):
    """
    Finds the maximum spanning tree using the passed-in edge weights to
    determine the line switching actions. The edges that do not belong
    to the MST are switched off.

    Return the cost (sum of weights) and edge indices of the switched lines.
    """
    # Compute the reduced graph (undirected for MST)
    rg = ReducedGraph(G, partition).RG.to_undirected()

    # Consider the negative edge weights
    neg = {(u, v, (e)): {"neg_weight": -G.edges[e][weight]} for (u, v, (e)) in rg.edges}
    nx.set_edge_attributes(rg, neg)

    # Find the minimum spanning tree, i.e., maximum w.r.t. weight
    MST = nx.algorithms.minimum_spanning_tree(rg, weight="neg_weight")

    # Total power flow disruption is the sum of edges minus the weight of MST
    cost = abs(sum(nx.get_edge_attributes(rg, "neg_weight").values()))
    cost -= abs(sum(nx.get_edge_attributes(MST, "neg_weight").values()))

    # Switched lines (those not in MST)
    lines = [e for (u, v, (e)) in rg.edges if (u, v, (e)) not in MST.edges]

    return cost, lines
