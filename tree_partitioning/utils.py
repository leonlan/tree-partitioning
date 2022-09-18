import re

import networkx as nx

from tree_partitioning.constants import _EPS


def name2size(name: str) -> int:
    """
    Extracts the instance size (i.e., num clients) from the instance name.
    """
    return int(re.search(r"_case(\d+)", name).group(1))


def maximum_congestion(G):
    """
    Compute the maximum congestion of the graph G.
    """
    weights = nx.get_edge_attributes(G, "f")
    capacities = nx.get_edge_attributes(G, "c")

    max_congestion = 0
    for line in weights.keys():
        max_congestion = max(max_congestion, abs(weights[line]) / capacities[line])

    return max_congestion


def remove_lines(G, lines):
    """
    Remove the passed-in lines from G. Return all connected components.
    """
    H = G.copy()
    H.remove_edges_from(lines)
    return [H.subgraph(c).copy() for c in nx.weakly_connected_components(H)]


def congested_lines(G):
    """
    Return the congested lines from G.
    """
    weights = nx.get_edge_attributes(G, "f")
    capacities = nx.get_edge_attributes(G, "c")

    congested = []

    for line in weights.keys():
        if abs(weights[line]) / capacities[line] > 1 + _EPS:
            congested.append(line)

    return congested


def compute_cross_edges(G, partition):
    """
    Return the cross edges defined on G and partition.
    """
    membership = partition.membership
    edges = []

    for (i, j, idx) in G.edges:
        if i in membership and j in membership:
            u, v = membership[i], membership[j]
            if u != v:
                edges.append((u, v, (i, j, idx)))

    return edges
