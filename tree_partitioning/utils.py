import networkx as nx


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
