#!/usr/bin/env ipython
import networkx as nx


def _G_from_netdict(netdict):
    """
    Create a networkx graph G from a netdict representation.
    """
    G = nx.MultiDiGraph()

    G.add_edges_from(
        [
            (data["from_bus"], data["to_bus"], data)
            for i, data in netdict["lines"].items()
        ]
    )

    nx.set_node_attributes(G, 0, name="community")
    nx.set_node_attributes(G, netdict["buses"])

    return G
