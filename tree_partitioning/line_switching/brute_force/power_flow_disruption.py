import networkx as nx

from ._base_brute_force import _base_brute_force


def power_flow_disruption(G, partition, **kwargs):
    """
    Solve the Optimal Line Switching problem minimizing power flow disruption
    using the brute force approach.
    """

    def compute_pfd(G, switched_lines, **kwargs):
        """
        Compute the power flow disruption of the switched lines.
        """
        weight = kwargs["weight"] if "weight" in kwargs else "weight"
        weights = nx.get_edge_attributes(G, weight)
        return sum(weights[line] for line in switched_lines)

    return _base_brute_force(G, partition, compute_pfd, **kwargs)
