from tree_partitioning.dcpf import dcpf
from tree_partitioning.utils import maximum_congestion as compute_max_cong

from ._base_brute_force import _base_brute_force


def maximum_congestion(G, partition, **kwargs):
    """
    Solve the Optimal Line Switching problem minimizing maximum congestion
    using the brute force approach.
    """

    def compute_congestion(G, switched_lines):
        """
        Compute the maximum congestion on the post switching graph G \ E.
        """
        H = G.copy()
        H.remove_edges_from(switched_lines)
        return compute_max_cong(dcpf(H, in_place=True))

    return _base_brute_force(G, partition, compute_congestion, **kwargs)
