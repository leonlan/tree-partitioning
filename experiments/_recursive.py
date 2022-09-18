import networkx as nx
from _sanity_check import _sanity_check
from _select import _select_line_switching, _select_partitioning

from tree_partitioning.classes import Partition
from tree_partitioning.dcpf import dcpf


def _recursive(case, generators, **kwargs):
    """
    Solve the tree partitioning problem considering transient stability using
    the two-stage MILP+MST approach.
    """
    G = case.G.copy()
    groups = generators

    final_lines = []
    final_partition = [list(G.nodes)]
    cluster = final_partition[0]

    total = []

    for k in range(len(generators) - 1):
        # Partitioning stage
        new_partition = _select_partitioning(
            G.subgraph(cluster), groups, recursive=True, **kwargs
        )

        assert new_partition.is_connected_clusters(G.subgraph(cluster))

        # Line switching stage
        cost, lines = _select_line_switching(G, new_partition, **kwargs)
        final_lines += lines
        total.append(cost)

        # Update the final partition, switch off lines, rerun dcpf
        final_partition = _update(final_partition, new_partition)
        G.remove_edges_from(lines)
        assert len(final_partition) == k + 2

        # Find the cluster with the most generator groups
        num_groups = lambda cl: (gens[0] in cl for idx, gens in generators.items())
        cluster = max([cl for cl in final_partition], key=lambda x: sum(num_groups(x)))

        groups = {
            idx: gens
            for idx, gens in generators.items()
            if all(g in cluster for g in gens)
        }

    _sanity_check(
        case.G, generators, Partition(dict(enumerate(final_partition))), final_lines
    )

    print("multi_stage", len(generators), total, sum(total), max(total))

    return 0, 0  # TODO


def _update(partition: list, subpartition: Partition):
    # Update the final partition with the newly found partition
    nodes = [v for cluster in subpartition.clusters.values() for v in cluster]

    # Remove all clusters that have nodes in the subpartition
    partition = [cluster for cluster in partition if cluster[0] not in nodes]

    # Add the new subpartition clusters
    for cluster in subpartition.clusters.values():
        partition.append(cluster)

    return partition
