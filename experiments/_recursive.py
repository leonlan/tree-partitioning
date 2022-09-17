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

    solver, options = kwargs["solver"], kwargs["options"]

    final_partition = [list(G.nodes)]
    final_lines = []
    cluster = final_partition[0]

    total = 0

    for _ in range(len(generators) - 1):
        # Partitioning stage
        new_partition = _select_partitioning(G.subgraph(cluster), generators, **kwargs)

        assert new_partition.is_connected_clusters(G.subgraph(cluster))

        # Line switching stage
        cost, lines = _select_line_switching(
            G.subgraph(cluster), new_partition, **kwargs
        )  # REVIEW give subgraph or full graph to line switching
        final_lines += lines
        total += cost

        # Update the final partition, switch off lines, rerun dcpf
        final_partition = _update(final_partition, new_partition)
        G.remove_edges_from(lines)
        G, _ = dcpf(G)  # this is needed for max congestion problem

        # Find the cluster with the most generator groups
        num_groups = lambda cl: sum(gens[0] in cl for idx, gens in generators.items())
        cluster = max([cl for cl in final_partition], key=num_groups)

        groups = {
            idx: gens
            for idx, gens in generators.items()
            if all(g in cluster for g in gens)
        }

    print("multi_stage", len(generators), total)
    _sanity_check(
        case.G, generators, Partition(dict(enumerate(final_partition))), final_lines
    )  # case.G is the original, unmodified network

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
