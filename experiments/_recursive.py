import time

from _sanity_check import _sanity_check
from _select import _select_line_switching, _select_partitioning

from tree_partitioning.classes import Partition


def _recursive(case, generators, **kwargs):
    """
    Solve the tree partitioning problem considering transient stability using
    the two-stage MILP+MST approach.
    """
    G = case.G.copy()
    groups = generators

    # Shorten time limit for recursive algorithm
    kwargs = kwargs.copy()
    kwargs["options"] = {
        "TimeLimit": kwargs["options"]["TimeLimit"] / (len(groups) - 1)
    }

    final_lines = []
    final_partition = [list(G.nodes)]
    cluster = final_partition[0]

    total = []

    start = time.perf_counter()
    for k in range(len(generators) - 1):
        # Split the largest cluster into two parts
        new_partition = _select_partitioning(
            G.subgraph(cluster), groups, recursive=True, **kwargs
        )

        # Find the corresponding line switching actions
        cost, lines = _select_line_switching(G, new_partition, **kwargs)
        final_lines += lines
        total.append(cost)

        # Update the final partition, switch off the lines
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

    final_partition = Partition(dict(enumerate(final_partition)))

    _sanity_check(case.G, generators, final_partition, final_lines)
    print("multi_stage", len(generators), total, sum(total), max(total))

    return final_partition, final_lines, time.perf_counter() - start


def _update(partition: list, subpartition: Partition):
    # Update the final partition with the newly found partition
    nodes = [v for cluster in subpartition.clusters.values() for v in cluster]

    # Remove all clusters that have nodes in the subpartition
    partition = [cluster for cluster in partition if cluster[0] not in nodes]

    # Add the new subpartition clusters
    for cluster in subpartition.clusters.values():
        partition.append(cluster)

    return partition
