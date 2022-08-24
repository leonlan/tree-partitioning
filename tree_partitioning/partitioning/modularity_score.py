from __future__ import annotations

from itertools import combinations

from tree_partitioning.classes import Case


def modularity_score(partition: Partition, normalized=False):
    """Compute the normalized modularity score of the partition."""
    lines = Case().netdict["lines"]

    M = 1 / 2 * sum(data["f"] for line, data in lines.items())

    def flow_balance(cluster: list[int]):
        total = 0

        for i, j in combinations(cluster, 2):
            try:
                f_ij = sum(
                    data["f"] for line, data in lines.items() if i in line and j in line
                )

                # If the line doesn't exist, then stop
                if f_ij == 0:
                    break

                F_i = (
                    1 / 2 * sum(data["f"] for line, data in lines.items() if i in line)
                )
                F_j = (
                    1 / 2 * sum(data["f"] for line, data in lines.items() if j in line)
                )
                total += f_ij - (F_i * F_j / 2 / M)

            except KeyError:
                pass

        if normalized:
            # Normalization by outgoing flows
            normalization = sum(
                sum(data["f"] for line, data in lines.items() if r in line)
                for r in cluster
            )

            return total / normalization

        else:
            return total

    return (0.5 / M) * sum(
        flow_balance(cluster) for cluster in partition.clusters.values()
    )
