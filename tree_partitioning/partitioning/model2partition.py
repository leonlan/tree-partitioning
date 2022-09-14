from collections import defaultdict

from tree_partitioning.classes import Partition


def model2partition(model):
    partition = defaultdict(list)

    for x, value in model.assign_bus.items():
        # NOTE Rounding values because solver returns non-integral vals
        if round(value()) == 1:
            bus, cluster = x
            partition[cluster].append(bus)

    return Partition(partition)
