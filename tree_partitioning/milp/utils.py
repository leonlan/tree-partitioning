from collections import defaultdict

from tree_partitioning.classes import Partition


def get_partition(model):
    partition = defaultdict(list)

    for x, value in model.assign_bus.items():
        # NOTE Rounding values because solver returns non-integral vals
        if round(value()) == 1:
            bus, cluster = x
            partition[cluster].append(bus)

    return Partition(partition)


def get_cluster_sizes(model):
    cluster_sizes = defaultdict(int)

    for (_, cluster), val in model.assign_bus.items():
        if round(val()) == 1:
            cluster_sizes[cluster] += 1

    return list(cluster_sizes.values())


def get_inactive_lines(model):
    lines = []

    for line, val in model.active_line.items():
        if val() is not None and round(val()) == 0:
            lines.append(line)

    return lines


def get_active_lines(model):
    lines = []

    for line, val in model.active_line.items():
        if round(val()) == 1:
            lines.append(line)

    return lines


def get_cross_edges(model):
    lines = defaultdict(int)

    for (*line, cluster), val in model.assign_line.items():
        lines[tuple(line)] += val()

    return [line for line, val in lines.items() if round(val) == 0]


def warm_start(ref_model, target_model):
    for key, value in ref_model.assign_bus.items():
        target_model.assign_bus[key] = round(value())

    for key, value in ref_model.assign_line.items():
        target_model.assign_line[key] = round(value())

    for key, value in ref_model.active_cross_edge.items():
        target_model.active_cross_edge[key] = round(value())

    for key, value in ref_model.active_line.items():
        target_model.active_line[key] = round(value())

    for key, value in ref_model.commodity_flow.items():
        target_model.commodity_flow[key] = round(value())
