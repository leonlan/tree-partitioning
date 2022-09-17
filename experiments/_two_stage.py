from time import perf_counter

from _sanity_check import _sanity_check
from _select import _select_line_switching, _select_partitioning


def _two_stage(case, generators, **kwargs):
    G = case.G

    # Partitioning stage
    start_partitioning = perf_counter()
    partition = _select_partitioning(G, generators, **kwargs)
    time_partitioning = perf_counter() - start_partitioning

    # Line switching stage
    start_line_switching = perf_counter()
    cost, lines = _select_line_switching(G, partition, **kwargs)
    time_line_switching = perf_counter() - start_line_switching

    # _sanity_check(G, generators, partition, lines)
    print("two_stage", len(generators), cost)

    return (0, 0)  # TODO
