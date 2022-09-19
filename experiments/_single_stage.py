import time

from _sanity_check import _sanity_check

import tree_partitioning.milp.utils as model_utils


def _single_stage(case, generators, tree_partitioning_alg, **kwargs):
    """
    Solve TP-PFD considering transient stability.
    """
    model = tree_partitioning_alg(case.G, generators, **kwargs)
    start = time.perf_counter()

    solver, options = kwargs["solver"], kwargs["options"]
    print(options)
    solver.solve(model, tee=True, options=options)

    partition = model_utils.get_partition(model)
    inactive_lines = model_utils.get_inactive_lines(model)

    _sanity_check(case.G, generators, partition, inactive_lines)
    print("single-stage", len(generators), model.objective())

    return partition, inactive_lines, time.perf_counter() - start
