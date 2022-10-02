import time

from _sanity_check import _sanity_check

import tree_partitioning.milp.line_switching.maximum_congestion as milp_line_switching
import tree_partitioning.milp.partitioning as partitioning
import tree_partitioning.milp.tree_partitioning as single_stage
import tree_partitioning.milp.utils as model_utils


def _single_stage_warm_start(case, generators, **kwargs):
    """
    Use two-stage to warm-start single stage.
    """
    solver, options = kwargs["solver"], kwargs["options"]

    # Halven time limit for two-stage algorithm
    kwargs = kwargs.copy()
    kwargs["options"] = {"TimeLimit": kwargs["options"]["TimeLimit"] / 2}

    G = case.G
    start = time.perf_counter()

    # Solve the two stage models
    part_model = partitioning.power_flow_disruption(G, generators, **kwargs)
    solver.solve(part_model, tee=False, options=options)

    partition = model_utils.get_partition(part_model)
    ls_model = milp_line_switching(G, partition, **kwargs)
    solver.solve(ls_model, tee=False, options=options)
    _, lines = ls_model.objective(), model_utils.get_inactive_lines(ls_model)

    _sanity_check(G, generators, partition, lines)

    # Double time limit back to original for single stage
    kwargs["options"].update({"TimeLimit": kwargs["options"]["TimeLimit"] * 2})

    # Warm start the single stage model
    model = single_stage.maximum_congestion(G, generators, **kwargs)
    _warm_start(model, part_model, ls_model)
    solver.solve(model, tee=False, warmstart=True, options=options)

    partition = model_utils.get_partition(model)
    inactive_lines = model_utils.get_inactive_lines(model)

    _sanity_check(case.G, generators, partition, inactive_lines)
    print("single-stage (warm)", len(generators), model.objective())

    return partition, inactive_lines, time.perf_counter() - start


def _warm_start(tp_model, part_model, ls_model):
    """
    Warm start the new model using the (solved) old model.
    """
    # Base partitioning
    for k, v in part_model.assign_bus.items():
        tp_model.assign_bus[k] = round(v())

    for k, v in part_model.assign_line.items():
        tp_model.assign_line[k] = round(v())

    # Base line switching
    # Active lines in ls_model are only active cross edges
    for k, v in ls_model.active_line.items():
        tp_model.active_line[k] = round(v()) if v() is not None else 1

    for k, v in ls_model.active_line.items():
        tp_model.active_cross_edge[k] = round(v()) if v() is not None else 0

    # Max. cong. line switching
    tp_model.gamma = ls_model.gamma()

    for k, v in ls_model.flow.items():
        tp_model.flow[k] = v()

    for k, v in ls_model.theta.items():
        tp_model.theta[k] = v()
