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
    kwargs["options"] = {
        "TimeLimit": kwargs["options"]["TimeLimit"] / 2,
        # "FeasibilityTol": 0.01,
    }

    G = case.G
    start = time.perf_counter()

    model = single_stage.maximum_congestion(G, generators, **kwargs)

    # Warm start with PFD
    # # Double time limit back to original for single stage
    # kwargs["options"].update({"TimeLimit": kwargs["options"]["TimeLimit"] * 2})

    # # Warm start the single stage model
    # pfd_model = single_stage.power_flow_disruption(G, generators, **kwargs)
    # solver.solve(pfd_model, tee=True, options=options)
    # _warm_start_pfd(model, pfd_model)

    # Warmstart with two-stage MC
    # Solve the two stage models
    part_model = partitioning.power_flow_disruption(G, generators, **kwargs)
    solver.solve(part_model, tee=False, options=options)

    partition = model_utils.get_partition(part_model)
    ls_model = milp_line_switching(G, partition, **kwargs)
    solver.solve(ls_model, tee=False, options=options)
    _, lines = ls_model.objective(), model_utils.get_inactive_lines(ls_model)
    _warm_start(model, part_model, ls_model)

    _sanity_check(G, generators, partition, lines)

    solver.solve(model, tee=True, warmstart=True, options=options)

    partition = model_utils.get_partition(model)
    inactive_lines = model_utils.get_inactive_lines(model)

    _sanity_check(case.G, generators, partition, inactive_lines)
    print("single-stage (warm)", len(generators), model.objective())

    return partition, inactive_lines, time.perf_counter() - start


def _warm_start_pfd(mc_model, pfd_model):
    """
    Warm start the new model using the pfd model.
    """
    # Base partitioning
    for k, v in pfd_model.assign_bus.items():
        mc_model.assign_bus[k].value = round(v())
        # mc_model.assign_bus[k].fixed = True

    for k, v in pfd_model.assign_line.items():
        mc_model.assign_line[k].value = round(v())
        # mc_model.assign_line[k].fixed = True

    # Base line switching
    # Active lines in pfd_model are only active cross edges
    for k, v in pfd_model.active_line.items():
        mc_model.active_line[k].value = round(v()) if v() is not None else 1
        # mc_model.active_line[k].fixed = True
        # pass

    for k, v in pfd_model.active_cross_edge.items():
        mc_model.active_cross_edge[k].value = round(v()) if v() is not None else 0
        # mc_model.active_cross_edge[k].fixed = True
        # pass


def _warm_start(tp_model, part_model, ls_model):
    """
    Warm start the new model using the (solved) old model.
    """
    # Base partitioning
    for k, v in part_model.assign_bus.items():
        tp_model.assign_bus[k].value = round(v())
        # tp_model.assign_bus[k].fixed = True

    for k, v in part_model.assign_line.items():
        tp_model.assign_line[k].value = round(v())
        # tp_model.assign_line[k].fixed = True

    # Base line switching
    # Active lines in ls_model are only active cross edges
    for k, v in ls_model.active_line.items():
        tp_model.active_line[k].value = round(v()) if v() is not None else 1
        # tp_model.active_line[k].fixed = True
        # pass

    for k, v in ls_model.active_line.items():
        tp_model.active_cross_edge[k].value = round(v()) if v() is not None else 0
        # tp_model.active_cross_edge[k].fixed = True
        # pass

    # Max. cong. line switching
    tp_model.gamma = ls_model.gamma()

    for k, v in ls_model.flow.items():
        tp_model.flow[k].value = v()

    for k, v in ls_model.theta.items():
        tp_model.theta[k].value = v()
