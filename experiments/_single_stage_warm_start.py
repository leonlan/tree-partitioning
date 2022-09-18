from _sanity_check import _sanity_check
from _select import _select_line_switching

import tree_partitioning.milp.line_switching.maximum_congestion as milp_line_switching
import tree_partitioning.milp.partitioning as partitioning
import tree_partitioning.milp.tree_partitioning as single_stage
import tree_partitioning.milp.utils as model_utils


def _single_stage_warm_start(case, generators, **kwargs):
    """
    Use two-stage to warm-start single stage.
    """
    solver, options = kwargs["solver"], kwargs["options"]

    kwargs["options"].update({"TimeLimit": 30})

    G = case.G

    # Solve the two stage models
    part_model = partitioning.power_flow_disruption(G, generators, **kwargs)
    solver.solve(part_model, tee=False, options=options)

    partition = model_utils.get_partition(part_model)
    ls_model = milp_line_switching(G, partition, **kwargs)
    solver.solve(ls_model, tee=False, options=options)
    _, lines = ls_model.objective(), model_utils.get_inactive_lines(ls_model)

    _sanity_check(G, generators, partition, lines)

    kwargs["options"].update({"TimeLimit": 300})

    # Warm start the single stage model
    model = single_stage.maximum_congestion(G, generators, **kwargs)
    _warm_start(model, part_model, ls_model)
    solver.solve(model, tee=True, warmstart=True, options=options)

    partition = model_utils.get_partition(model)
    inactive_lines = model_utils.get_inactive_lines(model)

    _sanity_check(case.G, generators, partition, inactive_lines)
    print("single-stage (warm)", len(generators), model.objective())


def _warm_start(tp_model, part_model, ls_model):
    """
    Warm start the new model using the (solved) old model.
    """
    # Base partitioning
    for k, v in part_model.assign_bus.items():
        tp_model.assign_bus[k] = v()

    for k, v in part_model.assign_line.items():
        tp_model.assign_line[k] = v()

    # Base line switching
    # Active lines in ls_model are only active cross edges
    for k, v in ls_model.active_line.items():
        tp_model.active_line[k] = v() if v() is not None else 1

    for k, v in ls_model.active_line.items():
        tp_model.active_cross_edge[k] = v() if v() is not None else 0

    # Max. cong. line switching
    tp_model.gamma = ls_model.gamma()

    for k, v in ls_model.flow.items():
        tp_model.flow[k] = v()

    for k, v in ls_model.theta.items():
        tp_model.theta[k] = v()
